"""
Agents manage the exploration and game playing of multiple trees,
managing the inference steps and provisioning the outputs back to leaf nodes.

An agent will play a target number of games of patterns according to the current network iteration.

Extensions:
schedule for the breadth of the exploration,

schedule for number of exploration steps with depth of the node

schedule for random play

so in particular, we are running N trees in parallel. The leaf nodes returned at each
step will be at different depths. We don't need to apply the tensor eval until they all need it.

So do I just keep going for each tree until it requests an eval?

NOTE

Say for node 1. Each node wants to search for

todo

NEED to sit down and write out where the responsibilities should sit...
For the exploring, controlling depth search, controlling rtandoimness etc.

Some part of it needs to be controlled in bulk, because otherwise we won't get the efficiency of bulking
our inference together into massive batches.

But it also feels hard to control what each node is doing in parallel, when some will be exploring with different
number of explore steps, some will be using different breadth, some will be using inference or not etc.

also todo; if you see a next state winning move, that NEEDS to be treated as better than anything else.


2. way to store the terminal games for the first bit! No reason to ever chuck away 1-move-to-win states.
2. redraft how the responsibilities are shared
3. find a way to schedule random explorations (I think random look aheads should be truly random so no need for
            explorations)


okay so...

- don't create tensor state if you don't need it? Can always create on back up if we want it for the replay buffer.


AGENT:
1. check if root node has a one move win and stop exploring.
2. Store replay buffer and save to disk.

Note we do not want to store tensor states for the games, as

"""
import torch
from typing import Optional

from game import Patterns
from mcts.mcts import Node, Tree


class Agent:
    """ Each agent will run num_trees search trees in series. However, the
    inference step can occur in one go. That is, all N inferences can be done in parallel.

    Given that the search tree steps are very quick versus the inference, this helps
    to bring the inference bottleneck time up to the search tree time.
    """
    def __init__(self,
                 agent_id: str,
                 network: torch.nn.Module,
                 device: torch.device,
                 num_trees: int = 20,
                 target_games: int = 1000,
                 selection_temperature: float = 1.0,
                 topn: int = 4, # breadth restriction parameters
                 randn: int = 4,
                 save_depth: Optional[int] = None,

                 # Schedule that dictates how many explore steps a root node requires at a given depth:
                 # This is a TREE quantity, not a NODE quantity. If 0 is given for explore steps,
                 # random exploration is used instead!
                 explore_steps_schedule: Optional[list[tuple]] = None,
                 ):
        # unique string for this agent, for the cpu driver to provision
        self.agent_id = agent_id

        # put the network into eval mode, move it to device:
        self.network = network
        self.network.eval()

        self.device = device
        self.network.eval()
        self.network.to(self.device)

        # this many trees will be run in series:
        self.num_trees = num_trees
        self.selection_temperature = selection_temperature

        # track number of games completed:
        self.num_completed = 0
        self.target_games = target_games
        self.completed_games = []

        self.topn = topn
        self.randn = randn

        self.save_depth = save_depth
        self.explore_steps_schedule = explore_steps_schedule

        # create stash of games equal to number of trees + target games,
        # to eval all at once:
        self.root_nodes: list[Node] = []
        self.create_root_nodes()

        # create the trees and populate with some pre-generated root nodes:
        self.trees = []

        for _gameit in range(num_trees):
            # take a game:
            _rnod = self.root_nodes.pop()

            self.trees.append(
                Tree(
                    root_node=_rnod,
                    tree_id=f"{self.agent_id}_{_gameit}",
                    breadth_restriction=self.topn,
                    random_restriction=self.randn,
                    schedule=self.explore_steps_schedule,
                )
            )

    def create_root_nodes(self) -> None:
        """ create the necessary number of games, evaluate the tensor states,
        and provision the full policy to each.

        Assign the schedules to each at this point.
        """

        print(f"Generating initial games:")
        tensor_states = []

        for _it in range(self.num_trees + self.target_games):
            # create new game:
            new_game = Patterns()

            # assign new game to root node, along with breadth restriction params:
            new_nod = Node(
                game=new_game,
                breadth_restriction=self.topn,
                random_restriction=self.randn,
            )

            # create the NN tensor state:
            new_nod.create_tensor_state()
            self.root_nodes.append(new_nod)

            # store for batch eval:
            tensor_states.append(new_nod.tensor_state)

        print(f"Evaluating tensor states...")
        # size is (num_trees + target_games, 102, 8 8)
        tensor_stack = torch.stack(tensor_states)

        # two head inference results:
        with torch.inference_mode():
            value_stack, policy_stack = self.network(tensor_stack.float().to(self.device,
                                                                             non_blocking=True))

        value_stack = value_stack.to("cpu", non_blocking=True).numpy()
        policy_stack = policy_stack.to("cpu", non_blocking=True).numpy()

        print(f"Provisioning inference to root nodes...")

        # provision the results to the root nodes:
        for _nod, _val, _pol in zip(self.root_nodes, value_stack, policy_stack):
            # value score is normalized in the network to be between -1 and 1:
            _nod.value_score = _val

            # policy cannot be restricted just yet, as the game action space is not yet known:
            _nod.full_policy = _pol

            # set visit count to avoid the expand step that seeks inference:
            _nod.visit_count = 1

    def run_games(self) -> None:
        """ run exploration steps, manage inference as per our plan
        """
        num_completed = 0
        while self.num_completed < self.target_games:
            # expand the leaf nodes, determine which require inference, which are done exploring and wish to step (after this step!)
            # exploration happens according to the tree schedule, for number of root explorations and for depth whether to
            # choose a node randomly or according to the policy and exploration policy stuff. Big speed up if just random!
            all_leaf_nodes, ready_trees, inference_nodes, tensor_states = self.explore()

            # calculate the inference, assign it to the relevant nodes:
            self.assign_inference(tensor_states, inference_nodes)

            # back-propagate from each tree leaf node.
            for _tree, _node in zip(self.trees, all_leaf_nodes):
                _tree.back_propagate(_node)

            # for those nodes that want to step, step them, check if terminal, and reset
            self.step_trees(ready_trees)

            if self.num_completed != num_completed:
                num_completed = self.num_completed
                print(f"{self.num_completed} games have been completed!")

    def step_trees(self, ready_trees: list[Tree]) -> None:
        """ iterate over the trees that are ready to step...
        """
        for _tree in ready_trees:
            action_argument = _tree.choose_action_argument()
            _tree.step(action_argument)

            if _tree.root_node.result is not None:
                _game = _tree.get_replay_game(save_depth=self.save_depth)

                new_root_node = self.root_nodes.pop()
                _tree.reset(new_root_node)

                self.num_completed += 1
                self.completed_games.append(_game)

    def explore(self) -> tuple[list[Node], list[Tree], list[Node], list[torch.tensor]]:
        """ explore each tree, and return a list of all the leaf nodes"""
        all_nodes = []
        inference_nodes = []
        tensor_states = []
        ready_trees = []

        for _tree in self.trees:
            # flow to leaf node following puct scores, or randomly:
            leaf_node = _tree.get_leaf_node()

            # expand the leaf, either return yourself if you haven't been seen before, or create children and
            # return one at random otherwise. Set a flag to detemine if you need inference:
            leaf_node = leaf_node.expand()

            all_nodes.append(leaf_node)

            # if the root node is sufficiently explored, or a mate in 1 detected:
            if _tree.is_step_ready:
                ready_trees.append(_tree)

            # Determine which require inference:
            # todo figure out random works for this. No tensor state requires it:
            if not leaf_node.tensor_state:
                inference_nodes.append(leaf_node)
                leaf_node.create_tensor_state()
                tensor_states.append(leaf_node.tensor_state)

        return all_nodes, ready_trees, inference_nodes, tensor_states

    def assign_inference(self, tensor_states: list[torch.tensor], inference_nodes: list[Node]) -> None:
        """ Inference step!
        """
        tensor_stack = torch.stack(tensor_states)

        # two head inference results:
        with torch.inference_mode():
            value_stack, policy_stack = self.network(tensor_stack.float().to(self.device, non_blocking=True))

        value_stack = value_stack.to("cpu", non_blocking=True).numpy()
        policy_stack = policy_stack.to("cpu", non_blocking=True).numpy()

        # provision the results of the value and policy to each leaf node:
        for _leaf, _val, _pol in zip(inference_nodes, value_stack, policy_stack):
            # value score is normalized in the network to be between -1 and 1:
            _leaf.value_score = _val

            # policy cannot be restricted just yet, as the game action space is not yet known:
            _leaf.full_policy = _pol

    #
    # def run_games(self) -> None:
    #     # while more games are required for quota:
    #     count = 0
    #     while self.num_completed < self.target_games:
    #         # explore each tree for required number of moves:
    #         for _ in range(self.explore_steps):
    #             self.random_explore()
    #
    #         # select an action for each tree root node, reset any trees that
    #         # are completed:
    #         self.step_trees()
    #         count += 1
    #         print(f"move number {count} completed:")
    #         print(f"Completed games: {self.num_completed}")



    #
    # def step_trees(self) -> None:
    #     """ After the exploration steps have completed, step each tree in turn.
    #     If any tree lands in a terminal state, store the game, reset the tree.
    #     """
    #     for _tree in self.trees:
    #         action_argument = _tree.determine_best_action(temperature=self.selection_temperature)
    #         _tree.step(action_argument)
    #
    #         # check if the game has finished:
    #         if _tree.root_node.result is not None:
    #             # collect the full set of moves in that game:
    #             _full_game = _tree.store_complete_game()
    #
    #             # restart the tree to the start position and keep playing:
    #             new_root_node = self.root_nodes.pop()
    #             _tree.reset(new_root_node)
    #
    #             self.num_completed += 1
    #             self.completed_games.append(_full_game)

    # def explore(self) -> None:
    #     """
    #     Flow to each leaf, expand, collect and stack the states,
    #     and provision the inference.
    #     Collect the list of leaf nodes corresponding to the list of internal trees.
    #     Note: for patterns, there is no caching.
    #     """
    #     tensor_states, rleaf_nodes, rtrees = self.expand_and_create_tensors()
    #
    #     if tensor_states:
    #         self.get_and_assign_inference(tensor_states, rleaf_nodes, rtrees)

    # def random_explore(self) -> None:
    #     """
    #     take random explorations , where a random tensor is assigned and no need for a tensor state:
    #
    #     Note a random exploration is a permnent decision for a node; it creates the breadth restricted actions
    #     and this cannot be changed.
    #     """
    #     random_policies = np.random.rand(len(self.trees), 107)
    #     random_values = np.random.rand(len(self.trees))
    #
    #     for  _tree, _val, _pol in zip(self.trees, random_values, random_policies):
    #         # flow to current leaf following argmax of puct scores:
    #         leaf_node = _tree.get_leaf_node()
    #
    #         # leaf node is unchanged if terminal or first visit, else random child:
    #         leaf_node = leaf_node.expand()
    #
    #         # Only store those leaves that are non-terminal
    #         if leaf_node.result is None:
    #             # assign random full policy and values:
    #             leaf_node.value_score = _val
    #
    #             # policy cannot be restricted just yet, as the game action space is not yet known:
    #             leaf_node.full_policy = _pol
    #
    #         # result or no, back propagate the result:
    #         _tree.back_propagate(leaf_node)

    # def expand_and_create_tensors(self) -> tuple[list[torch.tensor], list[Node], list[Tree]]:
    #     """ if a leaf node ends in a terminal state, just backpropagae here, else return relevant
    #     trees and nodes and tensors
    #     """
    #     relevant_leaf_nodes = []
    #     relevant_trees = []
    #     tensor_states = []
    #
    #     # iterate over trees in series:
    #     for _it, _tree in enumerate(self.trees):
    #         # flow to current leaf following argmax of puct scores:
    #         leaf_node = _tree.get_leaf_node()
    #
    #         # leaf node is unchanged if terminal or first visit, else random child:
    #         leaf_node = leaf_node.expand()
    #
    #         # Only store those leaves that are non-terminal
    #         if leaf_node.result is None:
    #             relevant_leaf_nodes.append(leaf_node)
    #             relevant_trees.append(_tree)
    #
    #             # if the leaf node does not yet have a tensor state assigned, assign one:
    #             leaf_node.create_tensor_state()
    #             tensor_states.append(leaf_node.tensor_state)
    #
    #         # otherwise, just back-propagate now:
    #         else:
    #             _tree.back_propagate(leaf_node)
    #
    #     return tensor_states, relevant_leaf_nodes, relevant_trees
    #
    # def get_and_assign_inference(self,
    #                              tensor_states: list[torch.tensor],
    #                              leaf_nodes: list[Node],
    #                              trees: list[Tree]) -> None:
    #     """ evaluate the tensor states and assign the inference results to the leaf nodes
    #     """
    #     # size is (num_trees, 102, 8 8)
    #     tensor_stack = torch.stack(tensor_states)
    #
    #     # two head inference results:
    #     with torch.inference_mode():
    #         value_stack, policy_stack = self.network(tensor_stack.float().to(self.device, non_blocking=True))
    #
    #     # # test just on cpu for now:
    #     # value_stack = -1. + 2. * torch.rand((tensor_stack.shape[0]))
    #     # policy_stack = torch.rand((tensor_stack.shape[0], 107))
    #
    #     value_stack = value_stack.to("cpu", non_blocking=True).numpy()
    #     policy_stack = policy_stack.to("cpu", non_blocking=True).numpy()
    #
    #     # provision the results of the value and policy to each leaf node,
    #     # and back-propagate the result:
    #     for _leaf, _tree, _val, _pol in zip(leaf_nodes, trees, value_stack, policy_stack):
    #         # value score is normalized in the network to be between -1 and 1:
    #         _leaf.value_score = _val
    #
    #         # policy cannot be restricted just yet, as the game action space is not yet known:
    #         _leaf.full_policy = _pol
    #
    #         # if the leaf is terminal, this will have been back-propagated instead of the value result:
    #         _tree.back_propagate(_leaf)
