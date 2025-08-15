"""
Game-playing agent for the game of patterns.

Agents manage multiple trees in series, and provision the inference in parallel.

The structure is roughly:

HEAD - some manager of multiple agents, potentially across many machines. Centralizes the training stage
 |
AGENTS: 1 gpu, multiple threads: manage multiple search trees in series, provision inference in parallel batches. Collect
        completed games and return to the trainer to continually improve the network.
|
TREES: MCTS search trees, with root nodes. Undertake search steps which flow to a leaf and then solve horizon problem
        with NNs trained to recognise valuable states. The same NN guides the search by biasing the prior probability
        of choosing a branch.
|
NODES: the individual nodes of the search tree, managing their own games, states, value, rewards etc.

todo investigate potential for further vectorization eg in incrementing rewards,
"""

import torch
import numpy as np
from typing import Optional

from patterns.game import Patterns

from patterns.search import Node
from patterns.search import Tree


class Agent:
    """ Run search trees in series, with network inference batched in parallel.
    """
    def __init__(self,
                 agent_id: str,
                 network: torch.nn.Module,
                 device: torch.device,
                 num_trees: int = 1,
                 target_games: int = 1,
                 selection_temperature: float = 1.0,
                 restrict_topn: int = 4, # breadth restriction parameters
                 restrict_randm: int = 4,
                 save_depth: Optional[int] = None,
                 explore_steps_schedule: Optional[list[tuple]] = None,
                 ):
        # unique string to identify this agent:
        self.agent_id = agent_id

        # put the network into eval mode, move it to device:
        self.device = device
        self.network = network
        self.network.eval()
        self.network.to(self.device)

        # Number of search trees to run in series:
        self.num_trees = num_trees

        # Selection temperature for the root node steps:
        self.selection_temperature = selection_temperature

        # track number of games completed:
        self.num_completed = 0
        self.target_games = target_games

        # replay data: (states, visit_counts, final result, distance from terminal state)
        self.replay_data = []

        # restrictions placed on search tree breadth, to favor deeper trees over broad ones: see Node.
        self.restrict_topn = restrict_topn
        self.restrict_randm = restrict_randm

        # Number of states and visit counts prior to terminal to save replay data for;
        self.save_depth = save_depth

        # Schedule for number of explore steps with node depth. Often explore more for deeper nodes..
        self.explore_steps_schedule = explore_steps_schedule

        # Create batch of root nodes at instantiation, with batch provisioned inference:
        self.root_nodes: list[Node] = []
        self.create_root_nodes()

        # create the trees and populate each with a pre-inferenced root node:
        self.trees = []

        for _gameit in range(num_trees):
            _root_node = self.root_nodes.pop()

            self.trees.append(
                Tree(
                    root_node=_root_node,
                    tree_id=f"{self.agent_id}_{_gameit}",
                    restrict_topn=self.restrict_topn,
                    restrict_randm=self.restrict_randm,
                    schedule=self.explore_steps_schedule,
                )
            )

        # keep track of the most recent leaf node for debugging:
        self._debug_leaf = None

        # internal lists to track Nodes requiring some form of additional action:
        self.all_nodes = []
        self.inference_nodes = []
        self.states = []
        self.ready_trees = []

    def create_root_nodes(self) -> None:
        """ Create the maximum required root nodes in advance, to take advantage of efficient parallel inference:
        """
        print(f"Generating initial games:")
        states = []

        for _it in range(self.num_trees + self.target_games):
            new_game = Patterns()
            new_nod = Node(
                game=new_game,
                restrict_topn=self.restrict_topn,
                restrict_randm=self.restrict_randm,
            )

            # create the NN state:
            new_nod.ensure_state()
            self.root_nodes.append(new_nod)

            # store for batch eval:
            states.append(new_nod.state)

        print(f"Evaluating states...")

        # size (num_trees + target_games, 47, 8, 8)
        tensor_stack = torch.from_numpy(np.stack(states)).to(
            self.device, dtype=torch.float32, non_blocking=True,
        )

        # two-head inference results:
        with torch.inference_mode():
            value_stack, policy_stack = self.network(tensor_stack)

        value_stack = value_stack.to("cpu", non_blocking=True).numpy()
        policy_stack = policy_stack.to("cpu", non_blocking=True).numpy()

        print(f"Provisioning inference to root nodes...")

        # provision the results to the root nodes:
        for _nod, _val, _pol in zip(self.root_nodes, value_stack, policy_stack):
            _nod.value_score = _val
            _nod.full_policy = _pol

            # set visit count to skip the expand step that seeks inference:
            _nod.visit_count = 1
            _nod.is_inference_assigned = True

    def run_games(self) -> None:
        """ Continually run exploration, inference and back propagation steps until
        sufficiently many games have been completed.
        """
        num_completed = 0

        # continue until the agent has collected enough games:
        while self.num_completed < self.target_games:
            # Explore every tree in series, track which trees require inference:
            self.explore()

            # Provision inference to relevant nodes:
            self.assign_inference()

            # Back-propagate from each tree leaf node:
            for _tree, _node in zip(self.trees, self.all_nodes):
                _tree.back_propagate()

            # Step each tree that has sufficiently explored from root:
            self.step_trees()

            # if at least one game finished in this iteration, report the number completed:
            if self.num_completed != num_completed:
                num_completed = self.num_completed
                print(f"{self.num_completed} games have been completed!")

    def explore(self) -> None:
        """ A single explore step for each tree. Store all nodes that require inference, their states,,
        and the trees that are ready to take a move due to sufficient exploration.
        """
        # reset internal lists:
        self.all_nodes = []
        self.inference_nodes = []
        self.states = []
        self.ready_trees = []

        for _tree in self.trees:
            leaf_node = _tree.get_leaf_node()

            # track debug leaf:
            self._debug_leaf = leaf_node

            leaf_node = leaf_node.expand()

            # track all leaf nodes:
            self.all_nodes.append(leaf_node)

            # Track trees that are ready to take a move (sufficient exploration,  mate-in-one, random play)
            if _tree.is_step_ready:
                self.ready_trees.append(_tree)

            if not leaf_node.is_inference_assigned:
                # ensure there is a state for inference for this leaf node:
                leaf_node.ensure_state()

                # track which nodes will require inference results:
                self.inference_nodes.append(leaf_node)
                self.states.append(leaf_node.state)

    def assign_inference(self) -> None:
        """ Evaluate the network on the nodes that require inference and provision
        value and policy results to the internally stored nodes.
        """
        if not self.states:
            return

        tensor_stack = torch.from_numpy(np.stack(self.states)).to(
            self.device, dtype=torch.float32, non_blocking=True,
        )

        # two head inference results:
        with torch.inference_mode():
            value_stack, policy_stack = self.network(tensor_stack)

        value_stack = value_stack.to("cpu", non_blocking=True).numpy()
        policy_stack = policy_stack.to("cpu", non_blocking=True).numpy()

        # provision the results of the value and policy to each leaf node:
        for _leaf, _val, _pol in zip(self.inference_nodes, value_stack, policy_stack):
            # value score is normalized in the network to be between -1 and 1:
            _leaf.value_score = _val

            # policy cannot be restricted until the legal actions are known:
            _leaf.full_policy = _pol

            # switch flag for inference:
            _leaf.is_inference_assigned = True

    def step_trees(self):
        """ Progress the trees that have root nodes that are ready to make a move
        """
        for _tree in self.ready_trees:
            action_argument = _tree.choose_action_argument()
            _tree.step(action_argument)

            # save the results from each tree that is terminal:
            if _tree.root_node.result is not None:
                # Note: replay buffer does not contain the terminal node
                result = _tree.root_node.result
                replay_data = _tree.replay_buffer[-self.save_depth:]

                # (state, visit counts, result for final active player, distance from terminal state):
                replay_extension = [(_state, _vcs, result, _it + 1)
                                    for _it, (_state, _vcs) in enumerate(reversed(replay_data))]

                self.replay_data.extend(replay_extension)

                # Claim new root node and restart the tree:
                new_root_node = self.root_nodes.pop()
                _tree.reset(new_root_node)
                self.num_completed += 1

                # break if the number of completed games is sufficient:
                if self.num_completed >= self.target_games:
                    break
