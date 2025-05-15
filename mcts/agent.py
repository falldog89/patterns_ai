""" an agent who will play games of patterns according to current network iteration

todo note that we can just keep the inference results in a big tensor,
and provide the search trees with an object that their id indexes? that way there
is no need to provision the results to the trees?
"""

import torch
import numpy as np
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
                 explore_steps: int = 800,
                 target_games: int = 1000,
                 selection_temperature: float = 1.0,
                 ):
        # unique string for this agent, for the cpu driver to provision
        self.agent_id = agent_id

        # the network that will be used in the inference step to evaluate the current state and return
        # value and policy scores to the nodes:

        # put the network into eval mode, move it to device:
        self.network = network
        self.network.eval()

        self.device = device
        self.network.eval()
        self.network.to(self.device)

        # this many trees will be run in series:
        self.num_trees = num_trees
        self.explore_steps = explore_steps
        self.selection_temperature = selection_temperature

        # track number of games completed:
        self.num_completed = 0
        self.target_games = target_games
        self.completed_games = []

        # make a game for each tree, in turn:
        trees = []
        for _gameit in range(num_trees):
            root_game = Patterns()
            root_node = Node(game=root_game)
            trees.append(
                Tree(
                    root_node=root_node,
                    tree_id=f"{self.agent_id}_{_gameit}",
                )
            )

        self.trees = trees
        self.leaf_nodes: list[Optional[Node]] = [None] * num_trees

    def run_games(self):
        # while more games are required for quota:
        count = 0
        while self.num_completed < self.target_games:
            # explore each tree for required number of moves:
            for _ in range(self.explore_steps):
                self.explore()

            # select an action for each tree root node, reset any trees that
            # are completed:
            self.step_trees()
            count += 1
            print(f"move number {count} completed:")
            print(f"Completed games: {self.num_completed}")

    def step_trees(self) -> None:
        """ After the exploration steps have completed, step each tree in turn.
        If any tree lands in a terminal state, store the game, reset the tree.
        """
        for _tree in self.trees:
            action_argument = _tree.determine_best_action(temperature=self.selection_temperature)
            _tree.step(action_argument)

            # check if the game has finished:
            if _tree.root_node.result is not None:
                # collect the full set of moves in that game:
                _full_game = _tree.store_complete_game()

                # restart the tree to the start position and keep playing:
                _tree.reset()
                self.num_completed += 1
                self.completed_games.append(_full_game)

    def explore(self) -> None:
        """
        Flow to each leaf, expand, collect and stack the states,
        and provision the inference.
        Collect the list of leaf nodes corresponding to the list of internal trees.
        Note: for patterns, there is no caching.
        """
        leaf_nodes = []
        relevant_trees = []
        tensor_states = []

        # iterate over trees in series:
        for _tree in self.trees:
            # flow to current leaf following argmax of puct scores:
            leaf_node = _tree.get_leaf_node()

            # leaf node is unchanged if terminal or first visit, else random child:
            leaf_node = leaf_node.expand()

            # Only store those leaves that are non-terminal
            if leaf_node.result is None:
                leaf_nodes.append(leaf_node)
                relevant_trees.append(_tree)

                # if the leaf node does not yet have a tensor state assigned, assign one:
                leaf_node.create_tensor_state()
                tensor_states.append(leaf_node.tensor_state)

            # otherwise, just back-propagate now:
            else:
                _tree.back_propagate(leaf_node)

        # size is (num_trees, 102, 8 8)
        tensor_stack = torch.stack(tensor_states)

        # two head inference results:
        with torch.inference_mode():
            value_stack, policy_stack = self.network(tensor_stack.float().to(self.device, non_blocking=True))

        value_stack = value_stack.to("cpu", non_blocking=True).numpy()
        policy_stack = policy_stack.to("cpu", non_blocking=True).numpy()

        # provision the results of the value and policy to each leaf node,
        # and back-propagate the result:
        for _leaf, _tree, _val, _pol in zip(leaf_nodes, relevant_trees, value_stack, policy_stack):
            # value score is normalized in the network to be between -1 and 1:
            _leaf.value_score = _val

            # if len(_leaf.possible_actions) > 0:
            # policy vector is logits for a classifier, so must be soft-maxed:
            # softmax_policy = self.numpy_softmax(_pol[_leaf.possible_actions])

            # todo create an assignment function that either assigns
            # to full policy if there is no game, else to pol
            _leaf.assign_policy(_pol) = _pol
            # _leaf.policy_vector = softmax_policy

            # if the leaf is terminal, this will be back-propagated instead of the value result:
            _tree.back_propagate(_leaf)

    @staticmethod
    def numpy_softmax(logits: np.ndarray[float]) -> np.ndarray[float]:
        """ numpy implementation given the slowness of torch tensors for allocation
        """
        exp_demaxed_logits = np.exp(logits - np.max(logits))
        return exp_demaxed_logits / exp_demaxed_logits.sum()
