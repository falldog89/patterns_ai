""" an agent who will play games of patterns according to current network iteration

todo note that we can just keep the inference results in a big tensor,
and provide the search trees with an object that their id indexes? that way there
is no need to provision the results to the trees?
"""

import numpy as np
import torch

from game import Patterns
from mcts.mcts import Node, Tree
from typing import Optional


class Agent:
    """ Each agent will run num_trees search trees in series. However, the
    inference step can occur in one go. That is, all N inferences can be done in parallel.

    Given that the search tree steps are very quick versus the inference, this helps
    to bring the inference bottleneck time up to the search tree time.
    """
    def __init__(self,
                 agent_id: str,
                 network: torch.nn.module,
                 num_trees: int = 20,
                 explore_steps: int = 800,
                 target_games: int = 1000,
                 selection_temperature: float = 1.0,
                 ):
        # unique string for this agent, for the cpu driver to provision
        self.agent_id = agent_id

        # the network that will be used in the inference step to evaluate the current state and return
        # value and policy scores to the nodes:
        self.network = network

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

    def ____(self):
        # while more games are required for quota:
        while self.num_completed < self.target_games:
            # explore each tree for required number of moves:
            for _ in range(self.explore_steps):
                self.explore()

            # select an action for each tree, reset trees that are completed:
            self.step_trees()

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
        tensor_states = []

        # iterate over trees in series:
        for _tree in self.trees:
            # flow to current leaf following argmax of puct scores:
            leaf_node = _tree.get_leaf_node()

            # choose a random child or return yourself if you are visited for the first time:
            leaf_node = leaf_node.expand()

            # store the leaf node to provision the result of the inference:
            leaf_nodes.append(leaf_node)

            # get the tensor representation of the game state, and store for stacking:
            leaf_tensor = leaf_node.get_tensor_state()
            tensor_states.append(leaf_tensor)

        # size is (num_trees, 102, 8 8)
        tensor_stack = torch.stack(tensor_states)

        # two head inference results:
        value, policy = self.network(tensor_stack)

        # provision the results of the value and policy to each leaf node,
        # and back-propagate the result:
        for _leaf, _tree, _val, _pol in zip(leaf_nodes, self.trees, value, policy):
            _leaf.value_score = _val
            _leaf.policy_vector = _pol[_leaf.possible_actions]

            # if the leaf is terminal, this will be back-propagated instead of the value result:
            _tree.back_propagate(_leaf)
