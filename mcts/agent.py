""" an agent who will play games of patterns according to current network iteration
"""

import numpy as np
import torch

from game import Patterns
from mcts.mcts import Node, Tree
from typing import Optional


class Agent:
    """ Each agent will run num_trees search trees in series.

    Given that the search tree steps are very quick versus the inference, this helps
    to bring the inference bottleneck time up to the search tree time.
    """
    def __init__(self,
                 agent_id: str,
                 network: torch.nn.module,
                 num_trees: int = 20,
                 explore_steps: int = 800,
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

    def select_next_actions(self) -> list[list[tuple]]:
        """ Iterate over trees and select the best action from the distribution given by the tree
        visit counts
        """
        completed_games_list = []

        for _tree in self.trees:
            action_argument = _tree.determine_best_action(temperature=self.selection_temperature)
            _tree.step(action_argument)

            # check if the game has finished:
            if _tree.root_node.result is not None:
                # collect the full set of moves in that game:
                _full_game = _tree.get_terminal_positions()

                # restart the tree to the start position and keep playing:
                _tree.reset()
                self.num_completed += 1

                completed_games_list.append(_full_game)

        return completed_games_list

    def get_min_explore_steps(self) -> int:
        """ determine the lowest root visit count for each root node in trees
        """
        lowest_visit_count = 1e9
        for t in self.trees:
            _visit_count = t.root_node_explore_count

            if _visit_count < lowest_visit_count:
                lowest_visit_count = _visit_count

        return lowest_visit_count

    def explore_trees_to_inference(self,
                                   state_tuples: Optional[list[tuple]],
                                   inference_results: Optional[tuple[np.ndarray, np.ndarray]],
                                   ) -> list[tuple]:
        """ explore each tree until it hits a point where it requires inference or has hit its
        required number of explore steps
        """
        # add the results of the requested inference to the cache:
        self.add_to_cache(state_tuples, inference_results)

        # now that inference results are assigned, the leaf nodes can be backpropped:
        self.rollout_and_backprop()

        # get the next set of leaf nodes:
        leaf_nodes = self.explore_to_inference()

        # Store the leaf nodes in the internal store to resume after evaluation:
        self.leaf_nodes = leaf_nodes

        # get the unique state tuples for inference:
        unique_state_tuples = list(set([_nod.state_tuple for _nod in leaf_nodes
                               if _nod is not None and
                               _nod.state_tuple not in self.inference_cache]))

        return unique_state_tuples

    def rollout_and_backprop(self) -> None:
        """ Iterate over the leaf nodes, gather the tensor states, and evaluate in one batch, in inference mode.
        todo inference mode
        then backpropagate the result for each tree.
        """
        # note some of these leaf nodes could be 0
        tensor_states = [_leaf.get_tensor_state() for _leaf in self.leaf_nodes]
        full_tensor = torch.stack(tensor_states)
        value_inference, policy_inference = self.network(full_tensor)

        for _tree, _leaf, _val, _pol in zip(self.trees, self.leaf_nodes, value_inference, policy_inference):
            if _leaf is None:
                continue

            _leaf.value_score = _val
            _leaf.policy_vector = _pol[_leaf.possible_actions]
            _tree.back_propagate(_leaf)

    def explore_to_inference(self) -> list[HiveNode]:
        """ return the leaf nodes and every state tuple encountered:
        """
        # note that these will be kept in order with the trees:
        all_leaf_nodes = []

        # iterate over trees in series:
        for _tree in self.trees:
            while True:
                leaf_node = _tree.get_leaf_node()
                leaf_node = _tree.expand_leaf(leaf_node)

                # if the requested state has not been evaluated before:
                if leaf_node.state_tuple not in self.inference_cache:
                    all_leaf_nodes.append(leaf_node)
                    break

                # do not keep exploring if the tree has enough explore steps:
                if _tree.rnod_explore_count > self.explore_steps:
                    all_leaf_nodes.append(None)
                    break

                # don't wait if result is already accessible:
                value_score, full_policy = self.inference_cache[leaf_node.state_tuple]
                leaf_node.value_score = value_score
                leaf_node.full_policy = full_policy

                # the results of the inference are then back propagated up through the tree:
                _tree.back_propagate(leaf_node)

        return all_leaf_nodes