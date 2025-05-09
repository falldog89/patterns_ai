""" search tree code following MCTS algortithm, adapted to patterns
"""

import numpy as np
import random
from typing import Optional, Self

import torch

from game import Patterns


class Node:
    """ search tree node for patterns:
    """
    def __init__(self,
                 game: Patterns,
                 parent: Optional[Self] = None,
                 parent_action_arg: Optional[int] = None,
                 depth: int = 0,
                 ) -> None:
        # the only information necessary to calculate value of this node:
        self.game = game
        self.possible_actions = game.get_actions()
        self.result = game.result
        self.children = []
        self.child_tuples: Optional[tuple] = None

        self.parent = parent
        self.player = self.game.player
        self.children: Optional[list[Self]] = None
        self.parent_action_arg = parent_action_arg
        self.depth = depth

        # Node search results:
        self.visit_count: int = 0
        self.accrued_reward_to_parent: float = 0.0

        # NN prediction of current state value:
        self.value_score: Optional[float] = None

        # NN policy prediction restricted to legal moves:
        self.policy_vector: Optional[np.ndarray[np.double]] = None

        # MCTS attributes:
        arr_size = len(self.possible_actions)
        self.child_exploitation_scores = np.full(arr_size, np.inf, dtype=float)
        self.child_visit_counts = np.full(arr_size, 1, dtype=int)

    def calculate_reward(self) -> float:
        """ The REWARD for this node. Note this is not equivalent to result.

        Reward will be 1 if result matches player, else -1.

        Note that nn_value_score attempts to recreate result, and not reward.
        """
        if self.result is not None:
            return self.result * self.player

        return self.value_score * self.player

    def calculate_exploitation_score(self) -> float:
        """ the exploitation score of a single node, taking values between -1 and 1: """
        return self.accrued_reward_to_parent / self.visit_count

    def calculate_child_exploration_scores(self) -> np.ndarray[float]:
        """ the exploration scores for a single node, a variation on puct scores:
        """
        return self.policy_vector * (self.visit_count ** 0.5) / (self.child_visit_counts + 1.0)

    @staticmethod
    def numpy_softmax(logits: np.ndarray[float]) -> np.ndarray[float]:
        """ numpy implementation given the slowness of torch tensors for allocation
        """
        exp_demaxed_logits = np.exp(logits - np.max(logits))
        return exp_demaxed_logits / exp_demaxed_logits.sum()

    def expand(self) -> Self:
        """ expand this node by creating and assigning child nodes:
         """
        # If the leaf state is terminal:
        if self.result is not None:
            return self

        # 1st visit, do not create children, simply request inference:
        if self.visit_count == 0:
            return self

        # 2nd visit, populate children, return random child:
        for _it, _move in enumerate(self.possible_actions):
            # Create a new game:
            game = Patterns(self.parent.game)

            # Progress the new game to the relevant state:
            game.step(_move)

            new_node = Node(
                game=game,
                parent=self,
                parent_action_arg=_it,
                depth=self.depth + 1
            )

            self.children.append(new_node)

        # note there will be at least one child as otherwise the game is terminal
        # and result would be populated:
        return random.choice(self.children)

    def get_tensor_state(self) -> torch.Tensor:
        """ Use the game state to form a torch tensor that can be used to eval the position

        Currently, the history is not included, but it might be in the future.

        The tensor state is 8 x 8 x (6x3 + 6x6 + 6x6 + 2x6) binary tensor.

        The first 6x3 planes represent the board itself, with planes 0-5 denoting the presence of an unflipped token
        of that color, 6-11 representing flipped for active player of that color, and 12-17 flipped for passive player.

        This matches the numpy array, one-hot-encoded.

        The next 2x6x6 planes represent the color group order taken, for each player.

        In particular, these planes are constant 1 or constant 0.
        Planes 0-5 represent the order at which color 0 was taken for the active player
        Planes 6-11 represent the order at which color 1 was taken for the active player etc.

        Planes 36 - 41 represent the order at which color 0 was taken for the passive player.
        Planes 42 - 47 represent the order at which color 1 was taken for the passive player etc.

        Finally, the final 12 planes represent the color of the bowl token for the active player (0-5) and the
        passive player (6-11).
        """
        # the board is just a one hot encoded version of the numpy board:
        board_tensor = torch.nn.functional.one_hot(torch.tensor(self.game.active_board).long(), num_classes=18)

        # 36 channels for each player for color group: order mapping:
        order_tensor = torch.zeros((8, 8, 72), dtype=bool)
        for _it, (active_order, passive_order) in enumerate(zip(self.game.active_color_order, self.game.passive_color_order)):
            if active_order > 0:
                order_tensor[:, :, _it * 6 + active_order - 1] = 1

            if passive_order > 0:
                order_tensor[:, :, 36 + _it * 6 + passive_order - 1] = 1

        # bowl tokens:
        bowl_tensor = torch.zeros((8, 8, 12), dtype=bool)
        bowl_tensor[:, :, self.game.active_bowl_token] = 1
        bowl_tensor[:, :, 6 + self.game.passive_bowl_token] = 1

        concat_tensor = torch.cat([board_tensor, order_tensor, bowl_tensor], dim=-1)
        return concat_tensor.permute(2, 0, 1)


class Tree:
    """ Search tree functions, as most nodes will never need to access these.
    """
    def __init__(self,
                 root_node: Node,
                 tree_id: str,
                 dirichlet_noise_level: float = 1.0,
                 dirichlet_noise_epsilon: float = 0.2,
                 puct_constant: float = 2.0 ** 0.5,
                 ) -> None:
        # the game and the root node of the search tree
        self.root_node = root_node
        self.root_node_explore_count = 0

        self.tree_id = tree_id
        # noise property only added to root node exploration:
        self._noise = None

        # Constants:
        self.puct_constant = puct_constant
        self.dirichlet_noise_level = dirichlet_noise_level
        self.dirichlet_noise_epsilon = dirichlet_noise_epsilon

        self.log = None

    def reset(self) -> None:
        """ take the tree back to the initial position:
        """
        new_game = Patterns()
        self.root_node = Node(new_game)
        self.root_node_explore_count = 0

        # and reset the noise:
        self._noise = None

    @property
    def noise(self) -> np.ndarray[float]:
        """ Each agent starts with idiosyncratic dirichlet noise that will encourage subtle different
        exploration of the tree, that is the noise stays correlated from the root node.

        Note that the same noise will be used until the tree steps to the next node

        Noise is only used when the tree is in training mode. Agents should not use noise
        when playing optimally
        """
        if self._noise is None:
            self.set_noise()

        return self._noise

    def set_noise(self) -> None:
        """ setter for the noise property:
        """
        self._noise = np.random.dirichlet([self.dirichlet_noise_level] * len(self.root_node.possible_actions))

    def get_leaf_node(self) -> Node:
        """ flow from the root node to an unexpanded leaf node, following the highest puct score:
        """
        # increment the root node visit count:
        self.root_node_explore_count += 1

        if not self.root_node.children:
            return self.root_node

        # root node has additional epsilon weighted dirichlet noise to encourage diversity in exploration:
        puct_scores = self.calculate_root_puct_scores()
        node = self.root_node

        while node.children:
            node = node.children[np.argmax(puct_scores)]
            puct_scores = self.calculate_child_puct_scores(node)

        return node

    def calculate_root_puct_scores(self) -> np.ndarray[float]:
        """ Upon creation of the search tree, dirichlet noise is assigned to the root node that will
        bias the search tree in relatively few directions for the duration of the explorations. Once the
        search tree steps to a new root node, the dirichlet noise is reset, and new correlated noise is created.
        """
        root_exploration_scores = self.root_node.calculate_child_exploration_scores()

        # (1-eps) * normal policy + eps * dirichlet noise biases the prior for the agent
        eps_root_exploration_scores = ((1.0 - self.dirichlet_noise_epsilon) * root_exploration_scores
                                       + self.dirichlet_noise_epsilon * self.noise)

        # combine with exploitation as usual:
        return self.root_node.child_exploitation_scores + self.puct_constant * eps_root_exploration_scores

    def calculate_child_puct_scores(self, node: Node) -> np.ndarray[float]:
        """ the specific combination of exploitation and exploration scores
        that is used to flow through the search tree:
        """
        exploration_scores = node.calculate_child_exploration_scores()
        return node.child_exploitation_scores + self.puct_constant * exploration_scores

    def determine_best_action(self, temperature: Optional[float]) -> int:
        """ Sample from the child actions vector, from the distribution formed from the child visit counts:
        """
        if not self.root_node.possible_actions:
            raise ValueError("The game has no valid actions, and should have ended...")

        if temperature is not None:
            selection_scores = self.root_node.child_visit_counts ** (1.0 / temperature)
            selection_scores /= selection_scores.sum()
            return np.random.choice(range(len(selection_scores)), p=selection_scores).item()

        return np.argmax(self.root_node.child_visit_counts).item()

    def step(self, action_argument: int) -> None:
        """ Progress the tree according to the action argument """
        # step to new root node, taking the action selected:
        self.root_node = self.root_node.children[action_argument]

        # reset the dirichlet noise:
        self._noise = None

        # reset the root node visit count:
        self.root_node_explore_count = 0

    @staticmethod
    def update_node_child_scores(node: Node) -> None:
        """ Update the child scores for the parent of the node argument.

        Note that the child visit counts of zero would result in a NaN value.
        Therefore, this is initialized to 1, and the understanding is that the
        addition for child puct scores will grant the necessary inf value
        """
        if node.parent is None:
            return

        parent = node.parent

        parent.child_exploitation_scores[node.parent_action_arg] = node.calculate_exploitation_score()
        parent.child_visit_counts[node.parent_action_arg] = node.visit_count

    def back_propagate(self, node: Node) -> None:
        """ push the result of the leaf node rollout up the tree to the root node,
        updating the reward and visit counts of the nodes on this path.

        Stop once you update root.
        """
        reward = node.calculate_reward()

        # stop once you reach and update the root node:
        while node is not self.root_node.parent:
            # note negative, as reward to parent is -1 * reward to child:
            node.accrued_reward_to_parent -= reward
            node.visit_count += 1
            self.update_node_child_scores(node)
            node = node.parent

            # flip the sign of the reward:
            reward *= -1

            self.log = (node, self.root_node)

    def store_complete_game(self) -> list[tuple]:
        """ Once the root node is in a terminal state, store the various states for use in the replay buffer:
        """
        result = self.root_node.result

        if result is None:
            raise ValueError("this function should only be called on a completed game!")

        nod = self.root_node
        full_game_backwards = [0] * self.root_node.game.turn_number

        # work backwards up through the tree, storing the states and the visit counts:
        while nod:
            # todo: weighted average of results from a given position?

            # save list of game states, child visit counts, and _result:
            full_game_backwards[nod.game.turn_number] = (nod.state_tuple, nod.child_visit_counts, result)
            nod = nod.parent

        return full_game_backwards
