""" Search tree for MCTS algorithm, similar to alpha zero.

The search tree manages flow to root and backpropagation of leaf rewards, along with
the stepping and action selection functionality.

We have implemented several extensions here.

### Extensions:

Random play:
The search tree can play randomly, which allows for far faster game generation
in the early stage of the process, before any signal can be expected from the NNs.

A schedule can be set that will allow for random play for so many moves,
before moving to search tree deeper into the game.

Our prior assumption is that game understanding will flow up the game from
terminal positions to start positions, so that better understanding of game
states close to terminal positions will happen sooner in the training process.

Note that the search tree will always avoid losing moves if there are any
non-losing options, and will always take a win if there is one available.

The save depth may also be altered, so that distant-from-terminal positions
are not trained on until deep into the process.

Restricted search:
Our prior assumption is that the training process might be improved by favoring deeper
searches over broader ones. In particular, for a small number of exploration steps allowed,
it might be better to spend these on following the expected best path rather than exploring
every move once.

The code allows the user to specify TOPN and RANDN moves. TOPN will search only the top
n moves according to the prior policy, and RANDN will search a further N random moves to avoid
blind spots. Setting these values to 108 will of course result in an unchanged search
strategy.

todo add __repr__
"""

import numpy as np
import random
from typing import Optional

from game import Patterns
from node import Node


class Tree:
    """ Search tree functions, as most nodes will never need to access these.
    """
    def __init__(self,
                 root_node: Optional[Node] = None,
                 tree_id: str = 't1',
                 dirichlet_noise_level: float = 1.0,
                 dirichlet_noise_epsilon: float = 0.2,
                 puct_constant: float = 2.0 ** 0.5,

                 # parameters for restricting breadth of search:
                 restrict_topn: Optional[int] = 4,
                 restrict_randm: Optional[int] = 4,

                 # schedule for explorations steps scaling with tree depth:
                 schedule: Optional[list[tuple]] = None,
                 ) -> None:

        # Create a root node if none is supplied:
        if root_node is None:
            game = Patterns()
            root_node = Node(
                game=game,
                restrict_topn=restrict_topn,
                restrict_randm=restrict_randm
            )

        self.root_node = root_node

        # rather than marching up through the nodes using .parent, we store the path to leaf
        self._traversed_path = []

        # breadth restriction parameters:
        self.restrict_topn = restrict_topn
        self.restrict_randm = restrict_randm

        # string reference for the search tree:
        self.tree_id = tree_id

        # noise property:
        self._noise = None

        # Constants:
        self.puct_constant = puct_constant
        self.dirichlet_noise_level = dirichlet_noise_level
        self.dirichlet_noise_epsilon = dirichlet_noise_epsilon

        # Flag set to True when search tree has explored sufficiently many nodes and is ready to step root node:
        self.is_step_ready = False

        # save the schedule to pass down to the next root nodes, use it to determine the required exploration steps
        self.schedule = schedule or [(0, 100)] # default of 100 exploration steps
        self.root_node_explore_count: int = 0
        self._required_steps: Optional[int] = None

        # save a replay buffer for each game started:
        self.replay_buffer: list = []

    def reset(self, root_node: Optional[Node] = None) -> None:
        """ take the tree back to the initial position:
        """
        if root_node is None:
            new_game = Patterns()
            root_node = Node(
                game=new_game,
                restrict_topn=self.restrict_topn,
                restrict_randm=self.restrict_randm,
            )

        self.root_node = root_node
        self.root_node_explore_count = 0

        # Reset correlated dirichlet root node noise:
        self._noise = None

        # reset the flag for root node stepping:
        self.is_step_ready = False

        # Reset the required steps property:
        self._required_steps = None

        # Reset the replay buffer:
        self.replay_buffer = []

    @property
    def required_steps(self) -> int:
        """ use the schedule to determine how many explore steps are required for the root node.
        """
        if self._required_steps is None:
            self.set_required_steps()

        return self._required_steps

    def set_required_steps(self) -> None:
        """ use the schedule to determine how many root node explores this node should have:

        Schedule always starts with (0, X) to state that there are X required steps at depth 0

        Then either there is no other tuple, in which case all root nodes explore for X, or there are
        other schedules, and as soon as your depth is below the first entry, you take the previous

        Note we assume that schedule is strictly sorted in the first argument.
        """
        curr_explore = self.schedule[0][1]

        for _depth, _steps in self.schedule:
            if self.root_node.depth < _depth:
                break

            curr_explore = _steps

        self._required_steps = curr_explore

    @property
    def noise(self) -> np.ndarray[float]:
        """ Each agent starts with idiosyncratic dirichlet noise that will encourage subtle different
        exploration of the tree, that is the noise stays correlated from the root node.

        Note that the same noise will be used until the tree steps to the next node

        Noise is only used when the tree is in training mode. Agents should not use noise
        when playing optimally

        "The ϵ parameter for the Dirichlet noise is set to 0.25 and the α parameter to 0.03,
         which is consistent with the heuristic of using a = 10 / n with n the
         maximum number of possibles moves, which is 19 × 19 + 1 = 362
          in the case of Go."
        """
        if self._noise is None:
            self.set_noise()

        return self._noise

    def set_noise(self) -> None:
        """ setter for the noise property:
        """
        self._noise = np.random.dirichlet([self.dirichlet_noise_level] * len(self.root_node.possible_actions))

    def choose_action_argument(self, temperature: Optional[float] = None) -> int:
        """ Sample from the child actions vector according to the distribution formed from child visit counts:
        """
        # If the game is played randomly, the node will not have been expanded correctly and actions will not be assigned
        if not self.root_node.possible_actions:
            if not self.root_node.is_assigned_actions:
                # If the root node has not expanded fully yet, complete here:
                _ = self.root_node.populate_attributes_and_assign_children()

            else:
                raise ValueError("The game has no valid actions, and should have ended...")

        # If there are winning moves, return one at random:
        if self.root_node.winning_action_arguments:
            return np.random.choice(self.root_node.winning_action_arguments)

        # If there are ONLY losing moves, return one at random:
        if len(self.root_node.losing_action_arguments) >= len(self.root_node.possible_actions):
            return random.choice(self.root_node.losing_action_arguments)

        # If there are any non-losing moves, mask out 1-move losses:
        okay_arguments = list(set(range(len(self.root_node.possible_actions)))
                              - set(self.root_node.losing_action_arguments))

        # Playing randomly:
        if self.required_steps == 0:
            return random.choice(okay_arguments)

        filtered_visit_counts = self.root_node.child_visit_counts[okay_arguments]

        # Sample from the visit count distribution, according to temperature parameter:
        if temperature is not None:
            selection_scores = filtered_visit_counts ** (1.0 / temperature)
            selection_scores /= selection_scores.sum()
            return np.random.choice(okay_arguments, p=selection_scores).item()

        # if no selection score, instead select the most visited child that isn't a loss:
        _argmax = np.argmax(filtered_visit_counts)
        return okay_arguments[_argmax]

    def step(self, action_argument: int) -> None:
        """ Step root node to selected child, according to action_argument.

        Reset properties, and populate game if necessary.

        Note that when a tree is taking RANDOM moves, it will be moving to
        a child that has not been visited or seen before. These children will not have
        states, values or policies assigned.

        Delete the root node, but store the necessary attributes in the replay buffer, in particular the
        state, the visit counts etc.
        """
        # Before progressing, store the necessary replay information from the root node:
        self.store_replay()

        # the target to become the new root node:
        chosen_child = self.root_node.children[action_argument]

        # only create a new game if one is not already there:
        if chosen_child.game is None:
            # create new game for new root:
            game = Patterns(self.root_node.game)
            action = self.root_node.possible_actions[action_argument]
            game.step(action)

            # and assign to the child:
            chosen_child.game = game

        self.root_node = chosen_child
        self.root_node_explore_count = self.root_node.visit_count

        # reset the dirichlet noise:
        self._noise = None

        # check the schedule again after stepping:
        self._required_steps = None
        self.is_step_ready = False

        # if the root node requires random moving, set the full policy to be random
        if self.required_steps == 0:
            self.root_node.full_policy = np.random.rand(107) # 107 actions total:
            self.root_node.value_score = -1. + 2. * np.random.rand()

        # state is created from parent state, so create this before discarding parent:
        self.root_node.ensure_state()

        # Kill the top of the tree to save memory:
        self.root_node.parent = None

    def get_leaf_node(self) -> Node:
        """ Flow from the root node to an unexpanded leaf node, following the highest puct score:

        If node ever possesses winning arguments, one is taken and the leaf is terminal.

        If possible, nodes will always avoid losing arguments.
        """
        self.root_node_explore_count += 1

        # start the path at the root node:
        self._traversed_path.append(self.root_node)

        # if sufficiently many exploration steps have been taken, prepare to step the root node:
        if self.root_node_explore_count >= self.required_steps:
            self.is_step_ready = True

        # if mate in 1 detected, prepare to step straight away. Do not bother exploring further:
        if self.root_node.winning_action_arguments:
            self.is_step_ready = True

        # if this is a random player, do not perform exploration, always ready to step
        if self.required_steps == 0:
            self.is_step_ready = True
            return self.root_node

        # flow down from root to leaf following the best puct scores or winning arguments:
        node = self.root_node

        while node.children:
            node = self.next_node(node)
            # store the path taken to flow to leaf:
            self._traversed_path.append(node)

        return node

    def next_node(self, node: Node) -> Node:
        """ get the next node in order, avoiding 1 move losses if possible and taking 1 move wins:
        Follow the best puct score.
        """
        # Step to a random winning node if possible:
        if node.winning_action_arguments:
            return node.children[random.choice(node.winning_action_arguments)]

        num_actions = len(node.possible_actions)

        # Step to a random losing node if everything loses:
        if len(node.losing_action_arguments) >= num_actions:
            return random.choice(node.children)

        # calculate the PUCT scores:
        puct_scores = self.calculate_child_puct_scores(node)

        # If there are no losing arguments, take the best of those available:
        if not node.losing_action_arguments:
            return node.children[np.argmax(puct_scores)]

        # Mask out the losing arguments and choose the best of those remaining:
        losing_mask = np.zeros(num_actions, dtype=bool)
        losing_mask[node.losing_action_arguments] = True
        not_losing_mask = ~losing_mask

        masked_scores = np.where(not_losing_mask, puct_scores, -np.inf)
        best_arg = np.argmax(masked_scores)

        return node.children[best_arg]

    def calculate_child_puct_scores(self, node: Node) -> np.ndarray[float]:
        """ determine whether the node is the root node or not, and return the puct scores
        accordingly.

        Puct score is exploitation + puct constant * exploration scores.

        Exploitation is set to +inf initially, so that until a node is visited and the score corrected,
        unexplored nodes will be prioritised.
        """
        exploration_scores = node.calculate_child_exploration_scores()
        exploitation_scores = node.child_exploitation_scores

        # root node has additional correlated noise to favour deep exploration in given directions:
        if node is self.root_node:
            exploration_scores = ((1.0 - self.dirichlet_noise_epsilon) * exploration_scores
                                  + self.dirichlet_noise_epsilon * self.noise)

        return exploitation_scores + self.puct_constant * exploration_scores

    @staticmethod
    def update_parent_child_scores(node: Node) -> None:
        """ Update the child scores for the parent of the node argument.
        """
        if node.parent is None:
            return

        parent = node.parent
        parent_action_arg = node.parent_action_arg

        # exploitation score is the result if terminal or the average reward otherwise (accrued reward / visit count)
        # negated for zero sum 2 player game with alternating turns:
        parent.child_exploitation_scores[parent_action_arg] = -node.calculate_exploitation_score()

        # child visit counts are initialized to 1 to avoid divide by zero errors:
        parent.child_visit_counts[parent_action_arg] = node.visit_count

    def back_propagate(self) -> None:
        """ Push the leaf node rewards back up the tree to the root node,
        updating the reward and visit counts of the nodes on this path.
        """
        # either result for terminal or value judgement for the leaf node:
        reward = self._traversed_path[-1].calculate_reward()

        # if the final node is terminal, we do not want to switch the game and game reward:
        if self._traversed_path[-1].result is not None:

            # no need to do accrued reward for a terminal node:
            self._traversed_path[-1].visit_count += 1

            # and skip this node as the player will not switch after a terminal move:
            self._traversed_path.pop()

        # Iterate over the root-to_leaf path, alternating the reward:
        for _node in reversed(self._traversed_path):
            _node.visit_count += 1
            _node.accrued_reward += reward # reward for a leaf node starts off as the NN judgement.
            self.update_parent_child_scores(_node)
            reward = -reward # flip reward for opposing players:

        # after iterating, reset the exploration path:
        self._traversed_path = []

    def store_replay(self) -> None:
        """ Store the necessary information for the replay buffer in the internal replay list.
        """
        visit_counts = self.get_replay_visit_counts()
        self.replay_buffer.append( (self.root_node.state, visit_counts) )

    def get_replay_visit_counts(self) -> np.ndarray:
        """ The full visit counts array, with boosting for losses and wins
        """
        # Full action space:
        full_visit_counts = np.zeros(107, dtype=int)
        possible_actions = np.array(self.root_node.possible_actions)

        # when saving visit counts, boost winning arguments, minimize losing ones, and make sure all winning arguments
        # are viewed as equally good!
        if self.root_node.winning_action_arguments:
            full_visit_counts[ possible_actions[self.root_node.winning_action_arguments]] = 1
            return full_visit_counts

        full_visit_counts[self.root_node.possible_actions] = self.root_node.child_visit_counts

        if self.root_node.losing_action_arguments:
            # Boost all non-losing arguments:
            full_visit_counts *= 1000
            full_visit_counts[ possible_actions[self.root_node.losing_action_arguments] ] = 1

        return full_visit_counts
