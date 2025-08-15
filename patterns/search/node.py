""" Node for MCTS algorithm, similar to alpha zero.

Nodes contain information about the legal actions from their position, their parent,
the value and policy verdict on the current position and functionality to create a state
for consumption by the NN.

### Extensions:

Restricted search:
Our prior assumption is that the training process might be improved by favoring deeper
searches over broader ones. In particular, for a small number of exploration steps allowed,
it might be better to spend these on following the expected best path rather than exploring
every move once.

The code allows the user to specify TOPN and RANDN moves. TOPN will search only the top
n moves according to the prior policy, and RANDN will search a further N random moves to avoid
blind spots. Setting these values to 108 will of course result in an unchanged search
strategy.

The Node masks out actions according to this restriction and they are treated
as illegal moves, almost similar to an extreme version of the dirichlet noise at root.

todo add __repr__
"""

import numpy as np
import random
from typing import Optional, Self

from patterns.game import Patterns
from patterns.search import state


class Node:
    """ Search tree node for patterns.
    """
    def __init__(self,
                 parent_action_arg: Optional[int] = None,
                 game: Optional[Patterns] = None,
                 parent: Optional[Self] = None,
                 depth: int = 0,
                 restrict_topn: Optional[int] = None,
                 restrict_randm: Optional[int] = None,
                 ) -> None:
        ### Note: every node must have either a parent action argument or a game.

        # argument for parent node possible_actions:
        self.parent_action_arg = parent_action_arg

        # actual Patterns instance:
        self.game = game

        # parent Node instance:
        self.parent = parent

        # depth of the node in the search tree, with 0 being root-node depth:
        self.depth = depth

        # Restrict to top n actions from prior policy. No restriction if None.
        self.restrict_topn = restrict_topn or 107

        # Additional m random actions are selected for unbiased exploration:
        self.restrict_randm = restrict_randm or 0

        # Store children in a list:
        self.children: list[Self] = []
        self.active_player = 1 if parent is None else -1 * parent.game.active_player

        # list of actions (0-106)
        self.possible_actions: Optional[list[int]] = None

        # None if game is not terminal, else -1, 0, 1 for loss, draw, win for active player:
        self.result: Optional[int] = None # trichotomy

        # List of the action arguments that result in terminal positions:
        self.winning_action_arguments: list = []
        self.losing_action_arguments: list = []

        # store actions that result in a terminal state in a set:
        self.terminal_actions: set = set()

        # numpy arrays pointing to the exploitation and visit counts of the children in self.children:
        self.child_exploitation_scores: Optional[np.ndarray] = None
        self.child_visit_counts: Optional[np.ndarray] = None

        # numpy state, created from state.py:
        self.state: Optional[np.ndarray] = None

        # number of times this node has been touched by search tree
        self.visit_count: int = 0

        # average reward from children explored downstream of this node:
        self.accrued_reward: float = 0.0

        # NN prediction of current state value:
        self.value_score: Optional[float] = None

        # NN full policy, before restriction:
        self.full_policy: Optional[np.ndarray[np.double]] = None

        # After game is populated, policy is restricted to legal moves:
        self.policy_vector: Optional[np.ndarray[np.double]] = None

        # flag so that random trees remember to assign the children and so on correctly.
        self.is_assigned_actions = False

        # flag for inference provided or not:
        self.is_inference_assigned = False

    def populate_attributes(self) -> None:
        """ Only after a game has been created can certain attributes be populated.
        Note that a Node should always have a state by this point.
        """
        if not self.game:
            raise ValueError("A game is necessary to populate attributes.")

        self.result = self.game.result

        # if game is terminal, no need for further attributes:
        if self.result is not None:
            return

        self.assign_actions()

        # flag to assert that the attributes in this function have been populated:
        self.is_assigned_actions = True

        # Once the actions have been assigned, the policy can be restricted:
        restricted_policy = self.full_policy[self.possible_actions]
        self.policy_vector = self.numpy_softmax(restricted_policy)

        # MCTS attributes: note that possible actions already restricted above.
        arr_size = len(self.possible_actions)
        self.child_exploitation_scores = np.array([np.inf] * arr_size, dtype=float)
        self.child_visit_counts = np.ones(arr_size, dtype=int)

    def assign_actions(self) -> None:
        """ Populate the actions from the assigned game and according to the top and random
        breadth restrictions. Use these actions to restrict the full policy to "legal" moves.
        """
        # the full policy should always have been assigned:
        if self.full_policy is None:
            raise ValueError("The full policy should already have been assigned")

        # All the legal actions that can be played, before breadth restrictions:
        game_actions = self.game.get_actions()

        # Terminal actions are never restricted, as it is vital that these are always explored:
        self.terminal_actions = {_action for _action in game_actions if self.game.is_action_terminal(_action)}

        # restrict the full policy to legal actions only:
        restricted_policy = self.full_policy[game_actions]

        # If legal action space sufficiently small, do not restrict further:
        if len(game_actions) <= (self.restrict_topn + self.restrict_randm):
            self.possible_actions = game_actions
            return

        # sort actions according to the policy prior, to give the top n actions:
        sorted_arguments = np.argsort(-restricted_policy)

        # select the top n best, according to the breadth restriction:
        topn_actions = {game_actions[_arg] for _arg in sorted_arguments[:self.restrict_topn]}

        # select further actions randomly:
        remaining_args = sorted_arguments[self.restrict_topn:]
        np.random.shuffle(remaining_args)
        random_actions = {game_actions[_arg] for _arg in remaining_args[:self.restrict_randm]}

        # Re-introduce any terminal actions that were removed:
        self.possible_actions = list(topn_actions | random_actions | self.terminal_actions)

    def calculate_reward(self) -> float:
        """ If game is terminal, return result, else return the value judgement of the network from
        the perspective of the active player.
        """
        # If the result is not None, the game is terminal and the reward is known:
        if self.result is not None:
            return self.result

        # the network tries to predict the value to the current (active) player:
        return self.value_score

    def calculate_exploitation_score(self) -> float:
        """ Either the known result if terminal or the average reward for downstream nodes.
        """
        if self.result is not None:
            return self.result

        return self.accrued_reward / self.visit_count

    def calculate_child_exploration_scores(self) -> np.ndarray[float]:
        """ the exploration scores for a single node, a variation on puct scores:

        TODO:
        Consider alternative scaling for puct score,
        Consider entropic steering to avoid deleterious overconfidence,

        TODO:
        can we save this and just update the single entry that changes?
        """
        return self.policy_vector * (self.visit_count ** 0.5) / self.child_visit_counts

    @staticmethod
    def numpy_softmax(logits: np.ndarray[float]) -> np.ndarray[float]:
        """ numpy implementation of softmax:
        """
        exp_demaxed_logits = np.exp(logits - np.max(logits))
        return exp_demaxed_logits / exp_demaxed_logits.sum()

    def expand(self) -> Self:
        """ expand this node by creating and assigning child nodes:
        """
        # If the leaf state is terminal, do not expand. Terminal states never require a game.
        if self.result is not None:
            return self

        # no-op if the state is already expanded. There must be children, else result is not None.
        if len(self.children) > 0:
            return self

        # Upon first visit, create state from the parent state:
        if self.visit_count == 0:
            return self

        # On second visit, create own game if one doesn't exist and populate attributes:
        if not self.game:
            self.game = Patterns(self.parent.game)
            action = self.parent.possible_actions[self.parent_action_arg]
            self.game.step(action)

        return self.populate_attributes_and_assign_children()

    def populate_attributes_and_assign_children(self) -> Self:
        """ After a node gains a game, populate the action attributes and assign children:
        """
        self.populate_attributes()

        # Games are populated only with a parent action argument and a parent to minimize copy time:
        for _it, _action in enumerate(self.possible_actions):
            new_node = Node(
                parent=self,
                parent_action_arg=_it,
                depth = self.depth + 1,
                restrict_topn=self.restrict_topn,
                restrict_randm=self.restrict_randm,
            )

            # Note: player does not swap for terminal state.
            if _action in self.terminal_actions:
                # Create game to determine terminal state:
                terminal_game = Patterns(self.game)
                terminal_game.step(_action)
                new_node.result = terminal_game.result

                # As the player is not swapped after a terminal action, 1 is win, -1 is loss:
                if new_node.result == 1:
                    # store the argument, not the action itself:
                    self.winning_action_arguments.append(_it)

                # if the active player of the child node wins after this action, this node must view it as a loss:
                elif new_node.result == -1:
                    self.losing_action_arguments.append(_it)

            # store the children to correspond to the possible actions:
            self.children.append(new_node)

        # return a random child for the second expansion:
        return random.choice(self.children)

    def ensure_state(self) -> None:
        """ depending on available attributes, create state efficiently:
        """
        # No-op if the state has already been created:
        if self.state is not None:
           return

        # Most efficient way: create from parent game and parent state:
        if self.parent:
           if self.parent.state is not None:
               # action not action argument:
               parent_action = self.parent.possible_actions[self.parent_action_arg]

               self.state = state.create_state_from_parent_state(
                   self.parent.state,
                   self.parent.game,
                   parent_action)
               return

        # If you have a game, create the state directly from the game:
        if self.game:
           self.state = state.create_state_from_game(self.game)
           return

        # action not action argument:
        parent_action = self.parent.possible_actions[self.parent_action_arg]

        self.state = state.create_state_from_parent_game(self.parent.game,
                                                         parent_action)
