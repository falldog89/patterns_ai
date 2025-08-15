""" Node and Search tree for MCTS algorithm, similar to alpha zero.

Nodes contain information about the legal actions from their position, their parent,
the value and policy verdict on the current position and functionality to create a tensor state
for consumption by the NN.

The search tree has functions that relate to the tree search itself, including
back propagation functions, a root node, the methods used to flow to a leaf etc.

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
from typing import Optional, Self
import torch

from game import Patterns
from constants import loci, locj
from constants import EYE
from constants import SWAP_ACTIVE_PASSIVE_INDEX


class Node:
    """ Search tree node for patterns.
    """
    __slots__ = ('parent_action_arg', 'game', 'parent', 'depth', 'restrict_topn',
                 'restrict_randm', 'children', 'active_player', 'possible_actions',
                 'result', 'winning_action_arguments', 'losing_action_arguments',
                 'terminal_actions', 'child_exploitation_scores', 'child_visit_counts',
                 'tensor_state', 'visit_count', 'accrued_reward', 'value_score',
                 'full_policy', 'policy_vector', 'is_assigned_actions', 'is_inference_assigned')
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

        # numpy tensor state, as described in Node function:
        self.tensor_state: Optional[np.ndarray] = None

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
        Note that a Node should always have a tensor state by this point.
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
        can we save this and just update the single entry that changes? if we remove visit count that is

        visit count applied to everything just the same...
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

        # Upon first visit, create a tensor state from the parent tensor state:
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

    def get_state_attributes(self) -> tuple:
        """ return the active board, the active and passive orders and the active and passive bowl tokens
        that would belong to this node.

        tuple return is:
        board, active_order, passive_order, active_token, passive_token

        update: also provide the color groups so that score is also available:
        """
        # if game is populated, just use the board and state attributes from own game:
        if self.game is not None:
            return (np.array(self.game.active_board), self.game.active_color_order, self.game.passive_color_order,
                    self.game.active_bowl_token, self.game.passive_bowl_token, self.game.is_no_more_placing,
                    self.game.active_color_groups, self.game.passive_color_groups)

        # collect the game and parent action argument:
        game = self.parent.game
        action = self.parent.possible_actions[self.parent_action_arg]

        # always use passive board, as we will swap players from the parent:
        board = np.array(game.passive_board)

        active_token = game.passive_bowl_token
        passive_token = game.active_bowl_token

        # if start of game:
        if action >= 104:
            active_token = (action + 1) % 2
            passive_token = action % 2

            return (board, [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], active_token, passive_token, False,
                    {_col: [] for _col in range(6)}, {_col: [] for _col in range(6)})

        # active and passive reversed to represent the swap in players after action:
        active_order = game.passive_color_order[:]
        passive_order = game.active_color_order[:]

        # also get the color groups:
        active_color_groups = {_key: _val[:] for _key, _val in game.passive_color_groups.items()}
        passive_color_groups = {_key: _val[:] for _key, _val in game.active_color_groups.items()}

        # remaining actions represent a change to the board state:
        location = action % 52
        coords = loci[location], locj[location]

        # update the relevant color group with the coordinates of the flipped location:
        cg_update_color = passive_token if action < 52 else board[coords]
        passive_color_groups[cg_update_color].append(coords)

        # the passive token becomes the color at the location, the new location gets the COLOR of the passive token:
        if action < 52:
            new_token = board[coords]
            board[coords] = passive_token
            passive_token = new_token

            # update the passive order if a new color group would be formed:
            if not game.active_color_groups[game.active_bowl_token]:
                passive_order[game.active_bowl_token] = game.active_placing_number

        # Whether flipping or placing, the location takes on the +12 of the passive player:
        board[coords] += 12

        return (board, active_order, passive_order, active_token, passive_token, game.is_no_more_placing,
                active_color_groups, passive_color_groups)

    def create_tensor_state_from_parent(self) -> np.ndarray: #-> torch.tensor:
        """ Create this nodes tensor state through the parent tensor state and the parent action.
        Swap all active and passive layers, then correct the entries for passive.
        (board, score, bowl token, color order, bowl token value, is no placing)
        """
        parent_tensor = self.parent.tensor_state
        parent_action = self.parent.possible_actions[self.parent_action_arg]

        ### Permute parent tensor to swap active and passive slices:
        numpy_state = np.array(parent_tensor)[SWAP_ACTIVE_PASSIVE_INDEX]

        # Update each of the types of tensor slice in turn:
        self._tensor_update_no_more_placing(numpy_state, parent_action)
        self._tensor_update_board(numpy_state, parent_action)
        self._tensor_update_bowl_tokens(numpy_state, parent_action)
        self._tensor_update_color_order(numpy_state, parent_action)
        self._tensor_update_score(numpy_state, parent_action)
        self._tensor_update_bowl_token_values(numpy_state, parent_action)

        return numpy_state

    def _tensor_update_no_more_placing(self, numpy_state: np.ndarray, parent_action: int) -> None:
        """ correct the flag for no more placing:
        """
        if parent_action >= 104:
            return

        ### correct for no more placing:
        numpy_state[42] = self.parent.game.is_no_more_placing

    def _tensor_update_board(self, numpy_state: np.ndarray, parent_action: int) -> None:
        """ correct the child tensor board based on the parent action
        """
        if parent_action >= 104:
            return

        game = self.parent.game

        # first, determine the location targeted by the action:
        location = parent_action % 52
        coords = loci[location], locj[location]

        # this will either be flipped or replaced:
        board_color = game.active_board[coords]

        ### remove the unflipped in all cases:
        numpy_state[board_color][coords] = 0
        col = game.active_bowl_token if parent_action < 52 else board_color
        numpy_state[12 + col][coords] = 1

    def _tensor_update_bowl_tokens(self, numpy_state: np.ndarray, parent_action: int) -> None:
        """ update the child tensor to account for new passive bowl tokens:
        Note that 104 always means the active player takes 0 and 105 always means that the active player takes
        1
        """
        if parent_action in [104, 105]:
            inds = [30, 31, 36, 37]
            vals = [0, 1, 1, 0] if parent_action == 104 else [1, 0, 0, 1]

            for _ind, _val in zip(inds, vals):
                numpy_state[_ind] = _val

        if parent_action < 52:
            location = parent_action % 52
            coords = loci[location], locj[location]
            board_color = self.parent.game.active_board[coords]

            ### update the passive bowl token if it changes:
            if board_color != self.parent.game.active_bowl_token:
                numpy_state[36 + self.parent.game.active_bowl_token] = 0  # zero the old color
                numpy_state[36 + board_color] = 1  # take the board color as new token

    def _tensor_update_color_order(self, numpy_state: np.ndarray, parent_action: int) -> None:
        """ if the action leads to a new color having been placed, this must be addressed:
        """
        # only affects placing actions:
        if parent_action >= 52:
            return

        game = self.parent.game

        ### if the active bowl token wasn't in active color groups:
        if not game.active_color_groups[game.active_bowl_token]:
            ### update color order:
            numpy_state[24 + game.active_bowl_token] = game.active_placing_number / 6.0

    def _tensor_update_score(self, numpy_state: np.ndarray, parent_action: int) -> None:
        """ update the score slice by accounting for the just flipped or placed piece
        """
        if parent_action >= 104:
            # no change to score:
            return

        game = self.parent.game

        # first, determine the location targeted by the action:
        location = parent_action % 52
        coords = loci[location], locj[location]

        # this will either be flipped or replaced:
        board_color = game.active_board[coords]

        ### if placing action:
        if parent_action < 52:
            ### if the active bowl token wasn't in active color groups, both score and color order are updated:
            if not game.active_color_groups[game.active_bowl_token]:
                ### update score:
                score_value = game.active_placing_number  # amount we increment score by:

            else:
                score_value = game.active_color_order[game.active_bowl_token]

        else:
            ### update the score:
            score_value = game.active_color_order[board_color]

        ### increment the score layer by the score value:
        numpy_state[44] += score_value / 150.0

    def _tensor_update_bowl_token_values(self, numpy_state: np.ndarray, parent_action: int) -> None:
        """ if a new bowl token was taken, update the value of it
        """
        if parent_action >= 52:
            return

        game = self.parent.game

        # first, determine the location targeted by the action:
        location = parent_action % 52
        coords = loci[location], locj[location]

        # this will either be flipped or replaced:
        board_color = game.active_board[coords]

        ### if the board color that has been taken as passive token is not in color groups:
        if not game.active_color_groups[board_color]:
            # the value is the current next to play:
            bowl_token_value = game.active_placing_number

            # However, if the piece just played was a different next to play, increment again!
            if (not game.active_color_groups[game.active_bowl_token]) and (game.active_bowl_token != board_color):
                bowl_token_value += 1

        else:
            bowl_token_value = game.active_color_order[board_color]

        numpy_state[46] = bowl_token_value / 6.0

    def create_tensor_state(self) -> None:
        """ Ideally tensor state would be cheaply made by copying the previous tensor state

        Slices of the (47, 8, 8) state tensor are as follows.

        :18
        board. one hot encoded. :6 is unflipped, 6:12 is active player, 12:18 is passive player

        18:30
        The order in which each player took the colors. Float. 18:24 active player. 0 is untaken.

        30:42
        Bowl tokens. one hot encoded. :36 is active player.

        42:
        bool for is no more placing. ie the game will end soon and placing moves are illegal

        43:45
        Score / 150 for active/ passive player

        45:47
        current bowl token value (float)

        We adapt the tensor to key from the previous TENSOR not from the previous state.
        """
        # No-op if the tensor state has already been created:
        if self.tensor_state is not None:
            return

        # if the node has a parent, that parent should have a tensor state:
        if self.parent:
            if self.parent.tensor_state is not None:
                self.tensor_state = self.create_tensor_state_from_parent()
                return

        # If no parent tensor state exists, instead use the details from the state attributes:
        board, aorder, porder, atoken, ptoken, place_bool, acolgroups, pcolgroups = self.get_state_attributes()

        ### Board tensor:
        board_tensor = EYE.index_select(1, torch.tensor(board).long().flatten())  # shape (18, 64)
        board_tensor = board_tensor.view(18, 8, 8)  # now in (C, H, W)

        ### Order tensor:
        order_list = [_or / 6.0 for _or in aorder + porder]
        order_tensor = torch.tensor(order_list).view(12, 1, 1).expand(12, 8, 8)

        ### Bowl tensor:
        bowl_list = [0] * 12
        bowl_list[atoken] = 1
        bowl_list[ptoken + 6] = 1
        bowl_tensor = torch.tensor(bowl_list).view(12, 1, 1).expand(12, 8, 8)

        ### Is no more placing flag tensor:
        placing_tensor = torch.tensor([int(place_bool)]).view(1, 1, 1).expand(1, 8, 8)

        ### Current score tensor:
        active_score = 0
        passive_score = 0

        for _col, (_aorder, _porder) in enumerate(zip(aorder, porder)):
            active_score += len(acolgroups[_col]) * _aorder
            passive_score += len(pcolgroups[_col]) * _porder

        score_tensor = torch.tensor([active_score / 150., passive_score / 150.]).view(2, 1, 1).expand(2, 8, 8)

        ### Bowl token value tensor:
        apnum, ppnum = 1, 1

        for _col, (_a, _p) in enumerate(zip(aorder, porder)):
            # increment for each time a color has been taken:
            if _a > 0:
                apnum += 1

            if _p > 0:
                ppnum += 1

        active_value = aorder[atoken] if aorder[atoken] != 0 else apnum
        passive_value = porder[ptoken] if porder[ptoken] != 0 else ppnum

        token_value_tensor = torch.tensor([active_value / 6.0, passive_value / 6.0]).view(2, 1, 1).expand(2, 8, 8)

        ### Stack the tensors:
        self.tensor_state = torch.cat([board_tensor, order_tensor, bowl_tensor, placing_tensor,
                                       score_tensor, token_value_tensor], dim=0).numpy()


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
        tensor states, values or policies assigned.

        Delete the root node, but store the necessary attributes in the replay buffer, in particular the
        tensor state, the visit counts etc.
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

        # Tensor state is created from parent tensor state, so create this before discarding parent:
        self.root_node.create_tensor_state()

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
        self.replay_buffer.append( (self.root_node.tensor_state, visit_counts) )

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
