""" search tree code following MCTS algortithm, adapted to patterns.

Extensions:
1. Option for random play
2. action space restricted to promote depth over breadth
    - what about missing the terminal states... we could have a " is action terminal" checked
        for each action in the action space, and ensure that is collected?
3. Schedules for these.
4. Schedules for exploration steps to spend the biccies deeper rather than early on, until the network
    is learning something...

Extension 2.
At the beginning of the learning process, when the network has not yet understood what
is or is not a good position, the games can be played randomly. As more understanding is built up, and the network
is better, the games should be played progressively less randomly. In this way, we reduce time spent on inference
results before they are useful.

This requires two things to work
1. we must make sure that NONE of the nodes are exploring with tensor, as this will remove the batching speed up
    - one approach here, given there is no speed up associated with random play, is to assign the random policy
        without switching back to the agent?

3. first session has random play anyway
4. still need to assign random tensor to still restrict the exploration...

Extension 4.
Come up with a schedule for exploration steps that scales with depth. Ie start off with VERY few explorations,
then do progressively more as you get deeper. Eg 10 explore, 20, 30 etc.

Okay so todo is:
2. random play
    For random play, we want to
        avoid creating a tensor
        avoid evaluation
        STILL respect the restriction of the action space
        assign a random full_policy and value?
        NOT do all the exploring
            we don't want to train from the random data
            we don't want to spend resources playing like that

            So I think we want to use random play with single look ahead. That is, scan for a terminal in one,
            choose the best.
3. schedule for exploration steps

"""

import numpy as np
import random
from typing import Optional, Self

import torch

from game import Patterns
from int_to_board import loci, locj


class Node:
    """ search tree node for patterns. Root node is populated with full policy and a game so that it can
    populate the attributes correctly on first expansion.
    """
    def __init__(self,
                 parent_action_arg: Optional[int] = None,
                 game: Optional[Patterns] = None,
                 parent: Optional[Self] = None,
                 depth: int = 0,
                 breadth_restriction: Optional[int] = None,
                 random_restriction: Optional[int] = None,
                 ) -> None:
        # Note: every node must have either a parent action argument or a game.
        self.parent_action_arg = parent_action_arg
        self.game = game
        self.parent = parent
        self.depth = depth

        # parameters to reduce the breadth of the search and prioritise depth instead.
        self.breadth_restriction = breadth_restriction
        self.random_restriction = random_restriction

        self.children: list[Self] = []
        self.active_player = 1 if parent is None else -1 * parent.game.active_player

        self.possible_actions: Optional[list[int]] = None
        self.result: Optional[int] = None # trichotomy

        # List of the action arguments that result in terminal positions in each direction:
        self.winning_action_arguments: list = []
        self.losing_action_arguments: list = []

        # q: is it worth removing the tensor state once it has been used up to avoid too much memory use?
        self.child_exploitation_scores: Optional[np.ndarray] = None
        self.child_visit_counts: Optional[np.ndarray] = None
        self.tensor_state: Optional[torch.tensor] = None

        # Node search results:
        self.visit_count: int = 0
        self.accrued_reward: float = 0.0

        # NN prediction of current state value:
        self.value_score: Optional[float] = None

        # For look ahead, must initially store full policy, as only upon expansion can the full policy be
        # restricted to the legal restricted move set:
        self.full_policy: Optional[np.ndarray[np.double]] = None
        self.policy_vector: Optional[np.ndarray[np.double]] = None

        # flag so that random trees remember to assign the children and so on correctly.
        self.is_assigned_actions = False

    def populate_attributes(self) -> None:
        """ once a game is created, populate the various attributes:
        NOTE that it should ALREADY HAVE a tensor state by here!
        """
        if not self.game:
            raise ValueError("You should have a game if you are populating attributes...")

        self.result = self.game.result

        if self.result is not None:
            return

        self.assign_actions_and_policy()
        self.is_assigned_actions = True

        # MCTS attributes: note that possible actions already restricted above.
        arr_size = len(self.possible_actions)
        self.child_exploitation_scores = np.array([np.inf] * arr_size, dtype=float)
        self.child_visit_counts = np.ones(arr_size, dtype=int)
        # self.create_tensor_state()

    def assign_actions_and_policy(self) -> None:
        """ check whether the full policy is assigned yet, if it is, restrict to legal actions,
        and if there are breadth restrictions in place, restrict the action space.
        """
        # the full policy should always have been assigned:
        if self.full_policy is None:
            raise ValueError(" The full policy should have been assigned before these attributes "
                             "are assigned")

        # all the legal actions that can be played, without restriction:
        game_actions = self.game.get_actions()

        # Ensure that a terminal action is never restricted, as it is vital that these are always explored:
        terminal_actions = [_action for _action in game_actions if self.game.is_action_terminal(_action)]

        # restrict the full policy to legal actions only:
        restricted_policy = self.full_policy[game_actions]

        # If legal action space sufficiently small, do not restrict further:
        if len(game_actions) <= (self.breadth_restriction + self.random_restriction):
            self.policy_vector = self.numpy_softmax(restricted_policy)
            self.possible_actions = game_actions
            return

        # sort actions according to the policy prior, to give the top n actions:
        sorted_arguments = np.argsort(restricted_policy)

        # select the top n best, according to the breadth restriction:
        topn_actions = [game_actions[_arg] for _arg in sorted_arguments[:self.breadth_restriction]]

        # select further actions randomly:
        remaining_args = sorted_arguments[self.breadth_restriction:]
        np.random.shuffle(remaining_args)
        random_actions = [game_actions[_arg] for _arg in remaining_args[:self.random_restriction]]

        # make sure to add the terminal actions back in, no matter what!
        self.possible_actions = list(set(topn_actions + random_actions) | set(terminal_actions))

        # first restrict policy vector to the *legal actions*:
        self.policy_vector = self.numpy_softmax(self.full_policy[self.possible_actions])

    def calculate_reward(self) -> float:
        """ If game is terminal, return result, else return the value judgement of the network
         1  => active player is winning/ won
        -1  => active player is losing/ lost
        """
        # If the result is not None, the game is terminal and the reward is known:
        if self.result is not None:
            return self.result

        # the network tries to predict the value to the current (active) player:
        return self.value_score

    def calculate_exploitation_score(self) -> float:
        """ the average result of a game passing through this node, normalized by the visit count.
        """
        if self.result is not None:
            return self.result

        return self.accrued_reward / self.visit_count

    def calculate_child_exploration_scores(self) -> np.ndarray[float]:
        """ the exploration scores for a single node, a variation on puct scores:
        """
        # todo understand the scaling of this better: sqrt on top feels off:
        return self.policy_vector * (self.visit_count ** 0.5) / self.child_visit_counts

    @staticmethod
    def numpy_softmax(logits: np.ndarray[float]) -> np.ndarray[float]:
        """ numpy implementation given the slowness of torch tensors for allocation
        """
        exp_demaxed_logits = np.exp(logits - np.max(logits))
        return exp_demaxed_logits / exp_demaxed_logits.sum()

    def expand(self) -> Self:
        """ expand this node by creating and assigning child nodes:
        """
        # If the leaf state is terminal, do not expand. Note that terminal states never require a game.
        if self.result is not None:
            return self

        # 1st visit, create a tensor state only, based on parent:
        if self.visit_count == 0:
            return self

        # 2nd visit, create own game if one doesn't exist and populate attributes:
        if not self.game:
            self.game = Patterns(self.parent.game)
            action = self.parent.possible_actions[self.parent_action_arg]
            self.game.step(action)

        return self.populate_attributes_and_assign_children()

    def populate_attributes_and_assign_children(self) -> Self:
        """ after 2nd visit, populate all the attributes and then return a random child
        """
        self.populate_attributes()

        # Games are populated only with a parent action argument and a parent to minimize copy time:
        for _it, _move in enumerate(self.possible_actions):
            new_node = Node(
                parent=self,
                parent_action_arg=_it,
                depth = self.depth + 1,
                breadth_restriction=self.breadth_restriction,
                random_restriction=self.random_restriction,
            )

            # Detect games where the _move will result in a terminal state:
            # IMPORTANT: if a move results in a terminal step, the PLAYER WILL NOT SWAP!
            # therefore, do not view -1 as a win or 1 as a loss...
            if self.game.is_action_terminal(_move):
                # To determine win, loss or draw, we must create a game:
                terminal_game = Patterns(self.game)
                terminal_game.step(_move)
                new_node.result = terminal_game.result

                # As the player is not swapped after a terminal action, 1 is win, -1 is loss:
                if new_node.result == 1:
                    # store the argument, not the action itself:
                    self.winning_action_arguments.append(_it)

                # if the active player of the child node wins after this action, this node must view it as a loss:
                elif new_node.result == -1:
                    self.losing_action_arguments.append(_it)

            self.children.append(new_node)

        # return a random choice from the restricted/ legal actions:
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

        # Whether flipping or placing, the location takes on the +12 of the passive player:
        board[coords] += 12

        # only the passive order can be updated...
        if not game.active_color_groups[game.active_bowl_token]:
            passive_order[game.active_bowl_token] = game.active_placing_number

        return (board, active_order, passive_order, active_token, passive_token, game.is_no_more_placing,
                active_color_groups, passive_color_groups)

    def create_tensor_state(self) -> None:
        """ Adaptation from original idea.

        Rather than 72 planes (6 colors by 6 orders by 2 players) we reduce this to 12 float layers (6 colors by
        2 players).

        Here, the value is normalized for the order in which it was taken (0 - 6).

        We further add in score, which is the current score for the active player in this state. That is,
        order * number taken

        The final number of layers is then:

        18 for pieces (board) BOOL
        12 for orders FLOAT
        12 for bowl tokens BOOL
        2 for score FLOAT
        2 for current bowl token value
        1 for is no more placing BOOL

        = 47 layers instead of the 108 or whatever above. Further, the order actually makes SENSE to be increasing. It
        isn't just representing a different class.

        Note we can also, long term, introduce move history, the redundancy of which might be useful.
        """
        if self.tensor_state is not None:
            return

        board, aorder, porder, atoken, ptoken, place_bool, acolgroups, pcolgroups = self.get_state_attributes()

        # the board is just a one hot encoded version of the numpy board. value 18 (19th class) is board corners.
        board_tensor = torch.nn.functional.one_hot(
            torch.tensor(board).long(), num_classes=19)[:, :, :-1]

        # 36 channels for each player for color group: order mapping:
        order_tensor = torch.zeros((8, 8, 12), dtype=torch.float)
        order_values = torch.tensor(aorder + porder, dtype=torch.float) / 6.0 # normalize to between 0.0 and 1.0:
        order_tensor[:, :, range(12)] = order_values

        # bowl tokens: 12 additional channels.
        bowl_tensor = torch.zeros((8, 8, 12), dtype=torch.bool)
        bowl_tensor[:, :, atoken] = 1
        bowl_tensor[:, :, 6 + ptoken] = 1

        # is no more placing:
        placing_tensor = torch.ones((8, 8, 1), dtype=torch.bool) if place_bool else torch.zeros((8, 8, 1), dtype=torch.bool)

        # also capture the current score:
        score_tensor = torch.zeros((8, 8, 2), dtype=torch.float)
        active_score = 0
        passive_score = 0

        for _col, (_aorder, _porder) in enumerate(zip(aorder, porder)):
            active_score += len(acolgroups[_col]) * _aorder
            passive_score += len(pcolgroups[_col]) * _porder

        score_tensor[:, :, 0] = active_score / 150.
        score_tensor[:, :, 1] = passive_score / 150.

        # additional redundant information that may help: the current value of the token in hand:
        token_value_tensor = torch.zeros((8, 8, 2), dtype=torch.float)

        # determine the placing numbers:
        apnum, ppnum = 1, 1
        for _col, (_a, _p) in enumerate(zip(aorder, porder)):
            # increment for each time a color has been taken:
            if _a > 0:
                apnum += 1

            if _p > 0:
                ppnum += 1

        active_value = aorder[atoken] if aorder[atoken] != 0 else apnum
        passive_value = porder[ptoken] if porder[ptoken] != 0 else ppnum

        token_value_tensor[:, :, 0] = active_value / 6.0
        token_value_tensor[:, :, 1] = passive_value / 6.0

        # stack the three tensors together:
        concat_tensor = torch.cat([board_tensor, order_tensor, bowl_tensor, placing_tensor, score_tensor,
                                   token_value_tensor], dim=-1)

        # and put channels first, for CNN:
        self.tensor_state = concat_tensor.permute(2, 0, 1)

    def get_replay_state(self) -> tuple:
        """ returns everything needed to create the game state from this node, for use in the replay buffer
        """
        _g = self.game
        return (_g.active_board,
                _g.active_color_order,
                _g.passive_color_order,
                _g.active_bowl_token,
                _g.passive_bowl_token
                )

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
                 breadth_restriction: Optional[int] = 4,
                 random_restriction: Optional[int] = 4,

                 # schedule for explorations steps scaling with tree depth:
                 schedule: Optional[list[tuple]] = None,
                 ) -> None:

        # the game and the root node of the search tree
        if root_node is None:
            game = Patterns()
            root_node = Node(
                game=game,
                breadth_restriction=breadth_restriction,
                random_restriction=random_restriction
            )

        self.root_node = root_node

        self.breadth_restriction = breadth_restriction
        self.random_restriction = random_restriction

        self.tree_id = tree_id

        # noise property only added to root node exploration:
        self._noise = None

        # Constants:
        self.puct_constant = puct_constant
        self.dirichlet_noise_level = dirichlet_noise_level
        self.dirichlet_noise_epsilon = dirichlet_noise_epsilon

        # tree-level flag that dictates when a tree is done exploring and wishes to move on
        self.is_step_ready = False

        # save the schedule to pass down to the next root nodes, use it to determine the required exploration steps
        self.schedule = schedule
        self.root_node_explore_count: int = 0
        self.required_steps: int = 0

        self.determine_required_steps()

    def reset(self, root_node: Optional[Node] = None) -> None:
        """ take the tree back to the initial position:
        """
        if not root_node:
            new_game = Patterns()
            root_node = Node(game=new_game)

        self.root_node = root_node
        self.root_node_explore_count = 0

        # and reset the noise:
        self._noise = None

        # set the required exploration steps again, according to the same schedule:
        self.determine_required_steps()

    def determine_required_steps(self) -> None:
        """ use the schedule to determine how many root node explores this node should have:

        Schedule always starts with (0, X) to state that there are X required steps at depth 0

        Then either there is no other tuple, in which case all root nodes explore for X, or there are
        other schedules, and as soon as your depth is below the first entry, you take the previous
        """
        curr_explore = self.schedule[0][1]

        for _depth, _steps in self.schedule:
            if self.root_node.depth < _depth:
                break

            curr_explore = _steps

        self.required_steps = curr_explore

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
        if not self.root_node.possible_actions:
            # This indicated that we have arrived here in a random game where this node has
            # not been expanded correctly. We need to assign the actions
            if not self.root_node.is_assigned_actions:
                # If the root node has not expanded fully yet, complete here:
                _ = self.root_node.populate_attributes_and_assign_children()

            else:
                raise ValueError("The game has no valid actions, and should have ended...")

        # If there is a winning move, the tree should take that action:
        if self.root_node.winning_action_arguments:
            return np.random.choice(self.root_node.winning_action_arguments)

        # If there are any non-losing moves, mask out 1-move losses:
        if len(self.root_node.losing_action_arguments) < len(self.root_node.possible_actions):
            # remove losing arguments from the list:
            okay_arguments = list(set(range(len(self.root_node.possible_actions)))
                                  - set(self.root_node.losing_action_arguments))

        else:
            # if no choice, just return a random loss:
            return random.choice(self.root_node.losing_action_arguments)

        # if choosing randomly:
        if self.required_steps == 0:
            return random.choice(okay_arguments)

        filtered_visit_counts = self.root_node.child_visit_counts[okay_arguments]

        # sample an action randomly according to visit counts:
        if temperature is not None:
            selection_scores = filtered_visit_counts ** (1.0 / temperature)
            selection_scores /= selection_scores.sum()
            return np.random.choice(okay_arguments, p=selection_scores).item()

        # if no selection score, instead select the most visited child that isn't a loss:
        _argmax = np.argmax(filtered_visit_counts)
        return okay_arguments[_argmax]

    def step(self, action_argument: int) -> None:
        """ Progress the tree according to the action argument.

        We increment the visit count and create a game if there isn't one, as we never need to backprop this.

        Note that when a tree is taking RANDOM moves, it will be moving to
        a child that has not been visited or seen before. These children will not have
        tensor states, values or policies assigned.

        """
        # create new game for new root:
        game = Patterns(self.root_node.game)
        action = self.root_node.possible_actions[action_argument]
        game.step(action)

        # step to new root node, taking the action selected:
        self.root_node = self.root_node.children[action_argument]
        self.root_node.game = game
        self.root_node_explore_count = self.root_node.visit_count

        # reset the dirichlet noise:
        self._noise = None

        # check the schedule again after stepping:
        self.determine_required_steps()
        self.is_step_ready = False

        # if the root node requires random moving, set the full policy to be random
        if self.required_steps == 0:
            self.root_node.full_policy = np.random.rand(107)
            self.root_node.value_score = -1. + 2. * np.random.rand()

    def get_leaf_node(self) -> Node:
        """ flow from the root node to an unexpanded leaf node, following the highest puct score:

        If a winning argument is presented, it is taken,
        If a losing argument is not necessary, it is not taken.
        """
        # increment the root node EXPLORE count.
        self.root_node_explore_count += 1

        # if enough exploration steps have been taken, prepare to step the root node:
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

        return node

    def next_node(self, node: Node) -> Node:
        """ get the next node in order, avoiding 1 move losses if possible and taking 1 move wins:
        Follow the best puct score.
        """
        # Step to a winning node if possible:
        if node.winning_action_arguments:
            arg = random.choice(node.winning_action_arguments)
            return node.children[arg]

        # If there are any non-losing moves, mask out 1-move losses:
        if len(node.losing_action_arguments) < len(node.possible_actions):
            # calculate the puct scores:
            puct_scores = self.calculate_child_puct_scores(node)

            # determine which arguments are losing:
            not_loss_args = list(set(range(len(node.possible_actions))) - set(node.losing_action_arguments))

            # mask out the losing puct scores, and choose the best remaining:
            filtered_puct = puct_scores[not_loss_args]
            _argmax = np.argmax(filtered_puct)

            return node.children[not_loss_args[np.argmax(filtered_puct)]]

        # if no choice, just return a random child:
        return random.choice(node.children)

    def calculate_child_puct_scores(self, node: Node) -> np.ndarray[float]:
        """ determine whether the node is the root node or not, and return the puct scores
        accordingly
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

        Note that the child visit counts of zero would result in a NaN value.
        Therefore, this is initialized to 1, and the understanding is that the
        addition for child puct scores will grant the necessary inf value
        """
        if node.parent is None:
            return

        parent = node.parent

        # exploitation score is the result if terminal or the average reward otherwise (accrued reward / n):
        parent.child_exploitation_scores[node.parent_action_arg] = -node.calculate_exploitation_score()
        parent.child_visit_counts[node.parent_action_arg] = node.visit_count

    def back_propagate(self, node: Node) -> None:
        """ push the result of the leaf node rollout up the tree to the root node,
        updating the reward and visit counts of the nodes on this path.
        """
        # Reward is the result if terminal else the active player value judgement of the network:
        reward = node.calculate_reward()

        # If the node was terminal, jump to parent first and march up from there.
        if node.result is not None:
            # Note : If there is a terminal action from a given node (one identified as ending the game) then the game
            # after the move will have the SAME PLAYER as the current player.
            node.visit_count += 1
            node = node.parent

        # stop once you reach and update the root node, ie when the node is now the parent of the root node.
        while node is not self.root_node.parent:
            # increment the reward and the visit count:
            node.accrued_reward += reward
            node.visit_count += 1

            # Update the nodes view of the children, including of the winning and losing actions:
            self.update_parent_child_scores(node)

            # Step to parent and flip the reward:
            node = node.parent
            reward *= -1

    def get_replay_game(self,
                        save_depth: Optional[int],
                        is_save_nodes: bool=False) -> tuple[list[tuple], int]:
        """ Once the root node is in a terminal state, store the various states for use in the replay buffer:
        Note that terminal states do not create games, so we must first step up to parent.

        Save depth tells you how far from terminal state to bother saving. At early training, we wouldn't
        bother training on the very, very weak signal from near the starting position, for example.

        For now, only splitting by win loss draw?  To balance the target of -1, 0, 1.
        Store the other meta information for later use, and the trainer can choose how to use that data:

        NOTE final result is the result from the point of view of the parent of the root node.

        In particular, the search tree progresses until the root node is in a terminal state
        THEN it seeks to store the necessary number of replay steps back from this terminal state
        As we aren't interested in training on the
        """
        # Recall that after an action is terminal, the player doesn't swap, so the parent of the root node
        # will have the same result:
        final_result = self.root_node.result

        if final_result is None:
            raise ValueError("this function should only be called on a completed game!")

        # start at parent of root to avoid training on terminal states:
        nod = self.root_node.parent
        save_len = nod.game.turn_number if save_depth is None else save_depth

        # the buffer is saved in an ordered tuple - (state tensor, visit counts array, additional information):
        _replay = []

        # work backwards up through the tree, storing the states and the visit counts:
        for _ in range(save_len):
            # child visit counts - target for policy:
            full_visit_counts = np.zeros(107, dtype=int)  # full action space

            # when saving visit counts, boost winning arguments, minimize losing ones, and make sure all winning arguments
            # are viewed as equally good!
            if nod.winning_action_arguments:
                for _ in nod.winning_action_arguments:
                    full_visit_counts[nod.possible_actions[_]] = 1

            # we can minimize the effects of the losing arguments by assigning a minimum value of 1 and boosting all
            # other legal actions:
            else:
                # otherwise, use the visit counts from the search tree:
                full_visit_counts[nod.possible_actions] = nod.child_visit_counts

                if nod.losing_action_arguments:
                    # if losing arguments, boost all arguments then down boost the losing ones:
                    full_visit_counts *= 10000

                    # iterate over losing arguments and set to mimimum value:
                    for _ in nod.losing_action_arguments:
                        full_visit_counts[nod.possible_actions[_]] = 1

            # result for this node:
            nod_result = final_result if nod.active_player == self.root_node.active_player else -1 * final_result

            # number of flipped tokens:
            flipped_num = len(nod.game.flipped_locations)

            # state, visit counts, additional information:
            save_list = [nod.tensor_state, full_visit_counts, _, flipped_num, nod_result]

            # additional debug info if required:
            if is_save_nodes:
                save_list.append(nod)

            # save to the buffer:
            _replay.append(tuple(save_list))

            # continue up the tree:
            nod = nod.parent

        return _replay, final_result

### used to be the way of a node making a tensor state:
    # def _OLDcreate_tensor_state(self) -> None:
    #     """ Assign the tensor state that will be read by the agent's NN to provide the value and policy
    #     values for this node.
    #
    #     History is not currently included, but might be in future.
    #
    #     Tensor state is a 8 x 8 x (6x3 + 6x6 + 6x6 + 2x6) binary tensor.
    #
    #     The first 6x3 planes represent the board itself, with planes 0-5 denoting the presence of an unflipped token
    #     of that color, 6-11 representing flipped for active player of that color, and 12-17 flipped for passive player.
    #
    #     This matches the numpy array, one-hot-encoded.
    #
    #     The next 2 x 6 x 6 planes represent the color group order taken, for each player.
    #
    #     In particular, these planes are constant 1 or constant 0.
    #     Planes 0-5 represent the order at which color 0 was taken for the active player
    #     Planes 6-11 represent the order at which color 1 was taken for the active player etc.
    #
    #     Planes 36 - 41 represent the order at which color 0 was taken for the passive player.
    #     Planes 42 - 47 represent the order at which color 1 was taken for the passive player etc.
    #
    #     The next 12 planes represent the color of the bowl token for the active player (0-5) and the
    #     passive player (6-11).
    #
    #     Finally, we include a plane that dictates whether or not there is any more placing:
    #     """
    #     if self.tensor_state is not None:
    #         return
    #
    #     board, aorder, porder, atoken, ptoken, place_bool, acolgroups, pcolgroups = self.get_state_attributes()
    #
    #     # the board is just a one hot encoded version of the numpy board. value 18 (19th class) is board corners.
    #     board_tensor = torch.nn.functional.one_hot(board.long(), num_classes=19)[:, :, :-1]
    #
    #     # 36 channels for each player for color group: order mapping:
    #     order_tensor = torch.zeros((8, 8, 72), dtype=bool)
    #     order_indices = [_x - 1 + 6 * _it for _it, _x in enumerate(aorder + porder) if _x > 0]
    #     order_tensor[:, :, order_indices] = 1
    #
    #     # bowl tokens: 12 additional channels.
    #     bowl_tensor = torch.zeros((8, 8, 12), dtype=bool)
    #     bowl_tensor[:, :, atoken] = 1
    #     bowl_tensor[:, :, 6 + ptoken] = 1
    #
    #     # is no more placing:
    #     placing_tensor = torch.ones((8, 8, 1), dtype=bool) if place_bool else torch.zeros((8, 8, 1), dtype=bool)
    #
    #     # stack the three tensors together:
    #     concat_tensor = torch.cat([board_tensor, order_tensor, bowl_tensor, placing_tensor], dim=-1)
    #
    #     # and put channels first, for CNN:
    #     self.tensor_state = concat_tensor.permute(2, 0, 1)
    #
    #     ### no longer necessary as corners set to value 18 and removed in 1 hot:
    #     # remove the corners:
    #     # self.remove_tensor_corners(self.tensor_state)
    #
    # # def remove_tensor_corners(self, board: torch.tensor) -> None:
    # #     """ one hot encoding will give class values to the corners. Remove them
    # #     """
    # #     iinds = [0] * 12
    # #     jinds = [0, 0, 1, 7, 6, 7, 0, 1, 0, 7, 7, 6]
    # #     kinds = [0, 1, 0, 7, 7, 6, 7, 7, 6, 0, 1, 0]
    # #     board[iinds, jinds, kinds] = 0


# # todo check this can be removed? Already checking for terminal actions which should provide the result:
# # in particular, new nodes are populated with a result if the action would be terminal
# # After game is populated, can detect terminal state and return:
# if self.result is not None:
#     return self
