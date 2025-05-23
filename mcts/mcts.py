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

        # Flag that detects mate-in-one:
        self.winning_action_arguments: list = []
        self.losing_action_arguments: list = []

        self.child_exploitation_scores: Optional[np.ndarray] = None
        self.child_visit_counts: Optional[np.ndarray] = None
        self.tensor_state: Optional[torch.tensor] = None

        # Node search results:
        self.visit_count: int = 0
        self.accrued_reward_to_parent: float = 0.0

        # NN prediction of current state value:
        self.value_score: Optional[float] = None

        # NN policy prediction restricted to legal moves:
        self.full_policy: Optional[np.ndarray[np.double]] = None
        self.policy_vector: Optional[np.ndarray[np.double]] = None

    def populate_attributes(self) -> None:
        """ once a game is created, populate the various attributes:
        """
        if not self.game:
            raise ValueError("You should have a game if you are populating attributes...")

        self.result = self.game.result

        if self.result is not None:
            return

        self.assign_actions_and_policy()

        # MCTS attributes: note that possible actions already restricted above.
        arr_size = len(self.possible_actions)
        self.child_exploitation_scores = np.array([np.inf] * arr_size, dtype=float)
        self.child_visit_counts = np.ones(arr_size, dtype=int)
        self.create_tensor_state()

    def assign_actions_and_policy(self) -> None:
        """ check whether the full policy is assigned yet, if it is, restrict to legal actions,
        and if there are breadth restrictions in place, restrict the action space.
        """
        # the full policy should always have been assigned:
        if self.full_policy is None:
            raise ValueError(" The full policy should have been assigned before these attributes "
                             "are assigned")

        game_actions = self.game.get_actions()

        # determine which actions are terminal: do NOT curtail an action that is terminal, as these HAVE to be
        # explorable every time!
        terminal_actions = [_action for _action in game_actions if self.game.is_action_terminal(_action)]

        # restrict the full policy to legal actions, arg sort.
        restricted_policy = self.full_policy[game_actions]

        # if the search space is already suitably restrictive, do nothing:
        if len(game_actions) <= (self.breadth_restriction + self.random_restriction):
            self.policy_vector = self.numpy_softmax(restricted_policy)
            self.possible_actions = game_actions
            return

        # sort actions according to the policy prior:
        sorted_arguments = np.argsort(restricted_policy)

        # select the top n best, according to the breadth restriction:
        topn_actions = [game_actions[_arg] for _arg in sorted_arguments[:self.breadth_restriction]]

        remaining_args = sorted_arguments[self.breadth_restriction:]
        random.shuffle(remaining_args)
        random_actions = [game_actions[_arg] for _arg in remaining_args[:self.random_restriction]]

        # make sure to add the terminal actions back in, no matter what!
        self.possible_actions = list(set(topn_actions + random_actions) | set(terminal_actions))

        # first restrict policy vector to the *legal actions*:
        self.policy_vector = self.numpy_softmax(self.full_policy[self.possible_actions])

    def calculate_reward(self) -> float:
        """ The REWARD for this node. Note this is not equivalent to result.

        Reward will be 1 if result matches player, else -1.

        Note that nn_value_score attempts to recreate result, and not reward.
        """
        if self.result is not None:
            return self.result * self.active_player

        return self.value_score * self.active_player

    def calculate_exploitation_score(self) -> float:
        """ the exploitation score of a single node, taking values between -1 and 1: """
        # if this node is terminal and lost, return pos inf to the parent, to
        # force repeated exploration of this mate-in-one position:
        if self.result == -1:
            # flip sign to represent reward to parent:
            return np.inf

        return self.accrued_reward_to_parent / self.visit_count

    def calculate_child_exploration_scores(self) -> np.ndarray[float]:
        """ the exploration scores for a single node, a variation on puct scores:
        """
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

        self.populate_attributes()

        # After game is populated, can detect terminal state and return:
        if self.result is not None:
            return self

        # Games are populated only with a parent action argument and a parent to minimize copy time:
        for _it, _move in enumerate(self.possible_actions):
            new_node = Node(
                parent=self,
                parent_action_arg=_it,
                depth = self.depth + 1,
                breadth_restriction=self.breadth_restriction,
                random_restriction=self.random_restriction,
            )

            # a win should have a positive infinite score and a loss should have a negative infinite score
            if self.game.is_action_terminal(_move):
                # If the game will terminate, create game to calculate score.
                terminal_game = Patterns(self.game)
                terminal_game.step(_move)

                # set the result as would be seen IF we switched players (nodes alternate):
                new_node.result = -terminal_game.result

                # if the other player is lost after taking this step:
                if new_node.result == -1:
                    # store the argument, not the action itself:
                    self.winning_action_arguments.append(_it)

                # also try and avoid 1 move losses!
                elif new_node.result == 1:
                    self.losing_action_arguments.append(_it)

            self.children.append(new_node)

        # return a random choice from the restricted/ legal actions:
        return random.choice(self.children)

    def get_state_attributes(self) -> tuple:
        """ return the active board, the active and passive orders and the active and passive bowl tokens
        that would belong to this node.

        tuple return is:
        board, active_order, passive_order, active_token, passive_token
        """
        # if game is populated, just use the board and state attributes from own game:
        if self.game is not None:
            return (torch.tensor(self.game.active_board), self.game.active_color_order, self.game.passive_color_order,
                    self.game.active_bowl_token, self.game.passive_bowl_token)

        # collect the game and parent action argument:
        game = self.parent.game
        action = self.parent.possible_actions[self.parent_action_arg]

        # always use passive board, as we will swap players from the parent:
        board = torch.tensor(game.passive_board)

        # active and passive reversed to represent the swap in players after action:
        active_order = game.passive_color_order[:]
        passive_order = game.active_color_order[:]
        active_token = game.passive_bowl_token
        passive_token = game.active_bowl_token

        # if start of game:
        if action >= 104:
            active_token = (action + 1) % 2
            passive_token = action % 2

            return board, [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], active_token, passive_token

        # remaining actions represent a change to the board state:
        location = action % 52
        coords = loci[location], locj[location]

        # passive board location -> active bowl token, as these will be added on to in update_locations()
        if action < 52:
            # + 12 to represent that the placed token belongs to the passive player, from the point of view of this
            # node:
            passive_token, board[coords] = board[coords].item(), passive_token + 12

        # only the passive order can be updated...
        if game.active_bowl_token not in game.active_color_groups:
            passive_order[game.active_bowl_token] = game.active_placing_number

        return board, active_order, passive_order, active_token, passive_token

    def create_tensor_state(self) -> None:
        """ Assign the tensor state that will be read by the agent's NN to provide the value and policy
        values for this node.

        History is not currently included, but might be in future.

        Tensor state is a 8 x 8 x (6x3 + 6x6 + 6x6 + 2x6) binary tensor.

        The first 6x3 planes represent the board itself, with planes 0-5 denoting the presence of an unflipped token
        of that color, 6-11 representing flipped for active player of that color, and 12-17 flipped for passive player.

        This matches the numpy array, one-hot-encoded.

        The next 2 x 6 x 6 planes represent the color group order taken, for each player.

        In particular, these planes are constant 1 or constant 0.
        Planes 0-5 represent the order at which color 0 was taken for the active player
        Planes 6-11 represent the order at which color 1 was taken for the active player etc.

        Planes 36 - 41 represent the order at which color 0 was taken for the passive player.
        Planes 42 - 47 represent the order at which color 1 was taken for the passive player etc.

        Finally, the final 12 planes represent the color of the bowl token for the active player (0-5) and the
        passive player (6-11).
        """
        if self.tensor_state is not None:
            return

        board, active_order, passive_order, active_token, passive_token = self.get_state_attributes()

        # the board is just a one hot encoded version of the numpy board:
        board_tensor = torch.nn.functional.one_hot(board.long(), num_classes=18)

        # 36 channels for each player for color group: order mapping:
        order_tensor = torch.zeros((8, 8, 72), dtype=bool)
        order_indices = [_x - 1 + 6 * _it for _it, _x in enumerate(active_order + passive_order) if _x > 0]
        order_tensor[:, :, order_indices] = 1

        # bowl tokens: 12 additional channels.
        bowl_tensor = torch.zeros((8, 8, 12), dtype=bool)
        bowl_tensor[:, :, active_token] = 1
        bowl_tensor[:, :, 6 + passive_token] = 1

        # stack the three tensors together:
        concat_tensor = torch.cat([board_tensor, order_tensor, bowl_tensor], dim=-1)

        # and put channels first, for CNN:
        self.tensor_state = concat_tensor.permute(2, 0, 1)
        # remove the corners:
        self.remove_tensor_corners(self.tensor_state)

    def remove_tensor_corners(self, board: torch.tensor) -> None:
        """ one hot encoding will give class values to the corners. Remove them
        """
        iinds = [0] * 12
        jinds = [0, 0, 1, 7, 6, 7, 0, 1, 0, 7, 7, 6]
        kinds = [0, 1, 0, 7, 7, 6, 7, 7, 6, 0, 1, 0]
        board[iinds, jinds, kinds] = 0

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

    def get_leaf_node(self) -> Node:
        """ flow from the root node to an unexpanded leaf node, following the highest puct score:
        """
        # increment the root node visit count:
        self.root_node_explore_count += 1

        # if sufficient exploring:
        if self.root_node_explore_count >= self.required_steps:
            self.is_step_ready = True

        # if mate in 1 detected:
        if self.root_node.winning_action_arguments:
            self.is_step_ready = True

        # if this is a random player:
        if self.required_steps == 0:
            return self.root_node

        if not self.root_node.children:
            return self.root_node

        # root node has additional epsilon weighted dirichlet noise to encourage diversity in exploration:
        puct_scores = self.calculate_root_puct_scores()
        node = self.root_node

        while True:
            node = node.children[np.argmax(puct_scores)]

            if not node.children:
                break

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

    def choose_action_argument(self, temperature: Optional[float] = None) -> int:
        """ Sample from the child actions vector, from the distribution formed from the child visit counts:
        """
        if not self.root_node.possible_actions:
            raise ValueError("The game has no valid actions, and should have ended...")

        if self.root_node.winning_action_arguments:
            return np.random.choice(self.root_node.winning_action_arguments)

        # if choosing randomly, try not to choose a loser-in-1:
        if self.required_steps == 0:
            if len(self.root_node.losing_action_arguments) < len(self.root_node.possible_actions):
                # choosing arguments, not actions:
                return random.choice(list(set(range(len(self.root_node.possible_actions)))
                                          - set(self.root_node.losing_action_arguments)))

            # if no losing actions, return a random index:
            return random.randint(0, len(self.root_node.possible_actions) - 1)

        # sample an action randomly according to visit counts:
        if temperature is not None:
            selection_scores = self.root_node.child_visit_counts ** (1.0 / temperature)
            selection_scores /= selection_scores.sum()
            return np.random.choice(range(len(selection_scores)), p=selection_scores).item()

        # if no selection score, instead select the most visited child:
        return np.argmax(self.root_node.child_visit_counts).item()

    def step(self, action_argument: int) -> None:
        """ Progress the tree according to the action argument.

        We increment the visit count and create a game if there isn't one, as we never need to backprop this.
        """
        # create new game for new root:
        game = Patterns(self.root_node.game)
        action = self.root_node.possible_actions[action_argument]
        game.step(action)

        # step to new root node, taking the action selected:
        self.root_node = self.root_node.children[action_argument]
        self.root_node.game = game
        self.root_node.visit_count += 1

        # reset the dirichlet noise:
        self._noise = None

        # check the schedule again after stepping:
        self.determine_required_steps()
        self.is_step_ready = False

        # if the root node requires random moving, set the full policy to be random
        if self.required_steps == 0:
            self.root_node.full_policy = np.random.rand(107)
            self.root_node.value_score = -1. + 2. * np.random.rand()

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

        # if a child has a terminal result that gives the parent a win, the exploitation is inf.,
        # as there is no need to explore any further.
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

    def get_replay_game(self, save_depth: Optional[int]) -> tuple[list[torch.tensor], list[np.ndarray], list[int]]:
        """ Once the root node is in a terminal state, store the various states for use in the replay buffer:
        Note that root node in a terminal state => no game! So we must first step up to parent.

        save depth tells you how far from terminal state to save
        """
        result = self.root_node.result

        if result is None:
            raise ValueError("this function should only be called on a completed game!")

        # start at parent, as root node is terminal and therefore has no game:
        nod = self.root_node.parent

        save_len = nod.game.turn_number if save_depth is None else save_depth

        # save the tensor states, the visit counts, and the result
        replay_tensors = [0] * save_len
        visit_counts = [0] * save_len

        # todo: weighted average of results from a given position?
        final_results = [result] * save_len

        # work backwards up through the tree, storing the states and the visit counts:
        for _ in range(save_len - 1, -1, -1):

            # save the tensors, for training, and the final result:
            replay_tensors[_] = nod.tensor_state

            # populate full action space for the visit counts:
            full_visit_counts = np.zeros(107, dtype=int)
            full_visit_counts[nod.possible_actions] = nod.child_visit_counts
            visit_counts[_] = full_visit_counts
            nod = nod.parent

        return replay_tensors, visit_counts, final_results
