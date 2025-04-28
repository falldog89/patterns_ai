"""
The main game file for patterns.

#####################
State explanation:
state is a list of size 66:

Entries 0-51 represent the state of locations 1-52 on the boards:

0-5 = unflipped colors
6-11 = flipped for ACTIVE player
12-17 = flipped for PASSIVE player

By having active and passive players in this way, we avoid having to inform whos turn it is. It is
always the turn of the active player.

Entries 52-63 represent the order in which each player took a color

52-57 represent colors 0-5 for the active player and 0 means untaken, 1-6 mean taken in that order.
58-63 represent colors 0-5 for passive player and 0 means untaken, 1-6 means taken in that order.

64 represents the hand token of the active player, values 0-5.
65 represents the hand token of the passive player, values 0-5.

Note that there is a slight oddness in that the initial state will have 0 0 for 64-65, but
we assume that this is inconsequential for choosing initial move.
#################


#####################
Action space: 107 distinct actions.
 #  0 -  51:  place bowl token in given location
        # 52 - 103: flip piece at given location
        #      104: choose color 0
        #      105: choose color 1
        #      106: choose to play first and defer first turn

A list of 106 actions that can ever be taken.

0   -  51: Place the active bowl token at the indicated location,
52  - 103: Flip the token at the indicated location (action % 52)
      104: Choose to stick with bowl token 0 on turn 1.
      105: Choose swap bowl tokens on turn 1.
      106: Choose to play first, force player 2 to choose bowl token.

NOTE:
Once a player has no available placing moves on their turn, only flipping moves are allowed from that point on.

Once a player cannot flip anymore, all remaining moves are taken in one go. Ie all remaining possible
flips are flipped.
#####################

#####################
State space: list of size 66 (may change tensor representation):

Indices 0 - 51 represent the state of the tokens at locations 1-52 on the board.
 0 -  5: unflipped token of this color
 6 - 11: flipped token of (this % 6) color for the ACTIVE player
12 - 17: flipped token of (this % 6) color for the PASSIVE player

Indices 52 - 57 represent the order taken of each color for the ACTIVE player.
52: Color 0 was not taken yet (0) or was taken first (1) to sixth (6) for the ACTIVE player
...
57: Color 5 was not taken yet (0) or was taken first (1) to sixth (6) for the ACTIVE player

Indices 58 - 63 represent the order taken of each color for the PASSIVE player.

Index 64 represent the color of the bowl token for the active player. Before choice, set to 0
Index 65 represents the color of the bowl token for the passive player. Before choice, set to 1
#####################

todo
if placing and flipping have the same affect, BAN the placing action to reduce the action space.
move to enums for the colors will be easier.
"""
import random
import numpy as np
from int_to_board import all_neighbours, loci, locj


class Patterns:
    def __init__(self):
        self.action_space = (107,)
        self.state_space = (66,)

        # track the board as well, as this will be used in the NN implementation:
        self.active_board = np.zeros((8, 8), dtype=int)
        self.passive_board = np.zeros((8, 8), dtype=int)
        self.active_color_order = [0] * 6
        self.passive_color_order = [0] * 6

        self.turn_number = 1

        # The active player - can be 0 or 1.
        self.player = 0

        # track locations that have been permanently claimed by a player:
        self.flipped_locations = set()

        # Once this flag is True, no more placing actions are allowed for either player:
        self.is_no_more_placing = False

        # active and passive bowl tokens:
        self.active_hand_piece, self.passive_hand_piece = 0, 1

        # dictionary of colour: list[int locations]
        self.active_color_groups = {_col: [] for _col in range(6)}
        self.passive_color_groups = {_col: [] for _col in range(6)}

        # Track locations that are orthogonal to taken color groups:
        self.active_orthogonals = {}
        self.passive_orthogonals = {}

        # Track the next color-group order token to be placed:
        self.active_placing_number, self.passive_placing_number = 1, 1

        # Note that the active and passive states will differ. More efficient to track separately.
        self.active_state = [0] * 66
        self.passive_state = [0] * 66

        self.initialize_states() # populate an initial random state:

    def initialize_states(self):
        """ create an initial random assortment of colours, missing 1, 2 in the middle:
        """
        # There are 9 of each color in total, but the locations of the final token in each color are either in the
        # center, or in the bowls.
        tiles = list(range(6)) * 8
        random.shuffle(tiles)

        # middle tiles always start with colors 2 - 5, with 0 and 1 starting in the player bowls WLOG:
        middle = [2, 3, 4, 5]
        random.shuffle(middle)
        initial_state = (
                tiles[:21] + middle[:2] + tiles[21:27] + middle[2:] + tiles[27:]
                + [0] * 12 + [0, 1]
        )

        self.active_board[loci, locj] = initial_state
        self.passive_board[loci, locj] = initial_state

        self.active_state = [_ for _ in initial_state]
        self.passive_state = [_ for _ in initial_state]

    def get_actions(self) -> list[int]:
        """ a legal action is:
        1. Place the hand piece you have on the board
        2. Flip over a piece next to one of your own
        """
        # Either choose a color or pass the choice:
        if self.turn_number == 1:
            return [104, 105, 106]

        # if a hand piece has not been chosen yet, the first player passed the choice:
        if self.active_hand_piece is None:
            return [104, 105]

        if self.is_no_more_placing:
            placing_actions = []

        else:
            placing_actions = self.get_placing_actions()

            # if no placing locations are possible, then set the flag
            if not placing_actions:
                self.is_no_more_placing = True

        flipping_actions = self.get_flipping_actions()

        # todo: remove placing actions that are contained implicitly in the flipping actions. That is,
        # if your placing action would result in the same state as the flipping action

        return placing_actions + flipping_actions

    def get_placing_actions(self) -> list[int]:
        """ Rules:
        1. you cannot place a piece adjacent to an opponents piece of the same color
        2. you cannot place a piece in a location containing another flipped piece of any type
        3. A piece sharing a color with an existing color group may only be placed adjacent to a piece from
                that color group.

        Return list of the INTEGER LOCATIONS, not the actions that these relate to.
        In particular, return list of all locations orthogonal to the active color group of this color, remove
        those locations that are flipped and remove those locations that are adjacent to the passive color group
        """
        return list(self.active_orthogonals.get(self.active_hand_piece, set(range(52)))
                    - self.passive_orthogonals.get(self.active_hand_piece, set())
                    - self.flipped_locations)

    def get_flipping_actions(self) -> list[int]:
        """ Flips are:
        1. Orthogonal to an already-flipped piece,
        2. The same colour as the flipped piece,
        3. Not next to an opponents piece of the same colour.
        """
        flip_moves = []

        # iterate over the populated color groups:
        for _color, orthogonal_locations in self.active_orthogonals.items():
            # unflipped orthogonals of the same color:
            matching_locations = [loc + 52 for loc in orthogonal_locations
                                  # todo change to board not state:
                                  if self.active_state[loc] == _color
                                  if loc not in self.passive_orthogonals.get(_color, set())]

            flip_moves.extend(matching_locations)

        return flip_moves

    def swap_players(self) -> None:
        """ swap the player and update all the pointers to the correct attributes
        """
        self.turn_number += 1
        self.player = (self.player + 1) % 2

        # switch active and passive:
        self.active_hand_piece, self.passive_hand_piece = self.passive_hand_piece, self.active_hand_piece
        self.active_placing_number, self.passive_placing_number = self.passive_placing_number, self.active_placing_number
        self.active_color_groups, self.passive_color_groups = self.passive_color_groups, self.active_color_groups
        self.active_orthogonals, self.passive_orthogonals = self.passive_orthogonals, self.active_orthogonals
        self.active_state, self.passive_state = self.passive_state, self.active_state
        self.active_board, self.passive_board = self.passive_board, self.active_board
        self.active_color_order, self.passive_color_order = self.passive_color_order, self.active_color_order

    def step(self, action: int) -> None:
        """ progress the state according to the action
        """
        location = None

        # Choose which tile you want to play with:
        if action in [104, 105]:
            # color is always either 0 or 1 by construction:
            acol = action - 104
            pcol = (action + 1) % 2

            # assign bowl pieces to each player:
            self.active_hand_piece = acol
            self.passive_hand_piece = pcol

            # update active and passive states:
            self.active_state[-2] = acol
            self.active_state[-1] = pcol
            self.passive_state[-2] = pcol
            self.passive_state[-1] = acol


        # placing moves:
        if action < 52:
            location = action

            # if piece is the first in the colour group, update ORDER TAKEN in state:
            if len(self.active_color_groups[self.active_hand_piece]) == 0:
                # update the ORDER TAKEN in state:
                # active_state_index = 52 + self.active_hand_piece
                #
                # self.active_state[active_state_index] = self.active_placing_number
                # self.passive_state[active_state_index + 6] = self.active_placing_number
                #
                self.active_color_order[self.active_hand_piece] = self.active_placing_number

                # increment the scoring number (initialized at 1):
                self.active_placing_number += 1

            # update the color group by appending location:
            self.active_color_groups[self.active_hand_piece].append(location)

            # the new hand piece becomes the color that was picked up:
            # picked_up_color = self.active_state[location]
            # self.active_state[location] = self.active_hand_piece + 6
            # self.passive_state[location] = self.active_hand_piece + 12

            # update board for both players:
            boardi, boardj = loci[location], locj[location]
            picked_up_color = self.active_board[boardi, boardj]
            self.active_board[boardi, boardj] = self.active_hand_piece + 6
            self.passive_board[boardi, boardj] = self.active_hand_piece + 12
            self.active_hand_piece = picked_up_color

            # # the state indices 64-65 represent the hand pieces:
            # self.active_state[64] = self.active_hand_piece
            # self.passive_state[65] = self.active_hand_piece

        if 52 <= action < 104:
            location = action % 52
            color_to_be_flipped = self.active_state[location]

            # if you are flipping, the color group must already be established:
            self.active_color_groups[color_to_be_flipped].append(location)

            # only thing that changes is that the color is flipped.
            self.active_state[location] += 6
            self.passive_state[location] += 12

        # add the flipped or placed location to the set of used-up values:
        if location is not None:
            self.flipped_locations.add(location)

            # set union the relevant orthogonals set:
            color_group = self.active_state[location] % 6

            # update the active orthogonals dictionary with the neighbours of the new flipped location:
            self.active_orthogonals.update(
                {color_group: self.active_orthogonals.get(color_group, set()) | all_neighbours[location]}
            )

        # swap the player attributes:
        self.swap_players()
