"""
The main game file for patterns.

### STATE #####################################
State is represented naturally through the use of an 8x8 board, with the corners ignored.

The numbers  0 -  5 represent unflipped colors.
The numbers  6 - 11 represent flipped colors for the active player.
The numbers 12 - 17 represent flipped colors for the passive player.

This board state is supplemented by 2 vectors to denote the order in which each color group is taken
by the active/ passive player. Here, 0 implies that a player has not taken that color yet, which 1-6 implies
taken at that order.

Finally, the bowl tokens for each player are stored. This completes the state store.

###############################################

### ACTION ####################################
The action space is far simpler.
0 -  51:  place bowl token in given location
52 - 103: flip piece at given location
104: choose color 0
105: choose color 1
106: choose to play first and defer first turn

NOTE:
Once a player has no available placing moves on their turn, only flipping moves are allowed from that point on.

Once a player cannot flip anymore, all remaining moves are taken in one go. Ie all remaining possible
flips are flipped.

###############################################
"""
import random
import numpy as np
from typing import Optional, Self


from int_to_board import orthogonal_neighbors, loci, locj, location_to_coordinates, coordinates_to_location


class Patterns:
    def __init__(self, clone_game: Optional[Self] = None) -> None:
        self.action_space = (107,)
        self.state_space = (66,)

        # track the board as well, as this will be used in the NN implementation:
        self.active_board = np.zeros((8, 8), dtype=int)
        self.passive_board = np.zeros((8, 8), dtype=int)
        self.active_color_order = [0] * 6
        self.passive_color_order = [0] * 6

        # setting the initial game state for first couple of turns:
        self.turn_number = 1
        self.is_game_started = False
        self.first_turn_passed = False
        self.is_terminal = False
        self.result: Optional[int] = None

        # The active player can be 1 or -1:
        self.player = 1

        # track locations that have been permanently claimed by a player:
        self.flipped_locations = set()

        # Once this flag is True, no more placing actions are allowed for either player:
        self.is_no_more_placing = False

        # active and passive bowl tokens:
        self.active_bowl_token, self.passive_bowl_token = 0, 1

        # dictionary of colour: list[(int, int) locations]
        self.active_color_groups = {_col: [] for _col in range(6)}
        self.passive_color_groups = {_col: [] for _col in range(6)}

        # store dictionary of color: potential locations for placing:
        self.active_placing_groups = {_col: set(location_to_coordinates) for _col in range(6)}
        self.passive_placing_groups = {_col: set(location_to_coordinates) for _col in range(6)}

        # likewise store dictionary of color: potential flips:
        self.active_flipping_groups = {_col: set() for _col in range(6)}
        self.passive_flipping_groups = {_col: set() for _col in range(6)}

        # and store and update all possible flipping moves, not keyed by color:
        self.active_flipping_actions= set()
        self.passive_flipping_actions = set()

        # Track the next color-group order token to be placed:
        self.active_placing_number, self.passive_placing_number = 1, 1

        if clone_game is not None:
            self.clone(clone_game)

        else:
            self.initialize_boards() # populate an initial random state:

    def clone(self, patterns_game: Self) -> None:
        """ populate this game with an identical copy:
        """
        # track the board as well, as this will be used in the NN implementation:
        self.active_board = patterns_game.active_board
        self.passive_board = patterns_game.passive_board
        self.active_color_order = patterns_game.active_color_order
        self.passive_color_order = patterns_game.passive_color_order

        # setting the initial game state for first couple of turns:
        self.turn_number = patterns_game.turn_number
        self.is_game_started = patterns_game.is_game_started
        self.first_turn_passed = patterns_game.first_turn_passed
        self.is_terminal = patterns_game.is_terminal
        self.result = patterns_game.result

        # The active player can be 1 or -1:
        self.player = patterns_game.player

        # track locations that have been permanently claimed by a player:
        self.flipped_locations = patterns_game.flipped_locations

        # Once this flag is True, no more placing actions are allowed for either player:
        self.is_no_more_placing = patterns_game.is_no_more_placing

        # active and passive bowl tokens:
        self.active_bowl_token = patterns_game.active_bowl_token
        self.passive_bowl_token = patterns_game.passive_bowl_token

        # dictionary of colour: list[(int, int) locations]
        self.active_color_groups = patterns_game.active_color_groups
        self.passive_color_groups = patterns_game.passive_color_groups

        # store dictionary of color: potential locations for placing:
        self.active_placing_groups = patterns_game.active_placing_groups
        self.passive_placing_groups = patterns_game.passive_placing_groups

        # likewise store dictionary of color: potential flips:
        self.active_flipping_groups = patterns_game.active_flipping_groups
        self.passive_flipping_groups = patterns_game.passive_flipping_groups

        # and store and update all possible flipping moves, not keyed by color:
        self.active_flipping_actions = patterns_game.active_flipping_actions
        self.passive_flipping_actions = patterns_game.passive_flipping_actions

        # Track the next color-group order token to be placed:
        self.active_placing_number = patterns_game.active_placing_number
        self.passive_placing_number = patterns_game.passive_placing_number

    def initialize_boards(self):
        """ create an initial random assortment of colours, missing 1, 2 in the middle:
        """
        # There are 9 of each color in total, but the locations of the final token in each color are either in the
        # center, or in the bowls.
        tiles = list(range(6)) * 8
        random.shuffle(tiles)

        # middle tiles always start with colors 2 - 5, with 0 and 1 starting in the player bowls WLOG:
        middle = [2, 3, 4, 5]
        random.shuffle(middle)
        initial_state = tiles[:21] + middle[:2] + tiles[21:27] + middle[2:] + tiles[27:]

        self.active_board[loci, locj] = initial_state
        self.passive_board[loci, locj] = initial_state

    def calculate_score(self) -> tuple[int, int]:
        """ return the score for each player in the current state by looking at the
        color orders and the color groups
        """
        active_score = 0
        passive_score = 0

        for _color, (aorder, porder) in enumerate(zip(self.active_color_order, self.passive_color_order)):
            active_score += len(self.active_color_groups[_color]) * aorder
            passive_score += len(self.passive_color_groups[_color]) * porder

        return active_score, passive_score

    def get_actions(self) -> list[int]:
        """ a legal action is:
        1. Place the hand piece you have on the board
        2. Flip over a piece next to one of your own
        """
        # Either choose a color or pass the choice:
        if not self.is_game_started:
            return [104, 105, 106]

        # if a hand piece has not been chosen yet, the first player passed the choice:
        if self.first_turn_passed:
            return [104, 105]

        if self.is_no_more_placing:
            placing_actions = []

        else:
            placing_actions = [coordinates_to_location[_coord]
                               for _coord in self.active_placing_groups[self.active_bowl_token]]

            # if no placing locations are possible, then set the flag
            if not placing_actions:
                self.is_no_more_placing = True

        flipping_actions = self.active_flipping_actions

        return placing_actions + list(flipping_actions)

    def swap_players(self) -> None:
        """ swap the player and update all the pointers to the correct attributes
        """
        self.turn_number += 1
        self.player *= -1

        # switch active and passive:
        self.active_bowl_token, self.passive_bowl_token = self.passive_bowl_token, self.active_bowl_token
        self.active_placing_number, self.passive_placing_number = self.passive_placing_number, self.active_placing_number
        self.active_color_groups, self.passive_color_groups = self.passive_color_groups, self.active_color_groups
        self.active_board, self.passive_board = self.passive_board, self.active_board
        self.active_color_order, self.passive_color_order = self.passive_color_order, self.active_color_order
        self.active_placing_groups, self.passive_placing_groups = self.passive_placing_groups, self.active_placing_groups
        self.active_flipping_groups, self.passive_flipping_groups = self.passive_flipping_groups, self.active_flipping_groups
        self.active_flipping_actions, self.passive_flipping_actions = self.passive_flipping_actions, self.active_flipping_actions

    def step(self, action: int) -> tuple[bool, Optional[int]]:
        """ progress the state according to the action
        """

        # Simply pass choice of initial bowl token to the other player:
        if action == 106:
            # flag to indicate it is no longer the first turn:
            self.is_game_started = True

            # set flag to indicate turn passed:
            self.first_turn_passed = True

        # Choose which tile you want to play with:
        if action in [104, 105]:
            # flag to indicate it is no longer the first turn:
            self.is_game_started = True

            # remove potential flag for token choice being passed:
            self.first_turn_passed = False

            # assign bowl pieces to each player:
            self.active_bowl_token = action % 2
            self.passive_bowl_token = (action + 1) % 2

        # actions that represent a change to the board state:
        if action < 104:
            location = action % 52
            coords = loci[location], locj[location]

            if action < 52:
                self.active_bowl_token, self.active_board[coords], self.passive_board[coords] = (self.active_board[coords],
                                                                                                 self.active_bowl_token,
                                                                                                 self.active_bowl_token)

            # function to update the various orthogonals and valid moves attributes:
            self.update_locations(coords)

        # if there are no flips and either no more placing actions OR placing actions have stopped:
        if len(self.passive_flipping_actions) == 0:
            if (len(self.passive_placing_groups[self.passive_bowl_token]) == 0) or self.is_no_more_placing:
                self.take_all_flips()
                p1_score, p2_score = self.calculate_score()
                result = 0

                if p1_score > p2_score:
                    result = 1

                if p1_score < p2_score:
                    result = -1

                self.is_terminal = True
                self.result = result

                return True, result

        # swap the player attributes:
        self.swap_players()

        return False, None

    def update_locations(self, coords: tuple[int, int]) -> None:
        """ Update the active and passive location-sensitive attributes, to
        allow the calculation of future moves to be far simpler

        1. Update the active color group, and introduce it if necessary
        2. Update the orthogonals, the placing groups and the flipping groups as necessary
        3.
        """
        # Determine the color of the token that was flipped or placed:
        token_color = int(self.active_board[coords])
        self.update_board(coords)
        self.remove_new_location(coords)
        self.update_color_groups(coords, token_color)
        self.update_placing_and_flipping_groups(coords, token_color)

    def update_board(self, coords: tuple[int, int]) -> None:
        # Flip the color that has been placed:
        self.active_board[coords] += 6
        self.passive_board[coords] += 12

        # Add the new token to the set of flipped locations:
        self.flipped_locations.add(coords)

    def remove_new_location(self, coords: tuple[int, int]) -> None:
        # Remove the flipped location from all passive and active placing and flipping locations and orthogonals:
        coords_set = {coords}

        for _col in range(6):
            self.active_flipping_groups[_col] -= coords_set
            self.active_placing_groups[_col] -= coords_set
            self.passive_flipping_groups[_col] -= coords_set
            self.passive_placing_groups[_col] -= coords_set

    def update_color_groups(self, coords: tuple[int, int], color: int) -> None:
        # If the token is the first in the color group, update order taken and create initial (empty) placing group:
        if len(self.active_color_groups[color]) == 0:
            self.active_color_order[color] = self.active_placing_number
            self.active_placing_number += 1
            self.active_placing_groups[color] = set()

        # Add the newly placed token to the relevant color group:
        self.active_color_groups[color].append(coords)

    def update_placing_and_flipping_groups(self, coords: tuple[int, int], color: int) -> None:
        # empty token orthogonals:
        token_orthogonals = orthogonal_neighbors[coords] - self.flipped_locations

        # detect spots that exist in both the passive and active placing groups:
        if self.passive_color_order[color] == 0:
            mutually_orthogonal = set()

        else:
            mutually_orthogonal = self.passive_placing_groups[color] & token_orthogonals

        # Remove the mutual orthogonals from both the passive and the active placing groups:
        self.passive_placing_groups[color] -= mutually_orthogonal
        self.passive_flipping_groups[color] -= mutually_orthogonal
        legal_places = token_orthogonals - mutually_orthogonal

        # Remove anything that doesn't match color for the active flipping group:
        legal_flips = set([_lp for _lp in legal_places if self.active_board[_lp] == color])

        # add the empty and legal places to the active placing groups:
        self.active_placing_groups[color] |= legal_places

        # add the empty, legal and matching spaces to the flipping groups:
        self.active_flipping_groups[color] |= legal_flips

        # Remove the new location action represented by the coordinates:
        self.active_flipping_actions -= {coordinates_to_location[coords] + 52}
        self.passive_flipping_actions -= {coordinates_to_location[coords] + 52}

        # add the new flips enabled by the new location:
        self.active_flipping_actions |= set([coordinates_to_location[_coord] + 52 for
                                             _coord in legal_flips])

    def take_all_flips(self) -> None:
        """ It has been determined that the passive player will have no moves remaining. Take all
        remaining flipping moves as the active player now and end the game.
        """
        for _col in range(6):
            for _location in self.active_flipping_groups[_col]:
                self.active_board[_location] += 6
                self.passive_board[_location] += 12
                self.active_color_groups[_col].append(_location)
