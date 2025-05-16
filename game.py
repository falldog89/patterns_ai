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

        if clone_game is not None:
            self.clone(clone_game)
            return

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

        # The active player, that is the player about to make a move, can be 1 or -1:
        self.active_player = 1

        # track locations that have been permanently claimed by a player:
        self.flipped_locations = set()

        # Once this flag is True, no more placing actions are allowed for either player:
        self.is_no_more_placing = False

        # active and passive bowl tokens:
        self.active_bowl_token, self.passive_bowl_token = 0, 1#

        # dictionary of color: list[(int, int) coordinates showing orthogonals to existing groups:
        # occupied locations are removed.
        self.active_orthogonal_groups = {_col: set() for _col in range(6)}
        self.passive_orthogonal_groups = {_col: set() for _col in range(6)}

        # dictionary of colour: list[(int, int) locations]
        self.active_color_groups = {_col: [] for _col in range(6)}
        self.passive_color_groups = {_col: [] for _col in range(6)}

        # likewise store dictionary of color: potential flips:
        self.active_flipping_groups = {_col: set() for _col in range(6)}
        self.passive_flipping_groups = {_col: set() for _col in range(6)}

        # Track the next color-group order token to be placed:
        self.active_placing_number, self.passive_placing_number = 1, 1
        self.initialize_boards() # populate an initial random state:

    def clone(self, patterns_game: Self) -> None:
        """ populate this game with an identical copy:
        Make sure deep not shallow copy:
        """
        # Copy numpy arrays:
        self.active_board = np.array(patterns_game.active_board)
        self.passive_board = np.array(patterns_game.passive_board)

        # slice copy lists:
        self.active_color_order = patterns_game.active_color_order[:]
        self.passive_color_order = patterns_game.passive_color_order[:]

        # single numbers: no need to copy:
        self.turn_number = patterns_game.turn_number
        self.is_game_started = patterns_game.is_game_started
        self.first_turn_passed = patterns_game.first_turn_passed
        self.is_terminal = patterns_game.is_terminal
        self.result = patterns_game.result
        self.active_player = patterns_game.active_player
        self.is_no_more_placing = patterns_game.is_no_more_placing
        self.active_bowl_token = patterns_game.active_bowl_token
        self.passive_bowl_token = patterns_game.passive_bowl_token
        self.active_placing_number = patterns_game.active_placing_number
        self.passive_placing_number = patterns_game.passive_placing_number

        # set copy:
        self.flipped_locations = set(patterns_game.flipped_locations)

        # dictionary copy: must deep copy by explicitly setting lists within
        self.active_color_groups = {_col: _val[:] for _col, _val in patterns_game.active_color_groups.items()}
        self.passive_color_groups = {_col: _val[:] for _col, _val in patterns_game.passive_color_groups.items()}

        self.active_orthogonal_groups = {_col: set(_val) for _col, _val in patterns_game.active_orthogonal_groups.items()}
        self.passive_orthogonal_groups = {_col: set(_val) for _col, _val in patterns_game.passive_orthogonal_groups.items()}

        self.active_flipping_groups = {_col: set(_val) for _col, _val in patterns_game.active_flipping_groups.items()}
        self.passive_flipping_groups = {_col: set(_val) for _col, _val in patterns_game.passive_flipping_groups.items()}

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
            # if the active bowl token color group has not yet been taken:
            if self.active_color_order[self.active_bowl_token] == 0:
                # place in any unoccupied spot:
                empty_spaces = (set(location_to_coordinates) - self.flipped_locations
                                - self.passive_orthogonal_groups[self.active_bowl_token])

            else:
                # place next to the existing group:
                empty_spaces = (self.active_orthogonal_groups[self.active_bowl_token]
                                - self.passive_orthogonal_groups[self.active_bowl_token])

            placing_actions = [coordinates_to_location[_coord] for _coord in empty_spaces]

            # if no placing locations are possible, then set the flag
            if not placing_actions:
                self.is_no_more_placing = True

        flipping_actions = [coordinates_to_location[_coord] + 52 for
                            _key, _val in self.active_flipping_groups.items() for _coord in _val]

        return placing_actions + flipping_actions

    def is_action_terminal(self, action: int) -> bool:
        """ bool return determine whether taking said action would end the game.

        It will end the game if, as a result of the action, the next player will not be able to take a move.
        """
        if action >= 104:
            return False

        # the location being flipped and the orthogonal neighbors:
        removed_location = location_to_coordinates[action % 52]
        removed_orthogonal = orthogonal_neighbors[removed_location]
        set_removed_location = {removed_location}
        color = self.active_bowl_token if action < 52 else self.active_board[removed_location]

        if self.is_no_more_placing:
            placing_locations = set()

        else:
            # Placing moves, flipping moves of same color, flipping moves of different colors:
            if self.passive_color_order[self.passive_bowl_token] == 0:
                placing_locations = (set(location_to_coordinates) - self.flipped_locations
                                     - self.active_orthogonal_groups[self.passive_bowl_token]
                                     - set_removed_location)

            else:
                placing_locations = (self.passive_orthogonal_groups[self.passive_bowl_token]
                                     - self.active_orthogonal_groups[self.passive_bowl_token]
                                     - set_removed_location)

        same_color_flipping_locations = self.passive_flipping_groups[color] - set_removed_location - removed_orthogonal
        different_color_flipping_locations = set([_coord for _key, _val in self.passive_flipping_groups.items()
                                                  for _coord in _val if _key != color]) - set_removed_location

        if color == self.passive_bowl_token:
            placing_locations -= removed_orthogonal

        # if there will be a single move remaining, the game is not terminal:
        if len(placing_locations) + len(same_color_flipping_locations) + len(different_color_flipping_locations) > 0:
            return False

        return True

    def swap_players(self) -> None:
        """ swap the player and update all the pointers to the correct attributes
        """
        self.turn_number += 1
        self.active_player *= -1

        # switch active and passive:
        self.active_bowl_token, self.passive_bowl_token = self.passive_bowl_token, self.active_bowl_token
        self.active_placing_number, self.passive_placing_number = self.passive_placing_number, self.active_placing_number
        self.active_color_groups, self.passive_color_groups = self.passive_color_groups, self.active_color_groups
        self.active_board, self.passive_board = self.passive_board, self.active_board
        self.active_color_order, self.passive_color_order = self.passive_color_order, self.active_color_order
        self.active_orthogonal_groups, self.passive_orthogonal_groups = self.passive_orthogonal_groups, self.active_orthogonal_groups
        self.active_flipping_groups, self.passive_flipping_groups = self.passive_flipping_groups, self.active_flipping_groups

    def step(self, action: int) -> tuple[bool, Optional[int]]:
        """ progress the state according to the action
        """
        is_game_terminal = self.is_action_terminal(action)

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

            # active bowl token -> token picked up (not flipped)
            # active board location -> active bowl token
            # passive board location -> active bowl token, as these will be added on to in update_locations()
            if action < 52:
                (self.active_bowl_token,
                 self.active_board[coords],
                 self.passive_board[coords]) = (self.active_board[coords],
                                                self.active_bowl_token,
                                                self.active_bowl_token)

            # function to update the various orthogonals and valid moves attributes:
            self.update_locations(coords)

        if is_game_terminal:
            # just being sure nothing slips through the cracks, but I believe it should not be possible to arrive
            # here without this flag set...
            self.is_no_more_placing = True

            # take actions until the game ends, without swapping players:
            actions = self.get_actions()

            while actions:
                self.step(actions[0])
                actions = self.get_actions()

            # Calculate the score:
            active_score, passive_score = self.calculate_score()
            result = 0 if active_score == passive_score else 1 if active_score > passive_score else -1

            # store the result
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
            self.active_orthogonal_groups[_col] -= coords_set
            self.passive_flipping_groups[_col] -= coords_set
            self.passive_orthogonal_groups[_col] -= coords_set

    def update_color_groups(self, coords: tuple[int, int], color: int) -> None:
        # If the token is the first in the color group, update order taken and create initial (empty) placing group:
        if len(self.active_color_groups[color]) == 0:
            self.active_color_order[color] = self.active_placing_number
            self.active_placing_number += 1

        # Add the newly placed token to the relevant color group:
        self.active_color_groups[color].append(coords)

    def update_placing_and_flipping_groups(self, coords: tuple[int, int], color: int) -> None:
        """ placing groups should contain the coordinates where the correct color bowl token could be placed,
        orthogonal to the existing group and NOT orthogonal to the opponent groups.
        """
        # add empty token orthogonals to the active orthogonal color group:
        token_orthogonals = orthogonal_neighbors[coords] - self.flipped_locations
        self.active_orthogonal_groups[color] |= token_orthogonals

        # iterate over the flipping groups for the given color for both passive and active players:
        self.active_flipping_groups[color] = set([_x for _x in self.active_orthogonal_groups[color]
                                                  if self.active_board[_x] == color
                                                  if _x not in self.passive_orthogonal_groups[color]])

        self.passive_flipping_groups[color] = set([_x for _x in self.passive_orthogonal_groups[color]
                                                   if self.active_board[_x] == color
                                                   if _x not in self.active_orthogonal_groups[color]])
