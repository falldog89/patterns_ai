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

NOTE the first time that a player cannot make a placing move, the game changes state; no placing move for either
player will be allowed again.

The first time this happens will be down to the active player having no placing moves, but this is not true
for subsequence players.

Therefore, the important thing is for the state to change AFTER the move?

In particular, it doesn't really matter WHEN the state changes the first time, because there IS not legal action
anyway. However, it would be nice to occur at initialization.

I think I am arguing, after all this, that the child state NEVER needs to know its own state, only the parents.
However, when the game is made and children are created that will involve checking for is no more placing,
so as long as create state happens after that, we are safe enough. It does mean that two equivalent states
can describe the same game state, but that is fine! However, it does also mean that if a game is created and
it was never queried, we might have to be careful.
"""
import random
import numpy as np
from typing import Optional, Self


from int_to_board import orthogonal_neighbors, loci, locj, location_to_coordinates, coordinates_to_location


class Patterns:
    def __init__(self, clone_game: Optional[Self] = None) -> None:
        self.action_space = (107,)

        if clone_game is not None:
            self.clone(clone_game)
            return

        # track the board as well, as this will be used in the NN implementation:
        # set to 18 rather than 0 so that corners are different from the other values
        self.active_board = 18 * np.ones((8, 8), dtype=int)
        self.passive_board = 18 * np.ones((8, 8), dtype=int)
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
        self._is_no_more_placing: Optional[bool] = None

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

        # rather than recalculating, store the calculated placing actions:
        self._placing_actions: Optional[list[int]] = None
        # distinguish between the actual placing actions, which considers is_no_more_placing, and this, which is
        # used in the definition of is_no_more_placing
        self._possible_placing_coordinates: Optional[list[tuple[int, int]]] = None
        self._flipping_actions: Optional[list[int]] = None

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
        self._placing_actions = patterns_game._placing_actions[:] if patterns_game._placing_actions is not None else None
        self._flipping_actions = patterns_game._flipping_actions[:] if patterns_game._flipping_actions is not None else None
        self._possible_placing_coordinates = (patterns_game._possible_placing_coordinates[:] if
                                              patterns_game._possible_placing_coordinates is not None else None)

        # single numbers: no need to copy:
        self.turn_number = patterns_game.turn_number
        self.is_game_started = patterns_game.is_game_started
        self.first_turn_passed = patterns_game.first_turn_passed
        self.is_terminal = patterns_game.is_terminal
        self.result = patterns_game.result
        self.active_player = patterns_game.active_player
        self._is_no_more_placing = patterns_game._is_no_more_placing
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

    @property
    def is_no_more_placing(self) -> bool:
        """ We have is_no_more_placing as a property, so that if it is ever queried and the placing actions haven't
        been checked, we check:

        When a child is created, adopt the parent is no more placing flag, unless it is false, in which case
        set it to None:
        """
        # if set to None, it means not checked yet. DO NOT use this setter if any parent game had it set to True:
        if self._is_no_more_placing is None:
            self.set_is_no_more_placing()

        return self._is_no_more_placing

    def set_is_no_more_placing(self) -> None:
        """ determine whether there are any legal placing moves if the property has not been set.
        """
        # if the placing actions haven't been checked, cannot know:
        if self._possible_placing_coordinates is None:
            self.set_possible_placing_coordinates()

        # If there are no possible legal placing actions, set the flag:
        if not self._possible_placing_coordinates:
            self._is_no_more_placing = True

        # It was previously inconclusive and placing actions still exist:
        else:
            self._is_no_more_placing = False

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
        """
        At the start of the game, a player can choose to take 0 as bowl token,
        swap for 1, or pass that choice to the other player.

        After the game begins, a player may either:
        1. Place a bowl token on the board
        2. Flip over a piece next to one of your own
        """
        # Either choose a color or pass the choice:
        if not self.is_game_started:
            return [104, 105, 106]

        # if a hand piece has not been chosen yet, the first player passed the choice:
        if self.first_turn_passed:
            return [104, 105]

        # no op if already set:
        self.set_placing_actions()
        self.set_flipping_actions()

        return self._placing_actions + self._flipping_actions

    def set_placing_actions(self) -> None:
        """ actions that involve placing the bowl token. Returns integer action, not coordinates
        """
        if self._placing_actions is not None:
            return

        if self.is_no_more_placing:
            self._placing_actions = []
            return

        # if the possible locations haven't been determined yet:
        if self._possible_placing_coordinates is None:
            self.set_possible_placing_coordinates()

        # turn the coordinates into locations for the action space:
        self._placing_actions = [coordinates_to_location[_coord] for _coord in self._possible_placing_coordinates]

    def set_possible_placing_coordinates(self) -> None:
        """ determine all the locations that are empty to be placed in, ignoring the is_no_more_placing flag
        """
        # if the active bowl token color group has not yet been taken:
        if self.active_color_order[self.active_bowl_token] == 0:
            # place in any unoccupied spot:
            empty_spaces = (set(location_to_coordinates) - self.flipped_locations
                            - self.passive_orthogonal_groups[self.active_bowl_token])

        else:
            # place next to the existing group:
            empty_spaces = (self.active_orthogonal_groups[self.active_bowl_token]
                            - self.passive_orthogonal_groups[self.active_bowl_token])

        self._possible_placing_coordinates = list(empty_spaces)

    def set_flipping_actions(self) -> None:
        """ flipping actions, which involve flipping a neutral token on the board that is adjacent to
        a taken color group of the same color and not adjacent to an opponents token of the same color.
        Additionally, flipping actions that would result in the same state as a placing action are not considered.
        That is, if the bowl token is the same color, it is equivalent to flipping.
        """
        if self._flipping_actions is not None:
            return

        # if the placing actions are not already calculated, calculate them now:
        if self._placing_actions is None:
            self.set_placing_actions()

        self._flipping_actions = [coordinates_to_location[_coord] + 52 for
                                  _key, _val in self.active_flipping_groups.items() for _coord in _val
                                  if not ((coordinates_to_location[_coord] in self._placing_actions)
                                          and (_key == self.active_bowl_token))]

    def _easy_win_placing(self, color: int, removed_location: tuple[int, int]) -> bool:
        """
        Efficiency function: it is easy to check that in most situations there is at least one move remaining
        after an action is taken, and the game is not terminal. This function is a messy composition of these checks.

        Returning True tells you that you have an easy win. No need to check further.

        Return False is an inconclusive. We do not know and further conclusive checks will be required.
        """
        # if placing moves are no longer allowed, no easy win: need to check flips.
        if self.is_no_more_placing:
            return False

        # the bowl token of the prospective player differs from the token to be flipped:
        if self.passive_bowl_token != color:

            # if the passive bowl token color has not been taken by passive yet:
            if self.passive_color_order[self.passive_bowl_token] == 0:
                # if neither player has this group, the game cannot end, as there MUST be those tokens left:
                if self.active_color_order[self.passive_bowl_token] == 0:
                    return True

                # number of the passive bowl token flipped by the active player:
                num_current_active = len(self.active_color_groups[self.passive_bowl_token])

                # the number of empty locations that cannot be orthogonal to active color group:
                # 52 locations - (number flipped - n) - (3 * n + 2) orthogonals - 1 taken location:
                if (52 - len(self.flipped_locations) - 2 * num_current_active - 3) > 0:
                    # If there is at least 1 flipping location, easy win:
                    return True

            # passive bowl token color has been taken by the passive color: they are adding to a color group:
            else:
                # active has not taken this color group yet:
                if self.active_color_order[self.passive_bowl_token] == 0:
                    # as long as there is one location that is not the removed location, placing is legal:
                    if len(self.passive_orthogonal_groups[self.passive_bowl_token] - {removed_location}) > 0:
                        return True

                # active has also taken this color:
                else:
                    # if there are at least 1 empty spots after removing the orthogonals and the removed location:
                    if len(self.passive_orthogonal_groups[self.passive_bowl_token]
                           - self.active_orthogonal_groups[self.passive_bowl_token] - {removed_location}) > 0:
                        return True

        # the token being flipped matches the passive players bowl token:
        else:
            # if the passive player has not taken this color group yet:
            if self.passive_color_order[color] == 0:
                # and neither has the active player:
                if self.active_color_order[color] == 0:
                    # must be safe to place:
                    return True

                # if the active player has an existing color group of this color:
                num_current_active = len(self.active_color_groups[color])

                # the most spots that can be blotted out are flipped locations, 3 for the new flip, (because it adds
                # to an existing group) plus twice the number of currently taken actives plus 2.
                if (52 - len(self.flipped_locations) - 2 * num_current_active - 5) > 0:
                    return True

        return False

    def _full_flipping_terminal(self,
                                color: int,
                                removed_location: tuple[int, int],
                                removed_orthogonals: set[tuple[int, int]]) -> bool:
        """ detect whether there will, with certainty, be at least one fliping move for the passive
        player. Returns True if it detects a move, otherwise returns False and inconclusive.
        """
        set_removed_location = {removed_location}

        # Check flipping actions first:
        for _color, _set_of_coords in self.passive_flipping_groups.items():
            # Additionally remove the orthogonals if colors match:
            if _color == color:
                if len(_set_of_coords - set_removed_location - removed_orthogonals) > 0:
                    return False

            # otherwise, only remove prospective location:
            else:
                # if there is a flipping location free, other than that occupied by the prospective action:
                if len(_set_of_coords - set_removed_location) > 0:
                    return False

        # couldn't find a flipping move:
        return True

    def _full_placing_terminal(self,
                               color: int,
                               removed_location: tuple[int, int],
                               removed_orthogonals: set[tuple[int, int]]) -> bool:
        """ detect whether there will, with certainty be at least one placing move for the passive player.
        Returns True if it detects terminality, otherwise False
        """
        set_removed_location = {removed_location}

        # Definitely no placing moves:
        if self.is_no_more_placing:
            return True

        bad_placing_locations = self.active_orthogonal_groups[self.passive_bowl_token] | set_removed_location

        if color == self.passive_bowl_token:
            bad_placing_locations |= removed_orthogonals

        # determine all locations that the passive player could place their bowl token:
        if self.passive_color_order[self.passive_bowl_token] == 0:
            # if the passive bowl token color group has not been established:
            placing_locations = set(location_to_coordinates) - self.flipped_locations - bad_placing_locations

        else:
            # the orthogonal groups already have the flipped locations removed:
            placing_locations = self.passive_orthogonal_groups[self.passive_bowl_token] - bad_placing_locations

        # if there would be a legal placing move after this :
        if len(placing_locations) > 0:
            return False

        # no placing actions remaining:
        return True

    def _full_is_terminal(self, color: int, removed_location: tuple[int, int]) -> bool:
        """ determine the full action set to decide whether the action could be terminal:
        Returns True if the action will result in a terminal state, False if the game can continue,
        ie if there will be at least one legal move for the passive player after the action is taken.
        """
        # Remove the new flipped location and potentially some orthogonals from possible placing locations:
        removed_orthogonals = orthogonal_neighbors[removed_location]

        # if False, there will be at least one legal flip action for the passive player:
        is_flipping_terminal = self._full_flipping_terminal(color, removed_location, removed_orthogonals)

        if not is_flipping_terminal:
            return False

        # no flipping actions, so check the placing actions rigorously:
        return self._full_placing_terminal(color, removed_location, removed_orthogonals)

    def is_action_terminal(self, action: int) -> bool:
        """ To check whether stepping by a given action from the current state would result in a
        terminal state or not.
        """
        # Initial actions cannot end the game:
        if action >= 104:
            return False

        # determine the coordinates and color of the token affected (in either flipping or placing) after acting:
        removed_location = location_to_coordinates[action % 52]
        color = self.active_bowl_token if action < 52 else self.active_board[removed_location]

        # returns True if you find out the game is not terminal:
        if self._easy_win_placing(color, removed_location):
            return False

        # returns True if the game IS terminal, False if the game is NOT terminal:
        return self._full_is_terminal(color, removed_location)

    def reset_properties(self) -> None:
        """ make sure that all properties are set to None after a move is taken
        """
        self._possible_placing_coordinates = None
        self._placing_actions = None
        self._flipping_actions = None

        # only reset this property after a step if False:
        if not self.is_no_more_placing:
            self._is_no_more_placing = None

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

        self.reset_properties()

        if is_game_terminal:
            # take all remaining flipping actions:
            self.finish_game()

            # Calculate the score:
            active_score, passive_score = self.calculate_score()

            # Result is a 1 if the active player wins, 0 if draw, -1 if the passive player wins:
            result = 0 if active_score == passive_score else 1 if active_score > passive_score else -1

            # store the result
            self.is_terminal = True
            self.result = result

            return True, result

        # swap the player attributes:
        self.swap_players()

        return False, None

    def finish_game(self) -> None:
        """ once the game is in a terminal state, players are allowed to take all possible flipping actions.

        Keep taking actions WITH A SINGLE PLAYER until all actions are used up.

        Remember to reset the properties each time a step is taken.
        """
        # It is possible that the opponent won't have a move, so set _is_no_more_placing
        # at this point:
        self._is_no_more_placing = True

        # reset the properties so that new moves are calculated:
        self.reset_properties()

        # take all legal flipping actions from this point on:
        actions = self.get_actions()

        while actions:
            # take a random flipping action:
            next_action = actions.pop()

            # update the board and attributes as usual:
            location = next_action % 52
            coords = loci[location], locj[location]
            self.update_locations(coords)

            # reset properties and check for new flipping actions:
            self.reset_properties()
            actions = self.get_actions()

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
