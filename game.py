"""
Main game file for patterns.


### Rules of play ###
Players aim to make color groups, which are orthogonally contiguous groups of flipped tokens of the same color

On their turn, players either place their bowl token on the board, or flip an unflipped token.

If a color group has not yet been taken, a token may be placed anywhere on the board except for a space orthogonally
adjacent to a flipped token of the same color that the opponent owns.

If a color group has already been taken, the token must be placed orthogonally adjacent to the existing one (so that
only one color group exists for each color and player)

An unflipped token may only be flipped if it is orthogonally adjacent to a color group of the same color that is
owned by the active player.

Once a player is unable to make a placing move on their turn (whether they have flipping actions or not), no more
placing moves may be taken. Instead, players alternate taking flipping actions until a player cannot move. At that point,
the remaining player takes all remaining flipping actions they would like and the game ends.

Scoring is according to the order in which color groups are taken. Each color group scores #of tokens in group multiplied
by the order it was taken. So 5 oranges, taken first, scores 5 x 1 =  5, whereas 3 blues taken last (6th) scores 18.

At the start of the game, the first player may either choose one of the two colors, in which case the other player
takes the remaining color and plays first, OR they may pass that choice to the opponent, effectively choosing
to play first.

Note that the board is set up with randomly placed color tokens, but the middle 4 squares must contain the colors
2, 3, 4, 5.


### Action space: 107 actions total ###
Actions 0 - 51 represent placing the bowl token that the active player has on the corresponding board location
(64 squares for the 8x8 grid, minus the 4 corners, each of which is 3 spaces)

Actions 52 - 103 represent flipping the token at those locations

Action 104 takes the first color,
Action 105 takes the second color,
Action 106 passes the choice
"""
import random
import numpy as np
from typing import Optional, Self

from int_to_board import orthogonal_neighbors, loci, locj, location_to_coordinates, coordinates_to_location
from int_to_board import set_location_to_coordinates


class Patterns:
    def __init__(self, clone_game: Optional[Self] = None) -> None:
        self.action_space = (107,)

        if clone_game is not None:
            self.clone(clone_game)
            return

        # Values 0-5 are unflipped colors, 6-11 are active player flipped, 12-17 are passive player flipped:
        self.active_board = 18 * np.ones((8, 8), dtype=int)
        self.passive_board = 18 * np.ones((8, 8), dtype=int)

        # The order in which each color was taken for each player. 0 is untaken.
        self.active_color_order = [0] * 6
        self.passive_color_order = [0] * 6

        # Starting game state, for turns 1-3
        self.turn_number = 1
        self.is_game_started = False    # actions 104-106 only taken at the start of the game
        self.first_turn_passed = False  # flag to determine whether the action 106 may be taken
        self.is_terminal = False        # has the game ended
        self.result: Optional[int] = None   # termminal state of the game. 1 is a win for the active player

        # The active player, that is the player about to make a move, can be 1 or -1:
        self.active_player = 1

        # track locations that have been permanently claimed by any player:
        self.flipped_locations = set()

        # Once this flag is True, no more placing actions are allowed for either player:
        self._is_no_more_placing: Optional[bool] = None

        # active and passive bowl tokens can be taken to start as 0 and 1 WLOG:
        self.active_bowl_token = 0
        self.passive_bowl_token = 1

        # color: [(int, int)] showing coordinates orthogonal to color groups, with flipped locations removed:
        self.active_orthogonal_groups = {_col: set() for _col in range(6)}
        self.passive_orthogonal_groups = {_col: set() for _col in range(6)}

        # color: [(int, int) ] coordinates. Note entries exist for all colors with empty lists.
        self.active_color_groups = {_col: [] for _col in range(6)}
        self.passive_color_groups = {_col: [] for _col in range(6)}

        # color: potential flips to rapidly compute legal moves:
        self.active_flipping_groups = {_col: set() for _col in range(6)}
        self.passive_flipping_groups = {_col: set() for _col in range(6)}

        # Track the order value of the next new color-group token to be placed:
        self.active_placing_number = 1
        self.passive_placing_number = 1

        # populate a random initial state for the board:
        self.initialize_boards()

        # rather than recalculating, store the calculated placing actions:
        self._placing_actions: Optional[list[int]] = None

        # distinguish between the actual placing actions, which considers is_no_more_placing, and this, which is
        # used in the definition of is_no_more_placing
        self._possible_placing_coordinates: Optional[set[tuple[int, int]]] = None
        self._flipping_actions: Optional[list[int]] = None

    def clone(self, patterns_game: Self) -> None:
        """ Efficiently create a new patterns game instance from an existing game:
        """
        # Copy numpy arrays:
        self.active_board = patterns_game.active_board.copy()
        self.passive_board = patterns_game.passive_board.copy()

        # Slice copy lists:
        self.active_color_order = patterns_game.active_color_order[:]
        self.passive_color_order = patterns_game.passive_color_order[:]
        self._placing_actions = patterns_game._placing_actions[:] if patterns_game._placing_actions is not None else None
        self._flipping_actions = patterns_game._flipping_actions[:] if patterns_game._flipping_actions is not None else None

        # single values: no need to copy:
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
        self._possible_placing_coordinates = (set(patterns_game._possible_placing_coordinates) if
                                              patterns_game._possible_placing_coordinates is not None else None)

        # dictionary copy: must deep copy by explicitly setting lists within
        self.active_color_groups = {_col: _val[:] for _col, _val in patterns_game.active_color_groups.items()}
        self.passive_color_groups = {_col: _val[:] for _col, _val in patterns_game.passive_color_groups.items()}

        self.active_orthogonal_groups = {_col: set(_val) for _col, _val in patterns_game.active_orthogonal_groups.items()}
        self.passive_orthogonal_groups = {_col: set(_val) for _col, _val in patterns_game.passive_orthogonal_groups.items()}

        self.active_flipping_groups = {_col: set(_val) for _col, _val in patterns_game.active_flipping_groups.items()}
        self.passive_flipping_groups = {_col: set(_val) for _col, _val in patterns_game.passive_flipping_groups.items()}

    def initialize_boards(self):
        """ Random initial board comprising 9 tiles for each of 6 different colors. The center square must contain
        the colors 2, 3, 4, 5 (0, 1 are initial bowl tokens):
        """
        # Non-center board:
        tiles = list(range(6)) * 8
        random.shuffle(tiles)

        # Center square, with colors 2, 3, 4, 5 WLOG:
        middle = [2, 3, 4, 5]
        random.shuffle(middle)
        initial_state = tiles[:21] + middle[:2] + tiles[21:27] + middle[2:] + tiles[27:]

        # Active and passive boards are initially identical as no flipping has taken place:
        self.active_board[loci, locj] = initial_state
        self.passive_board[loci, locj] = initial_state

    @property
    def is_no_more_placing(self) -> bool:
        """ We have is_no_more_placing as a property, so that if it is ever queried and the placing actions haven't
        been checked, we check:

        Note: for search tree, children adopt the parent is no more placing flag, unless it is false, in which case
        set it to None.
        """
        # if set to None, it means not checked yet. DO NOT use this setter if any parent game had it set to True:
        if self._is_no_more_placing is None:
            self.set_is_no_more_placing()

        return self._is_no_more_placing

    def set_is_no_more_placing(self) -> None:
        """ determine whether there are any legal placing moves if the property has not been set.
        """
        # evaluates to true if there are no possible placing coordinates, otherwise false:
        self._is_no_more_placing = not self.possible_placing_coordinates


    @property
    def possible_placing_coordinates(self) -> set[tuple[int, int]]:
        """ Determine the locations that are empty to be placed in, without regard for the
        is no more placing flag
        """
        if self._possible_placing_coordinates is None:
            self.set_possible_placing_coordinates()

        return self._possible_placing_coordinates

    def set_possible_placing_coordinates(self) -> None:
        """ determine all the locations that are empty to be placed in, ignoring the is_no_more_placing flag
        """
        # New color group can be started anywhere that is not orthogonal to an opponents matching color group:
        token = self.active_bowl_token
        color_order = self.active_color_order[token]
        opposing_orthogonals = self.passive_orthogonal_groups[token]

        if color_order == 0:
            empty_spaces = set_location_to_coordinates - self.flipped_locations - opposing_orthogonals

        else:
            empty_spaces = self.active_orthogonal_groups[token] - opposing_orthogonals

        self._possible_placing_coordinates = empty_spaces

    @property
    def placing_actions(self) -> list[int]:
        """ Set the actions for the placing actions
        """
        if self._placing_actions is None:
            self.set_placing_actions()

        return self._placing_actions

    def set_placing_actions(self) -> None:
        """ actions that involve placing the bowl token. Integer actions, not coordinates
        """
        if self.is_no_more_placing:
            self._placing_actions = []
            return

        # turn the coordinates into locations for the action space:
        self._placing_actions = [coordinates_to_location[_coord] for _coord in self.possible_placing_coordinates]

    @property
    def flipping_actions(self) -> list[int]:
        """ Actions that involve flipping an unflipped token on the board that is located orthogonally
        adjacent to a previously taken color group of the active player. A token that is also orthogonally adjacent
        to an opponents color group of the same color cannot be flipped.

        As an additional optimization, we do not consider flipping actions that would match an equivalent
        placing action
        """
        if self._flipping_actions is None:
            self.set_flipping_actions()

        return self._flipping_actions

    def set_flipping_actions(self) -> None:
        """ Integer actions not coordinates.
        """
        self._flipping_actions = [coordinates_to_location[_coord] + 52 for
                                  _key, _val in self.active_flipping_groups.items() for _coord in _val
                                  if not ((coordinates_to_location[_coord] in self.placing_actions)
                                          and (_key == self.active_bowl_token))]

    def calculate_score(self) -> tuple[int, int]:
        """ The score is the number of elements in a color group multiplied by the corresponding
        color order of the group.

        Return the score for (active player, passive player) in the current state.
        """
        active_score = 0
        passive_score = 0

        for _color, (aorder, porder) in enumerate(zip(self.active_color_order, self.passive_color_order)):
            active_score += len(self.active_color_groups[_color]) * aorder
            passive_score += len(self.passive_color_groups[_color]) * porder

        return active_score, passive_score

    def get_actions(self) -> list[int]:
        """ Return the legal actions, according to the action space definition, as a list of ints.

        At the start of the game, a player can choose to take 0 as bowl token,
        swap for 1, or pass that choice to the other player.

        If the choice was swapped, the second player can make the decision to take 0 or 1.

        After the game begins, a player may either:
        1. Place a bowl token on the board
        2. Flip over a piece next to one of your own

        Once placing has stopped, only flipping actions are allowed.
        """
        # Either choose a color or pass the choice:
        if not self.is_game_started:
            return [104, 105, 106]

        # if a hand piece has not been chosen yet, the first player passed the choice:
        if self.first_turn_passed:
            return [104, 105]

        return self.placing_actions + self.flipping_actions

    def is_action_terminal(self, action: int) -> bool:
        """ Determine whether taking the action indicated will result in a terminal state or not.
        Return False if the action is not terminal
        Return True if the action is terminal and will end the game:
        """
        # Initial actions cannot end the game:
        if action >= 104:
            return False

        # All checks will require the coordinates targeted and the color of the token:
        target_location = location_to_coordinates[action % 52]
        color = self.active_bowl_token if action < 52 else self.active_board[target_location]

        # returns True if you find out the game is not terminal:
        if self._easy_win_placing(color, target_location):
            return False

        # Conclusive check: returns True if the game is terminal, False if the game is not terminal:
        return self._full_is_terminal(color, target_location)

    def _easy_win_placing(self, color: int, target_location: tuple[int, int]) -> bool:
        """
        In many situations, it is easy to determine that there will be at least one remaining move
        available after an action is taken. It is efficient to run through a number of checks first before
        turning to an exhaustive check of the next action.

        Returning True tells you that you have an easy win and the game will definitely not end.
        No need to check further.

        Returning False indicates INCONCLUSIVE!! Further work is required to determine an exact answer.

        Color is an int determining the color that will exist in the
        """
        # if placing moves are no longer allowed, no easy win as flipping actions must be checked:
        if self.is_no_more_placing:
            return False

        if self._easy_win_passive_token_not_target_color(color, target_location):
            return True

        if self._easy_win_passive_token_is_target_color(color):
            return True

        return False

    def _easy_win_passive_token_not_target_color(self, color: int, target_location: tuple[int, int]) -> bool:
        """
        in the situation where the color of the target location is different from the passive
        bowl token, check whether there are obvious signs that the game will continue after the action taken
        """
        # this function only checks for situations where the passive bowl token is distinct from the target
        # location color:
        if self.passive_bowl_token == color:
            return False

        if self._easy_win_new_passive():
            return True

        if self._easy_win_existing_passive(target_location):
            return True

        return False

    def _easy_win_new_passive(self) -> bool:
        """
        To arrive here:
        1. passive bowl token is NOT the color indicated,
        2. the passive player has NOt started the color group associated with their bowl token
        """
        # Return if the passive bowl token group has been taken:
        if self.passive_color_order[self.passive_bowl_token] != 0:
            return False

        # Neither player has this color group, so the game cannot end - must be remaining tokens:
        if self.active_color_order[self.passive_bowl_token] == 0:
            return True

        # Number of passive bowl token color flipped by active player:
        num_current_active = len(self.active_color_groups[self.passive_bowl_token])

        ### new color groups cannot be started on flipped locations, nor orthogonal to the existing color group.
        # the number flipped after the action is number flipped + 1
        # the number maximally blocked out by the existing active color group is 2 * num + 2
        # therefore if 52 - (num flipped + 1) - (2 * num + 2) >0, there MUST be at least one legal placing location:
        if (52 - len(self.flipped_locations) - 2 * num_current_active - 3) > 0:
            return True

        return False

    def _easy_win_existing_passive(self, target_location: tuple[int, int]) -> bool:
        """
        To arrive here:
        1. passive bowl token is not the color indicated,
        2. the passive player has an existing color group associated with their bowl token
        """
        # Return if the passive bowl token group has not been taken:
        if self.passive_color_order[self.passive_bowl_token] == 0:
            return False

        # Active has not taken this group yet:
        if self.active_color_order[self.passive_bowl_token] == 0:
            # as long as there is one location remaining after the target location is removed, placing will be legal:
            if len(self.passive_orthogonal_groups[self.passive_bowl_token] - {target_location}) > 0:
                return True

            return False

        # Active has taken this group already:
        if len(self.passive_orthogonal_groups[self.passive_bowl_token]
               - self.active_orthogonal_groups[self.passive_bowl_token] - {target_location}) > 0:
            # if there are at least 1 empty spots after removing the orthogonals and the removed location:
            return True

        return False

    def _easy_win_passive_token_is_target_color(self, color: int) -> bool:
        """
        check for easy placing wins when passive bowl token matches the target location color:
        """
        # the token being flipped must match the passive players bowl token:
        if self.passive_bowl_token != color:
            return False

        # If the passive player has not taken this color group yet:
        if self.passive_color_order[color] == 0:
            # If the active player has not taken this color group yet:
            if self.active_color_order[color] == 0:
                # There must be many locations to place in:
                return True

            # If the active player has an existing color group of this color:
            num_current_active = len(self.active_color_groups[color])

            ### The number of locations that can be blocked by the active color group is:
            # 52 - (flipped locations + 1) - (2 * (num + 1) + 2)
            # = 52 - flipped locations - 2 * num - 5
            if (52 - len(self.flipped_locations) - 2 * num_current_active - 5) > 0:
                return True

        return False

    def _full_is_terminal(self, color: int, removed_location: tuple[int, int]) -> bool:
        """ determine the full action set to decide whether the action could be terminal:
        Returns True if the action will result in a terminal state, False if the game can continue,
        ie if there will be at least one legal move for the passive player after the action is taken.
        """
        # Remove the new flipped location and potentially some orthogonals from possible placing locations:
        removed_orthogonals = orthogonal_neighbors[removed_location]

        # If False, there will be at least one legal flip action for the passive player:
        is_flipping_move = self._full_flipping_terminal(color, removed_location, removed_orthogonals)

        # if there is a flipping move, we know the game cannot be terminal:
        if is_flipping_move:
            return False

        is_placing_move = self._full_placing_terminal(color, removed_location, removed_orthogonals)

        return not is_placing_move

    def _full_flipping_terminal(self,
                                color: int,
                                removed_location: tuple[int, int],
                                removed_orthogonals: set[tuple[int, int]]) -> bool:
        """ Return True if there is a legal flipping move, return False if there are no legal flipping moves
        """
        set_removed_location = {removed_location}

        # iterate over each flipping group:
        for _color, _set_of_coords in self.passive_flipping_groups.items():
            # If the colors match, also remove orthogonals:
            if _color == color:
                if len(_set_of_coords - set_removed_location - removed_orthogonals) > 0:
                    return True

            # otherwise, only remove prospective location:
            else:
                # if there is a flipping location free, other than that occupied by the prospective action:
                if len(_set_of_coords - set_removed_location) > 0:
                    return True

        # If there are no legal flipping moves, return False:
        return False

    def _full_placing_terminal(self,
                               color: int,
                               removed_location: tuple[int, int],
                               removed_orthogonals: set[tuple[int, int]]) -> bool:
        """ Return True if there is a legal placing move
        Return False if there are no legal placing moves.
        """
        set_removed_location = {removed_location}

        # Definitely no placing moves:
        if self.is_no_more_placing:
            return False

        # cannot place orthogonal to the opponents color group, nor the location just flipped:
        bad_placing_locations = self.active_orthogonal_groups[self.passive_bowl_token] | set_removed_location

        # additionally, cannot place adjacent to the removed location if this matches the color:
        if color == self.passive_bowl_token:
            bad_placing_locations |= removed_orthogonals

        # determine all locations that the passive player could place their bowl token:
        if self.passive_color_order[self.passive_bowl_token] == 0:
            # if the passive bowl token color group has not been established:
            placing_locations = set(location_to_coordinates) - self.flipped_locations - bad_placing_locations

        else:
            # the orthogonal groups already have the flipped locations removed:
            placing_locations = self.passive_orthogonal_groups[self.passive_bowl_token] - bad_placing_locations

        # if there is at least one placing location, return True.
        return len(placing_locations) > 0

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
        # Before any further logic, determine whether the action selected will end the game:
        is_game_terminal = self.is_action_terminal(action)

        # Simply pass choice of initial bowl token to the other player:
        if action == 106:
            self._action_106()

        # Choose which tile you want to play with:
        if action in [104, 105]:
            self._action_choose(action)

        # actions that represent a change to the board state:
        if action < 104:
            self._action_main(action)

        # always reset the properties after taking an action:
        self.reset_properties()

        # if the game is terminal, run the end game routine:
        if is_game_terminal:
            # take all remaining flipping actions:
            return self.finish_game()

        # swap the player attributes:
        self.swap_players()

        return False, None

    def _action_106(self) -> None:
        """ logic for passing the choice of color to the opponent:
        """
        # flag to indicate it is no longer the first turn:
        self.is_game_started = True

        # set flag to indicate turn passed:
        self.first_turn_passed = True

    def _action_choose(self, action: int) -> None:
        """ choose either color 0 or color 1
        """
        # flag to indicate it is no longer the first turn:
        self.is_game_started = True

        # remove potential flag for token choice being passed:
        self.first_turn_passed = False

        # lets be explicit:
        if action == 104:
            self.active_bowl_token = 0
            self.passive_bowl_token = 1

        if action == 105:
            self.active_bowl_token = 1
            self.passive_bowl_token = 0

    def _action_main(self, action: int) -> None:
        """
        Take a flipping or placing action
        """
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

    def update_locations(self, coords: tuple[int, int]) -> None:
        """
        - update the board to account for the newly flipped location
        - remove the newly flipped location from the flipping and orthogonal groups
        - update the color groups and color order with the new location
        - update the orthogonal and flipping groups
        """
        # Determine the color of the token that was flipped or placed:
        token_color = int(self.active_board[coords])

        self.update_board(coords)
        self.remove_new_location(coords)
        self.update_color_groups(coords, token_color)
        self.update_orthogonal_groups(coords, token_color)
        self.update_flipping_groups(token_color)

    def update_board(self, coords: tuple[int, int]) -> None:
        """
        Increment the flipping location. The new color has already been placed, so simply increment by
        6 for active, 12 for passive.

        Add the coordinates to the flipping locations set.
        """
        # Flip the color that has been placed:
        self.active_board[coords] += 6
        self.passive_board[coords] += 12

        # Add the new token to the set of flipped locations:
        self.flipped_locations.add(coords)

    def remove_new_location(self, coords: tuple[int, int]) -> None:
        """ Iterate over all colors and all groups to remove the coordinates of the flipped location.
        """
        coords_set = {coords}

        for _col in range(6):
            self.active_flipping_groups[_col] -= coords_set
            self.active_orthogonal_groups[_col] -= coords_set
            self.passive_flipping_groups[_col] -= coords_set
            self.passive_orthogonal_groups[_col] -= coords_set

    def update_color_groups(self, coords: tuple[int, int], color: int) -> None:
        """ If the color group is new, update active color order. Append the coordinates to the color group.
        """
        if self.active_color_order[color] == 0:
            self.active_color_order[color] = self.active_placing_number
            self.active_placing_number += 1

        # Add the newly placed token to the relevant color group:
        self.active_color_groups[color].append(coords)

    def update_orthogonal_groups(self, coords: tuple[int, int], color: int) -> None:
        """ update the orthogonal empties for the taken color
        """
        # add empty token orthogonals to the active orthogonal color group:
        token_orthogonals = orthogonal_neighbors[coords] - self.flipped_locations
        self.active_orthogonal_groups[color] |= token_orthogonals

    def update_flipping_groups(self, color: int) -> None:
        """ create new flipping groups for the new color
        """
        active_orthogonal_set = self.active_orthogonal_groups[color]
        passive_orthogonal_set = self.passive_orthogonal_groups[color]

        # iterate over the flipping groups for the given color for both passive and active players:
        self.active_flipping_groups[color] = {_x for _x in active_orthogonal_set if self.active_board[_x] == color
                                              if _x not in passive_orthogonal_set}

        self.passive_flipping_groups[color] = {_x for _x in passive_orthogonal_set if self.active_board[_x] == color
                                               if _x not in active_orthogonal_set}

    def finish_game(self) -> tuple[bool, int]:
        """ Take all flipping actions with the remaining player, return the final score and result flags.
        """
        # Only flipping actions for a terminal game:
        self._is_no_more_placing = True

        # Reset the properties so that new moves are still calculated:
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

        # Calculate the score:
        active_score, passive_score = self.calculate_score()

        # Result is a 1 if the active player wins, 0 if draw, -1 if the passive player wins:
        result = 0 if active_score == passive_score else 1 if active_score > passive_score else -1

        # store the result
        self.is_terminal = True
        self.result = result

        return True, result
