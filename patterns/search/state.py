""" We separate out the representation of the tensor state for the patterns game

This is not the responsibility of the game itself, as it is explicitly for NNs,
and it does not belong in the search tree as it is used far more broadly.

Tensor states can be created from parent states and an action, from parent games and an action
or from a game itself.

This class also includes functionality to create a Patterns game instance from a tensor state.

Throughout, all tensors are numpy, as it is far quicker to update and manipulate numpy arrays
and do a single batch transfer to torch during inference.

"""

import numpy as np

from patterns.game import Patterns
from patterns.constants import loci, locj
from patterns.constants import EYE
from patterns.constants import SWAP_ACTIVE_PASSIVE_INDEX
from patterns.constants import set_location_to_coordinates
from patterns.constants import coordinates_to_orthogonal_neighbors


def create_game_from_state_array(state: np.ndarray) -> Patterns:
    """ [board_tensor, order_tensor, bowl_tensor, placing_tensor, score_tensor, token_value_tensor]
    =
    [18, 12, 12, 1, 2, 2] = 47 slices
    """
    # will create a copy:
    numpy_state = np.array(state)

    # extract the layers by purpose:
    board_layers = numpy_state[:18, :, :]

    # multiply by 6 to get int values 0 - 6:
    p1_order_layers = (numpy_state[18:24, 0, 0] * 6).astype(int)
    p2_order_layers = (numpy_state[24:30, 0, 0] * 6).astype(int)

    p1_bowl_tokens = numpy_state[30:36, 0, 0]
    p2_bowl_tokens = numpy_state[36:42, 0, 0]

    # create game to assign attributes to:
    new_game = Patterns()

    # note that false would mean checked and definitely not true: fine to use None here
    new_game._is_no_more_placing = True if numpy_state[42, 0, 0] > 0 else None

    # assume that if we are validating, or wishing to create game from state, the game has started proper:
    # todo reconsider the interaction of this with the earliest moves, when we are predicting first
    #  moves from the board state...
    new_game.is_game_started = True
    new_game.first_turn_passed = False

    assign_bowl_tokens(new_game, p1_bowl_tokens, p2_bowl_tokens)
    populate_board(new_game, board_layers)
    assign_color_orders(new_game, p1_order_layers, p2_order_layers)
    assign_orth_flip_groups(new_game, )

    return new_game


def assign_bowl_tokens(new_game: Patterns,
                       p1_bowl_tokens: np.ndarray,
                       p2_bowl_tokens: np.ndarray) -> None:
    """ use the tensors to assign active and passive bowl tokens to the reconstructed game
    """
    p1_color = np.where(p1_bowl_tokens)[0].item()
    p2_color = np.where(p2_bowl_tokens)[0].item()

    new_game.active_bowl_token = p1_color
    new_game.passive_bowl_token = p2_color


def populate_board(new_game: Patterns,
                   board_tensor: np.ndarray) -> None:
    """ use the board tensor (18, 8, 8) to populate the active and passive boards in the game
    """
    # both active and passive boards are required for the game:
    new_game.active_board = np.zeros((8, 8), dtype=int)
    new_game.passive_board = np.zeros((8, 8), dtype=int)

    # color and user, i, j location on 2d board:
    vals, iind, jind = np.where(board_tensor)
    coordinate_list = np.array(list(zip(iind, jind)))

    # we can assign color groups with this:
    new_game.active_color_groups = {_col: [] for _col in range(6)}
    new_game.passive_color_groups = {_col: [] for _col in range(6)}

    for _color in range(6):
        new_game.active_color_groups[_color] = list(map(tuple, coordinate_list[vals == (_color + 6)]))
        new_game.passive_color_groups[_color] = list(map(tuple, coordinate_list[vals == (_color + 12)]))

    # masks for the active and passive tokens:
    active_mask = np.logical_and(vals >= 6, vals < 12)
    passive_mask = vals >= 12

    new_game.active_board[iind, jind] = vals
    new_game.passive_board[iind, jind] = vals

    # correct for the passive/ active representations - from the passive players pov, the active is passive!
    new_game.passive_board[iind[active_mask], jind[active_mask]] += 6
    new_game.passive_board[iind[passive_mask], jind[passive_mask]] -= 6

    # the game also wants to track all unflipped locations
    unflipped_locations = set(map(tuple, coordinate_list[vals < 6]))

    # subtract the unflipped locations from all the possible locations:
    new_game.flipped_locations = set_location_to_coordinates - unflipped_locations


def assign_color_orders(new_game: Patterns,
                        p1_order_layers: np.ndarray,
                        p2_order_layers: np.ndarray) -> None:
    """ 0 means untaken, and 1-6 are the orders in which the colors are taken by each player:
    """
    new_game.active_color_order = p1_order_layers
    new_game.passive_color_order = p2_order_layers
    new_game.active_placing_number = max(p1_order_layers) + 1
    new_game.passive_placing_number = max(p2_order_layers) + 1


def assign_orth_flip_groups(new_game: Patterns):
    """ The game requires attributes specifing orthogonal unoccupied locations to continue forward.
    """
    # populate the orthogonal groups and flipping groups from the color groups:
    for _color in range(6):
        active_group = new_game.active_color_groups[_color]
        passive_group = new_game.passive_color_groups[_color]

        active_orthogs = []
        passive_orthogs = []

        # find all the orthogonal locations then remove any that are flipped:
        for _coords in active_group:
            active_orthogs.extend(list(coordinates_to_orthogonal_neighbors[_coords]))

        for _coords in passive_group:
            passive_orthogs.extend(list(coordinates_to_orthogonal_neighbors[_coords]))

        new_game.active_orthogonal_groups[_color] = set(active_orthogs) - new_game.flipped_locations
        new_game.passive_orthogonal_groups[_color] = set(passive_orthogs) - new_game.flipped_locations

        new_game.active_flipping_groups[_color] = set([_x for _x in new_game.active_orthogonal_groups[_color]
                                                       if new_game.active_board[_x] == _color
                                                       if _x not in new_game.passive_orthogonal_groups[_color]])

        new_game.passive_flipping_groups[_color] = set([_x for _x in new_game.passive_orthogonal_groups[_color]
                                                        if new_game.active_board[_x] == _color
                                                        if _x not in new_game.active_orthogonal_groups[_color]])


def create_state_from_parent_state(parent_state: np.ndarray,
                                   parent_game: Patterns,
                                   parent_action: int) -> np.ndarray:
    """ Create tensor state through the parent tensor state, game and selected action.

    Swap all active and passive layers, then correct the entries for passive.
    (board, score, bowl token, color order, bowl token value, is no placing)

    Note: work explicitly with numpy arrays as elementwise manipulation of torch tensors is very slow

    """
    # Permute parent tensor to swap active and passive slices:
    state = np.array(parent_state)[SWAP_ACTIVE_PASSIVE_INDEX]

    location = parent_action % 52
    coords = loci[location], locj[location]
    board_color = parent_game.active_board[coords]

    # Update each of the types of tensor slice in turn:
    placing_flag = _determine_placing_flag_from_parent(parent_game, parent_action, coords)
    state[42] = float(placing_flag)

    # update the state in place for the rest:
    _state_update_board(state, parent_game, parent_action, coords, board_color)
    _state_update_bowl_tokens(state, parent_game, parent_action, board_color)
    _state_update_color_order(state, parent_game, parent_action)
    _state_update_score(state, parent_game, parent_action, board_color)
    _state_update_bowl_token_values(state, parent_game, parent_action, board_color)

    return state


def _determine_placing_flag_from_parent(parent_game: Patterns,
                                        parent_action: int,
                                        coords: tuple[int, int]) -> bool:
    """ Use the parent game and the parent action to correctly determine whether the flag could have changed.
    Alter the state in place.
    """
    # if start of game, no action:
    if parent_action >= 104:
        return False

    # If is_no_more_placing true for a parent, no children can ever be false:
    if parent_game.is_no_more_placing:
        return True

    # token color that will need to have a legal placing location:
    token_color = parent_game.passive_bowl_token
    color_order = parent_game.passive_color_order[token_color]
    opposing_orthogonals = parent_game.active_orthogonal_groups[token_color]

    if color_order == 0:
        empty_spaces = (set_location_to_coordinates - parent_game.flipped_locations - opposing_orthogonals)

    else:
        empty_spaces = parent_game.passive_orthogonal_groups[token_color] - opposing_orthogonals

    # Also consider the piece just flipped, which could affect the orthogonals:

    # Color is either the active bowl token if it was placed, or the board color if it was flipped:
    action_color = parent_game.active_bowl_token if parent_action < 52 else parent_game.active_board[coords]
    action_block = {coords}

    # if the flipped/ placed board token matches the color of the passive bowl token, additional orthogonals
    # could be blocked from play:
    if action_color == token_color:
        action_block |= coordinates_to_orthogonal_neighbors[coords]

    # remove the flipped location and any relevant orthogonal neighbors from the empty spaces that can be placed in:
    empty_spaces -= action_block

    # False if there are empty spaces left, else True
    return not len(empty_spaces) > 0

def _state_update_board(state: np.ndarray,
                        parent_game: Patterns,
                        parent_action: int,
                        coords: tuple[int, int],
                        board_color: int) -> None:
    """ correct the child state board based on the parent action:
    """
    if parent_action >= 104:
        return

    # remove unflipped:
    state[board_color, coords[0], coords[1]] = 0.0

    # determine new (flipped) color of the coordinates:
    new_color = parent_game.active_bowl_token if parent_action < 52 else board_color

    # +12 represents flipped for the passive player:
    state[12 + new_color, coords[0], coords[1]] = 1.0


def _state_update_bowl_tokens(state: np.ndarray,
                              parent_game: Patterns,
                              parent_action: int,
                              board_color: int,) -> None:
    """ Change bowl tokens.
    Note that 104 always means the active player takes 0 and 105 always means that the active player takes 1
    """
    # if the player chooses either 0 or 1 as their bowl token:
    if parent_action in [104, 105]:
        inds = [30, 31, 36, 37]
        # recall that this is from passive pov:
        vals = [0.0, 1.0, 1.0, 0.0] if parent_action == 104 else [1.0, 0.0, 0.0, 1.0]

        for _ind, _val in zip(inds, vals):
            state[_ind] = _val

    # If this was a placing action:
    if parent_action < 52:
        # update the passive bowl token entries if it changes:
        if board_color != parent_game.active_bowl_token:
            state[36 + parent_game.active_bowl_token] = 0.0  # zero the old color
            state[36 + board_color] = 1.0  # take the board color as new token


def _state_update_color_order(state: np.ndarray,
                              parent_game: Patterns,
                              parent_action: int) -> None:
    """ if the action leads to a new color having been placed, this must be addressed:
    """
    # only affects placing actions:
    if parent_action >= 52:
        return

    ### if the active bowl token has not been taken yet, but a placing action was taken:
    if parent_game.active_color_order[parent_game.active_bowl_token] == 0:
        ### update color order:
        state[24 + parent_game.active_bowl_token] = parent_game.active_placing_number / 6.0


def _state_update_score(state: np.ndarray,
                        parent_game: Patterns,
                        parent_action: int,
                        board_color: int) -> None:
    """ update the score slice by accounting for the just flipped or placed piece
    """
    if parent_action >= 104:
        # no change to score:
        return

    # if a placing action was taken, the active bowl token dictates the increment in value:
    if parent_action < 52:
        # the value if the color group already exists:
        score_value = parent_game.active_color_order[parent_game.active_bowl_token]

        # if the active bowl token wasn't in active color groups, both score and color order are updated:
        if score_value == 0:
            score_value = parent_game.active_placing_number

    else:
        # update the score:
        score_value = parent_game.active_color_order[board_color]

    # increment the score slice by the score value:
    state[44] += score_value / 150.0


def _state_update_bowl_token_values(state: np.ndarray,
                                    parent_game: Patterns,
                                    parent_action: int,
                                    board_color: int) -> None:
    """ if a new bowl token was taken, update the value of it
    """
    # flipping actions do not change bowl tokens:
    if parent_action >= 52:
        return

    # if the board color that has been taken as passive token is not in passive color groups:
    if parent_game.active_color_order[board_color] == 0:
        # the value is the current next to play:
        bowl_token_value = parent_game.active_placing_number

        # However, if the active piece played by the action was also a new color group, increment again:
        if ((parent_game.active_color_order[parent_game.active_bowl_token] == 0)
                and (parent_game.active_bowl_token != board_color)):
            bowl_token_value += 1

    # otherwise, it is just equal to the existing color order:
    else:
        bowl_token_value = parent_game.active_color_order[board_color]

    state[46] = bowl_token_value / 6.0


def create_state_from_parent_game(parent_game: Patterns,
                                   parent_action: int) -> np.ndarray:
    """ use the parent game to create a tensor state for the chosen child
    """
    game_attributes = get_attributes_from_parent_game(game=parent_game, action=parent_action)
    state_tensor = create_state_from_game_attributes(*game_attributes)
    return state_tensor


def create_state_from_game(game: Patterns) -> np.ndarray:
    """ use the current game to create a tensor state for the chosen child
    """
    game_attributes = get_attributes_from_game(game=game)
    state_tensor = create_state_from_game_attributes(*game_attributes)
    return state_tensor


def get_attributes_from_game(game: Patterns) -> tuple:
    """ return the necessary attributes to form a state tensor, from a given game:
    """
    return (game.active_board,
            game.active_color_order, game.passive_color_order,
            game.active_color_groups, game.passive_color_groups,
            game.active_bowl_token, game.passive_bowl_token,
            game.is_no_more_placing)


### This should largely not be necessary, as the search tree node will have either a game itself, or a parent which
### possess a tensor state:
def get_attributes_from_parent_game(game: Patterns, action: int) -> tuple:
    """ If neither game nor parent tensor are present, alter the parent attributes to represent the current node game
    after the provided action is taken:
    """
    # if start of game:
    if action >= 104:
        active_token = (action + 1) % 2
        passive_token = action % 2

        return (game.passive_board,
                [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0],
                {_col: [] for _col in range(6)}, {_col: [] for _col in range(6)},
                active_token, passive_token,
                False,
               )

    # change to board state:
    location = action % 52
    coords = loci[location], locj[location]

    # Copy the passive board:
    board = np.array(game.passive_board)

    # active <--> passive: copying necessary
    active_token = game.passive_bowl_token
    passive_token = game.active_bowl_token

    active_order = game.passive_color_order[:]
    passive_order = game.active_color_order[:]

    active_color_groups = {_key: _val[:] for _key, _val in game.passive_color_groups.items()}
    passive_color_groups = {_key: _val[:] for _key, _val in game.active_color_groups.items()}

    # update the relevant color group with the coordinates of the flipped location:
    # note: passive token here is the active bowl token
    update_color = passive_token if action < 52 else board[coords]
    passive_color_groups[update_color].append(coords)

    # the passive token becomes the color at the location, the new location gets the COLOR of the passive token:
    if action < 52:
        new_token = board[coords]
        board[coords] = passive_token
        passive_token = new_token

        # update the passive order if a new color group will be formed by the placing action;
        if game.active_color_order[game.active_bowl_token] == 0:
            passive_order[game.active_bowl_token] = game.active_placing_number

    # Whether flipping or placing, the location takes on the +12 of the passive player:
    board[coords] += 12

    is_no_more_placing_flag = _determine_placing_flag_from_parent(game, action, coords)

    return (board,
            active_order, passive_order,
            active_color_groups, passive_color_groups,
            active_token, passive_token,
            is_no_more_placing_flag,
            )


def create_state_from_game_attributes(board: np.ndarray,
                                      active_color_order: list, passive_color_order: list,
                                      active_color_groups: dict, passive_color_groups: dict,
                                      active_token: int, passive_token: int,
                                      placing_flag: bool,
                                      ) -> np.ndarray:
    """ Create the state directly from patterns game attributes:
    """
    board_array = _array_board(board)
    color_order_array = _array_color_order(active_color_order, passive_color_order)
    bowl_token_array = _array_bowl_tokens(active_token, passive_token)
    placing_flag_array = _array_placing_flag(placing_flag)
    score_array = _array_score(active_color_order, passive_color_order, active_color_groups, passive_color_groups)
    bowl_value_array = _array_bowl_value(active_color_order, passive_color_order, active_token, passive_token)

    return np.concatenate([board_array, color_order_array, bowl_token_array, placing_flag_array,
                           score_array, bowl_value_array], axis=0)


def _array_board(board: np.ndarray) -> np.ndarray:
    """ accept a game board (8x8) and return a (18, 8, 8) tensor indicating
    where colors are present
    """
    board_array = EYE[:, board.flatten()].reshape(18, 8, 8)
    return board_array


def _array_color_order(active_color_order: list, passive_color_order: list) -> np.ndarray:
    """ return a (12, 8, 8) with constant slices tensor representing the state of the two color order lists
    """
    order_list = [_order / 6.0 for _order in active_color_order + passive_color_order]
    order_array = np.tile(np.array(order_list).reshape(12, 1, 1), (1, 8, 8))
    return order_array


def _array_bowl_tokens(active_token: int, passive_token: int) -> np.ndarray:
    """  return a (12, 8, 8) with constant slices tensor representing the state of the two bowl token
    colors.
    """
    bowl_list = [0.0] * 12
    bowl_list[active_token] = 1.0
    bowl_list[passive_token + 6] = 1.0
    bowl_array = np.tile(np.array(bowl_list).reshape(12, 1, 1), (1, 8, 8))
    return bowl_array


def _array_placing_flag(placing_flag: bool) -> np.array:
    """ simply make a tensor of the right shape to fit the flag
    """
    return np.tile(np.array([float(placing_flag)]).reshape(1, 1, 1), (1, 8, 8))


def _array_score(active_color_order: list, passive_color_order: list,
                 active_color_groups: dict, passive_color_groups: dict) -> np.ndarray:
    """ use the number of tokens in each color groups and the value from the color orders
    to determine a current score
    """
    active_score = 0
    passive_score = 0

    for _col, (_active_order, _passive_order) in enumerate(zip(active_color_order, passive_color_order)):
        active_score += len(active_color_groups[_col]) * _active_order
        passive_score += len(passive_color_groups[_col]) * _passive_order

    score_array = np.tile(np.array([active_score / 150., passive_score / 150.]).reshape(2, 1, 1), (1, 8, 8))
    return score_array


def _array_bowl_value(active_color_order: list, passive_color_order: list,
                      active_token: int, passive_token: int) -> np.array:
    """ represent the value of the tokens currently held by each player
    """
    ### Bowl token value tensor:
    active_value = active_color_order[active_token]
    passive_value = passive_color_order[passive_token]

    # If the token is not yet taken, the value is one greater than the current maximum:
    if active_value == 0:
        active_value = max(active_color_order) + 1

    if passive_value == 0:
        passive_value = max(passive_color_order) + 1

    token_value_array = np.tile(np.array([active_value / 6.0, passive_value / 6.0]).reshape(2, 1, 1), (1, 8, 8))
    return token_value_array
