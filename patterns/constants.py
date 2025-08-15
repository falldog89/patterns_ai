""" Convenience and optimization constants that are used throughout the patterns package.
- efficiently pass between coordinates and integer locations for the trimmed 8x8 grid
- access precalculated orthogonal neighbors
- precalculated indices to speed up augmentation for symmetries
- indices that allow states to be efficiently moved from passive to active players
"""

import numpy as np


### Coordinates for an 8x8 board, with the corners removed:
# Note that coordinates are given in matrix notation, so (i, j) not (x, y):

loci = np.array([
    0, 0, 0, 0,
    1, 1, 1, 1, 1, 1,
    2, 2, 2, 2, 2, 2, 2, 2,
    3, 3, 3, 3, 3, 3, 3, 3,
    4, 4, 4, 4, 4, 4, 4, 4,
    5, 5, 5, 5, 5, 5, 5, 5,
    6, 6, 6, 6, 6, 6,
    7, 7, 7, 7,
])

locj = np.array([
    2, 3, 4, 5,
    1, 2, 3, 4, 5, 6,
    0, 1, 2, 3, 4, 5, 6, 7,
    0, 1, 2, 3, 4, 5, 6, 7,
    0, 1, 2, 3, 4, 5, 6, 7,
    0, 1, 2, 3, 4, 5, 6, 7,
    1, 2, 3, 4, 5, 6,
    2, 3, 4, 5,
])

# integer location <---> tuple coordinates on 8x8 board:
location_to_coordinates = list(zip(loci, locj))
coordinates_to_location = {_coords: _it for _it, _coords in enumerate(location_to_coordinates)}

# use this to avoid repeated set conversion:
set_location_to_coordinates = set(location_to_coordinates)

# pre-calculate the coordinates that are orthogonal and legal for each board coordinates:
corners = {(0, 0), (0, 1), (1, 0), (7, 0), (7, 1), (6, 0),
           (0, 7), (0, 6), (1, 7), (7, 7), (7, 6), (6, 7)}

coordinates_to_orthogonal_neighbors = {
    (_i, _j): { (max(0, _i - 1), _j), ( min(7, _i + 1), _j),
                (_i, max(0, _j - 1) ), (_i, min(7, _j + 1) ) } - {(_i, _j)} - corners
    for (_i, _j) in location_to_coordinates}

### Augmentation: precalculate indices for board locations after flipping and rotation:
board = np.zeros((8, 8), dtype=int)

# legal sites occupied by location integer label:
board[loci, locj] = range(52)

rot_board_1 = np.rot90(board, k=1, axes=(0, 1))
rot_board_2 = np.rot90(board, k=2, axes=(0, 1))
rot_board_3 = np.rot90(board, k=3, axes=(0, 1))
flip_rot_board = np.fliplr(board)
flip_rot_board_1 = np.fliplr(rot_board_1)
flip_rot_board_2 = np.fliplr(rot_board_2)
flip_rot_board_3 = np.fliplr(rot_board_3)

d4_permutation_indices = [
    board[loci, locj],
    rot_board_1[loci, locj],
    rot_board_2[loci, locj],
    rot_board_3[loci, locj],
    flip_rot_board[loci, locj],
    flip_rot_board_1[loci, locj],
    flip_rot_board_2[loci, locj],
    flip_rot_board_3[loci, locj],
]

### State: efficiently manage one hot creation and permutations of the parent tensor:
EYE = np.eye(19, dtype=int)[ :-1]  # shape (18, 19)

# Use indices to efficiently swap active and passive slices:
SWAP_ACTIVE_PASSIVE_INDEX = [
    ### board:
    0, 1, 2, 3, 4, 5, # unflipped
    12, 13, 14, 15, 16, 17, # active <----> passive
    6, 7, 8, 9, 10, 11,

    ### color orders:
    24, 25, 26, 27, 28, 29, # active <----> passive
    18, 19, 20, 21, 22, 23,

    ### bowl tokens:
    36, 37, 38, 39, 40, 41,
    30, 31, 32, 33, 34, 35,

    ### is no more placing:
    42,

    ### score:
    44,
    43,

    ### current bowl token value:
    46,
    45,
]
