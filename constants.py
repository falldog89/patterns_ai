""" move from the physical 2D board representation, with neighbours orthogaonl, to the list/ vector representation
"""
import numpy as np
import torch


# easiest way is to just make a board:
bad_locs = {
    (0, 0), (0, 1), (1, 0),
    (7, 0), (7, 1), (6, 0),
    (0, 7), (0, 6), (1, 7),
    (7, 7), (6, 7), (7, 6),
}

location_to_coordinates = [0] * 52
coordinates_to_location = {}
orthogonal_neighbors = {}
board = np.zeros((8, 8), dtype=int)

_count = 0

for _i in range(8):
    for _j in range(8):
        _loc = (_i, _j)

        # do not consider the corners:
        if _loc in bad_locs:
            continue

        board[_i, _j] = _count
        location_to_coordinates[_count] = _loc
        coordinates_to_location[_loc] = _count

        _neighbors = []

        if _i > 0:
            if (_i - 1, _j) not in bad_locs:
                _neighbors.append((_i - 1, _j))

        if _i < 7:
            if (_i + 1, _j) not in bad_locs:
                _neighbors.append((_i + 1, _j))

        if _j > 0:
            if (_i, _j - 1) not in bad_locs:
                _neighbors.append((_i, _j - 1))

        if _j < 7:
            if (_i, _j + 1) not in bad_locs:
                _neighbors.append((_i, _j + 1))

        orthogonal_neighbors[(_i, _j)] = set(_neighbors)
        _count += 1

# indices for the location to coordinates mapping:
zipped_loc_to_coords = list(zip(*location_to_coordinates))
loci = list(zipped_loc_to_coords[0])
locj = list(zipped_loc_to_coords[1])

# now get the rotations and flips from board:
rot_board_1 = np.rot90(board, k=1, axes=(0, 1))
rot_board_2 = np.rot90(board, k=2, axes=(0, 1))
rot_board_3 = np.rot90(board, k=3, axes=(0, 1))
flip_board = np.fliplr(board)

# use these to point from the original location to the transformed location:
location_to_rot1 = rot_board_1[loci, locj]
location_to_rot2 = rot_board_2[loci, locj]
location_to_rot3 = rot_board_3[loci, locj]
location_to_flip = flip_board[loci, locj]

# use this to avoid repeated set conversion:
set_location_to_coordinates = set(location_to_coordinates)

# For efficiently manipulating tensor state:

# for more efficiently creating one hot:
EYE = np.eye(19, dtype=int)[ :-1]  # shape (18, 19)

# Accessing the tensor using this index list swaps the indicies for passive and active players:
SWAP_ACTIVE_PASSIVE_INDEX = [
    ### board:
    0, 1, 2, 3, 4, 5,
    12, 13, 14, 15, 16, 17,
    6, 7, 8, 9, 10, 11,

    ### color orders:
    24, 25, 26, 27, 28, 29,
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

#
# ### coordinates for the board spaces on an 8x8 board with the corners cut off:
# loci = [
#     0, 0, 0, 0,
#     1, 1, 1, 1, 1, 1,
#     2, 2, 2, 2, 2, 2, 2, 2,
#     3, 3, 3, 3, 3, 3, 3, 3,
#     4, 4, 4, 4, 4, 4, 4, 4,
#     5, 5, 5, 5, 5, 5, 5, 5,
#     6, 6, 6, 6, 6, 6,
#     7, 7, 7, 7,
# ]
#
# locj = [
#     2, 3, 4, 5,
#     1, 2, 3, 4, 5, 6,
#     0, 1, 2, 3, 4, 5, 6, 7,
#     0, 1, 2, 3, 4, 5, 6, 7,
#     0, 1, 2, 3, 4, 5, 6, 7,
#     0, 1, 2, 3, 4, 5, 6, 7,
#     1, 2, 3, 4, 5, 6,
#     2, 3, 4, 5,
# ]
#
#
