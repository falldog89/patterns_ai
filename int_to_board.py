""" move from the physical 2D board representation, with neighbours orthogaonl, to the list/ vector representation
"""
import numpy as np

#
# location_to_coordinates = [0] * 52
# coordinates_to_location = {}
# orthogonal_neighbors = {}
# _count = 0
#
# for _i in range(8):
#     for _j in range(8):
#         _loc = (_i, _j)
#
#         if _loc in bad_locs:
#             continue
#
#         location_to_coordinates[_count] = _loc
#         coordinates_to_location[_loc] = _count
#
#         _neighbors = []
#
#         if _i > 0:
#             if (_i - 1, _j) not in bad_locs:
#                 _neighbors.append((_i - 1, _j))
#
#         if _i < 7:
#             if (_i + 1, _j) not in bad_locs:
#                 _neighbors.append((_i + 1, _j))
#
#         if _j > 0:
#             if (_i, _j - 1) not in bad_locs:
#                 _neighbors.append((_i, _j - 1))
#
#         if _j < 7:
#             if (_i , _j + 1) not in bad_locs:
#                 _neighbors.append((_i, _j + 1))
#
#         orthogonal_neighbors[(_i, _j)] = set(_neighbors)
#         _count += 1
#
# zipped_loc_to_coords = list(zip(*location_to_coordinates))
# loci = list(zipped_loc_to_coords[0])
# locj = list(zipped_loc_to_coords[1])
#
#
# # for the augmentation we want a mapping from the original action space to the flipped action space.
# # we want the same for each of the three rotations:
# starts_ends = [-1, 3, 9, 17, 25, 33, 41, 47, 51]
# flip = []
# for start, end in zip(starts_ends, starts_ends[1:]):
#     flip += list(range(end, start, -1))
#
#
# rot1 = list(range(3, -1, -1)) + list(range(9, 3, -1)) + list(range())

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

