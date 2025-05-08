""" move from the physical 2D board representation, with neighbours orthogaonl, to the list/ vector representation
"""

from enum import Enum


class Directions(Enum):
    RIGHT = 0, 1
    UP = -1, 0
    LEFT = 0, -1
    DOWN = 1, 0

    def __init__(self, i: int, j: int) -> None:
        self.i = i
        self.j = j

# the 52 board locations as coordinates (i, j):
bad_locs = {
    (0, 0), (0, 1), (1, 0),
    (7, 0), (7, 1), (6, 0),
    (0, 7), (0, 6), (1, 7),
    (7, 7), (6, 7), (7, 6),
}

location_to_coordinates = [0] * 52
coordinates_to_location = {}
orthogonal_neighbors = {}
_count = 0

for _i in range(8):
    for _j in range(8):
        _loc = (_i, _j)

        if _loc in bad_locs:
            continue

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
            if (_i , _j + 1) not in bad_locs:
                _neighbors.append((_i, _j + 1))

        orthogonal_neighbors[(_i, _j)] = set(_neighbors)
        _count += 1

zipped_loc_to_coords = list(zip(*location_to_coordinates))
loci = list(zipped_loc_to_coords[0])
locj = list(zipped_loc_to_coords[1])
