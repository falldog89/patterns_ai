""" move from the physical 2D board representation, with neighbours orthogaonl, to the list/ vector representation
"""

import numpy as np
from enum import Enum


class Directions(Enum):
    RIGHT = 0, 1
    UP = -1, 0
    LEFT = 0, -1
    DOWN = 1, 0

    def __init__(self, i: int, j: int) -> None:
        self.i = i
        self.j = j

list_vec = [0] * 52
numpy_arr = -1 * np.ones((8, 8), dtype=int)

all_adj = []
_count = 0

for _i in range(8):
    for _j in range(8):
        if (_i + _j) < 2:
            continue

        if ((7 - _i) + (7 - _j)) < 2:
            continue

        if ((7 - _i) + _j) < 2:
            continue

        if (_i + (7 - _j)) < 2:
            continue

        numpy_arr[_i, _j] = _count
        list_vec[_count] = (_i, _j)
        _count += 1


zipped_loc_to_coords = list(zip(*list_vec))
loci = list(zipped_loc_to_coords[0])
locj = list(zipped_loc_to_coords[1])

# now check neighbours
all_neighbours = {}

for _i in range(8):
    for _j in range(8):
        current_neighbours = []
        curr = numpy_arr[_i, _j]

        if curr < 0:
            continue

        for _dir in Directions:
            _ti = _i + _dir.i
            _tj = _j + _dir.j

            if ((_ti >= 0) == (_ti < 8)) and ((_tj >= 0) == (_tj < 8)):
                target = numpy_arr[_ti, _tj]
                if target >= 0:
                    current_neighbours.append(target)

        all_neighbours[curr] = set(current_neighbours)

