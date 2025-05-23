"""
There are many symmetries that can be exploited for the game of patterns.

We have 8 dihedral group symmetries; bool flip and 4 rotations.

We also have any permutations of the colors.

don't apply to colors 0 1 in the starting position

Note we ONLY use this at evaluation, which means we are applying it to the tensor state, not the
game. However, it can be applied to game from which tensor state is then created... easiest

To create tensor state from game you require:
board, active_order, passive_order, active_token, passive_token
"""
from game import Patterns

import numpy as np
import torch
import random


class Augmentor:
    def __init__(self, game: Patterns):
        self.game = game

    def full_augment(self):
        self.flip()
        self.rotate()
        self.permute_colors()

    def flip(self):
        """ randomly flip the board...
        """
        # 50% chance to flip the board:
        if random.random() < 0.5:
            return

        self.game.active_board = np.fliplr(self.game.active_board)

        # will also need to flip all the locations in the color groups to be able to plot:
        for _color in range(6):
            self.game.active_color_groups[_color] = [self.flip_coordinates(_x)
                                                     for _x in self.game.active_color_groups[_color]]
            self.game.passive_color_groups[_color] = [self.flip_coordinates(_x)
                                                     for _x in self.game.passive_color_groups[_color]]

    def rotate(self):
        """ choose a random rotation from 4
        """
        _rand = random.random()

        if _rand < 0.25:
            return

        if _rand < 0.5:
            k = 1

        elif _rand < 0.75:
            k = 2

        else:
            k = 3

        self.game.active_board = np.rot90(self.game.active_board, k=k)

        for _color in range(6):
            self.game.active_color_groups[_color] = [self.rotate_coordinates(_x, k)
                                                     for _x in self.game.active_color_groups[_color]]
            self.game.passive_color_groups[_color] = [self.rotate_coordinates(_x, k)
                                                      for _x in self.game.passive_color_groups[_color]]

    def flip_coordinates(self, coordinates: tuple[int, int]) -> tuple[int, int]:
        """ return the flipped coordinates. So (1, 3) -> (1, 4)
        """
        return (coordinates[0], 7 - coordinates[1])

    def rotate_coordinates(self, coordinates: tuple[int, int], k: int) -> tuple[int, int]:
        """ return the rotated coordinates by 90 degrees times k. (7-j, i)
        """
        tup = coordinates[0], coordinates[1]

        for _ in range(k):
            tup = 7 - tup[1], tup[0]

        return tup

    def permute_colors(self):
        """ choose two colors to swap
        """
        if random.random() < 0.5:
            return

        c1, c2 = random.sample([0, 1, 2, 3, 4, 5], 2)

        # swap all instances of the two colors:
        self.game.active_color_order[c1], self.game.active_color_order[c2] = (
            self.game.active_color_order[c2], self.game.active_color_order[c1])

        self.game.passive_color_order[c1], self.game.passive_color_order[c2] = (
            self.game.passive_color_order[c2], self.game.passive_color_order[c1]
        )

        # swap the bowl tokens if necessary:
        if self.game.active_bowl_token == c1:
            self.game.active_bowl_token = c2

        if self.game.active_bowl_token == c2:
            self.game.active_bowl_token = c1

        if self.game.active_bowl_token == c1:
            self.game.active_bowl_token = c2

        if self.game.active_bowl_token == c2:
            self.game.active_bowl_token = c1

        c1_ind = self.game.active_board == c1
        c2_ind = self.game.active_board == c2

        self.game.active_board[c1_ind] = c2
        self.game.active_board[c2_ind] = c1

        # swap the color groups:
        self.game.active_color_groups[c1], self.game.active_color_groups[c2] = (
            self.game.active_color_groups[c2], self.game.active_color_groups[c1]
        )

        self.game.passive_color_groups[c1], self.game.passive_color_groups[c2] = (
            self.game.passive_color_groups[c2], self.game.passive_color_groups[c1]
        )


class StateAugmentor:
    def __init__(self, state: torch.tensor) -> None:
        """ perform the same augmentations but from the tensor rather than
        the more natural state representation
        """
        self.state = state
        self.board = state[:18, :, :].numpy()
        self.p1order = state[18:54, 0, 0].numpy()
        self.p2order = state[54:90, 0, 0].numpy()
        self.p1bowl = state[90:96, 0, 0].numpy()
        self.p2bowl = state[96:, 0, 0].numpy()

    def full_augment(self) -> None:
        self.flip()
        self.rotate()
        self.permute_colors()
        self.combine()

    def combine(self) -> None:
        """ restack the state:
        """
        p1order = torch.zeros((36, 8, 8))
        p2order = torch.zeros((36, 8, 8))

        p1order[self.p1order] = 1
        p2order[self.p2order] = 1

        p1bowl = torch.zeros((6, 8, 8))
        p2bowl = torch.zeros((6, 8, 8))

        p1bowl[self.p1bowl] = 1
        p2bowl[self.p2bowl] = 1

        self.state = torch.concat([torch.tensor(_x)
                                   for _x in [self.board, p1order, p2order, p1bowl, p2bowl]])

    def flip(self) -> None:
        """ randomly flip the board...
        """
        # 50% chance to flip the board:
        if random.random() < 0.5:
            return

        self.board = np.flip(self.board, axis=2)

    def rotate(self) -> None:
        """ choose a random rotation from 4
        """
        _rand = random.random()

        if _rand < 0.25:
            return

        if _rand < 0.5:
            k = 1

        elif _rand < 0.75:
            k = 2

        else:
            k = 3

        self.board = np.rot90(self.board, k=k, axes=(1, 2))

    def permute_order(self, order: np.ndarray, c1: int, c2: int) -> None:
        """ permute the colors c1 and c2 in the color orders for a given player:
        """
        # get indices:
        c1_order, = np.where(order[(6 * c1) : (6 * c1 + 6)])
        c2_order, = np.where(order[(6 * c2): (6 * c2 + 6)])

        # zero the old:
        order[6 * c1 + c1_order[0]] = 0
        order[6 * c2 + c2_order[0]] = 0

        # unit for the new:
        order[6 * c1 + c2_order[0]] = 1
        order[6 * c2 + c1_order[0]] = 1

    def permute_colors(self, p=0.5) -> None:
        """ choose two colors to swap
        """
        if random.random() < p:
            return

        # select two colors:
        c1, c2 = random.sample([0, 1, 2, 3, 4, 5], 2)

        # swap axes in 1st board channel:
        self.board[c1, :, :], self.board[c2, :, :] = self.board[c2, :, :], self.board[c1, :, :]
        self.board[c1 + 6, :, :], self.board[c2 + 6, :, :] = self.board[c2 + 6, :, :], self.board[c1 + 6, :, :]
        self.board[c1 + 12, :, :], self.board[c2 + 12, :, :] = self.board[c2 + 12, :, :], self.board[c1 + 12, :, :]

        # swap the colors in the ordering for each player:
        self.permute_order(self.p1order, c1, c2)
        self.permute_order(self.p2order, c1, c2)

        # swap the bowl tokens if necessary:
        self.p1bowl[[c1, c2]] = self.p1bowl[[c2, c1]]
        self.p2bowl[[c1, c2]] = self.p2bowl[[c2, c1]]
