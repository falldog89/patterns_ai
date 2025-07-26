""" Class to exploit symmetries in patterns states and augment the training data.

A Patterns board has 8 dihedral group symmetries; bool flip and 4 rotations.

Further, the colors may be permuted, giving a further 6! = 720 equivalent states.

Therefore, each state can represent 720 * 8 = 5000 equivalent games (!)

We do not apply augmentations to permute colors for the first turns, as the game is restricted
to set up with colors 0 and 1 in the bowls, and 2-5 in the middle 4 locations.

"""
from game import Patterns

import numpy as np
import torch
import random
from typing import Optional

from constants import location_to_coordinates
from constants import orthogonal_neighbors

from constants import location_to_rot1
from constants import location_to_rot2
from constants import location_to_rot3
from constants import location_to_flip


class StateAugmentor:
    def __init__(self, state: torch.tensor, visit_counts: Optional[torch.tensor] = None) -> None:
        """ perform the same augmentations but from the tensor rather than
        the more natural state representation

        [board_tensor, order_tensor, bowl_tensor, placing_tensor, score_tensor, token_value_tensor]
        """
        self.state = state
        self.visit_counts = visit_counts
        self.board = state[:18, :, :].numpy()
        self.p1order = state[18:24, :, :]
        self.p2order = state[24:30, :, :]
        self.p1bowl = state[30:36, :, :] # bool arrays
        self.p2bowl = state[36:42, :, :]
        self.no_more_placing = state[42, :, :][None]
        self.scores = state[43:45, :, :]
        self.token_values = state[45:, :, :]

    def full_augment(self) -> None:
        self.flip()
        self.rotate()
        self.permute_colors()
        self.combine()

    def combine(self) -> None:
        """ restack the state:
        """
        # # tile and permute the orders:
        # p1order = torch.tile(torch.tensor(self.p1order), (8, 8, 1))
        # p2order = torch.tile(torch.tensor(self.p2order), (8, 8, 1))
        #
        # # permute to get layers of constant float value:
        # p1order = p1order.permute( (2, 0, 1))
        # p2order = p2order.permute((2, 0, 1))
        #
        # p1bowl = torch.zeros((6, 8, 8))
        # p2bowl = torch.zeros((6, 8, 8))
        #
        # # use the bool array to assign the layers correctly
        # p1bowl[self.p1bowl == 1] = 1
        # p2bowl[self.p2bowl == 1] = 1

        # board must be copied, as the flip and rotate operations are views in numpy, and torch
        # cannot cope with negative strides in these views:
        self.state = torch.concat([torch.tensor(self.board.copy()),
                                   self.p1order, self.p2order,
                                   self.p1bowl, self.p2bowl,
                                   self.no_more_placing,
                                   self.scores,
                                   self.token_values])

    def permute_visit_counts(self, loc_permutation: np.ndarray) -> None:
        """ apply the permutation to the first 52 and then the second 52 of the action space:
        """
        if self.visit_counts is None:
            return

        self.visit_counts[:52] = self.visit_counts[loc_permutation]
        self.visit_counts[52:104] = self.visit_counts[52:104][loc_permutation]

    def flip(self) -> None:
        """ randomly flip the board...
        """
        # 50% chance to flip the board:
        if random.random() < 0.5:
            return

        self.board = np.flip(self.board, axis=2)

        # flipping and placing actions are flipped:
        self.permute_visit_counts(location_to_flip)

    def rotate(self) -> None:
        """ choose a random rotation from 4
        """
        _rand = random.random()

        if _rand < 0.25:
            return

        if _rand < 0.5:
            self.permute_visit_counts(location_to_rot1)
            k = 1

        elif _rand < 0.75:
            self.permute_visit_counts(location_to_rot2)
            k = 2

        else:
            self.permute_visit_counts(location_to_rot3)
            k = 3

        self.board = np.rot90(self.board, k=k, axes=(1, 2))

    def permute_colors(self, p=0.0) -> None:
        """ choose two colors to swap.

        todo: do more than just switch two colors: apply multiple times:
        """
        if random.random() < p:
            return

        # permutation of the color array:
        shuffled_colors = [0, 1, 2, 3, 4, 5]
        random.shuffle(shuffled_colors)

        unflipped_board = self.board[:6][shuffled_colors]
        active_board = self.board[6:12][shuffled_colors]
        passive_board = self.board[12:][shuffled_colors]

        self.board = np.concatenate([unflipped_board, active_board, passive_board])

        # same permutation for bowl pieces and color orders:
        self.p1order = self.p1order[shuffled_colors]
        self.p2order = self.p2order[shuffled_colors]

        self.p1bowl = self.p1bowl[shuffled_colors]
        self.p2bowl = self.p2bowl[shuffled_colors]
        #
        # unflipped_board = unflipped_boared
        # self.board = self.board[shuffled_colors]
        # # select two colors:
        # c1, c2 = random.sample([0, 1, 2, 3, 4, 5], 2)
        #
        # boardc1 = self.board[c1, :, :]
        # boardc2 = self.board[c2, :, :]
        #
        # np.permute(self.board
        # # swap axes in 1st board channel:
        # self.board[c1, :, :], self.board[c2, :, :] = self.board[c2, :, :], self.board[c1, :, :]
        # self.board[c1 + 6, :, :], self.board[c2 + 6, :, :] = self.board[c2 + 6, :, :], self.board[c1 + 6, :, :]
        # self.board[c1 + 12, :, :], self.board[c2 + 12, :, :] = self.board[c2 + 12, :, :], self.board[c1 + 12, :, :]

        # swap the colors in the ordering for each player:
        # # new: these are now just vectors, so they just get swapped as per:
        # self.p1order[c1], self.p1order[c2] = self.p1order[c2], self.p1order[c1]
        # self.p2order[c1], self.p2order[c2] = self.p2order[c2], self.p2order[c1]
        #
        # # swap the bowl tokens if necessary:
        # self.p1bowl[c1], self.p1bowl[c2] = self.p1bowl[c2], self.p1bowl[c1]
        # self.p2bowl[c1], self.p2bowl[c2] = self.p2bowl[c2], self.p2bowl[c1]

