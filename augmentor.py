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
from typing import Optional

from int_to_board import location_to_coordinates
from int_to_board import orthogonal_neighbors

from int_to_board import location_to_rot1
from int_to_board import location_to_rot2
from int_to_board import location_to_rot3
from int_to_board import location_to_flip


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

    def create_game_from_state(self) -> Patterns:
        """ Use the second formulation of tensor state to create the full patterns game:

        here, there are 18 layers of the tensor for the board, as above (0-18)

        The following 12 layers are float layers representing the order that a color was taken:

        finally there are 2 (float) layers for score and for is taken

        Note that the float layers are normalized to sit between 0 and 1:

        [board_tensor, order_tensor, bowl_tensor, placing_tensor, score_tensor, token_value_tensor]
        =
        [18, 12, 12, 1, 2, 2] = 47
        """
        numpy_state = self.state.numpy()

        board_layers = numpy_state[:18, :, :]

        p1_order_layers = numpy_state[18:24, 0, 0]
        p2_order_layers = numpy_state[24:30, 0, 0]

        p1_bowl_tokens = numpy_state[30:36, 0, 0]
        p2_bowl_tokens = numpy_state[36:42, 0, 0]

        # create game to assign attributes to:
        new_game = Patterns()

        # also populate the is no more placing characteristic:
        if numpy_state[42, 0, 0] == 1:
            new_game._is_no_more_placing = True

        else:
            # set to NONE so that the property is checked, not to False which means already checked.
            new_game._is_no_more_placing = None

        # assume that if we are validating, or wishing to create game from state, the game has started proper:
        # todo reconsider this for the long future, when we are predicting first moves from the board state...
        new_game.is_game_started = True
        new_game.first_turn_passed = False

        self.assign_bowl_tokens(new_game, p1_bowl_tokens, p2_bowl_tokens)
        self.populate_board(new_game, board_layers)
        self.assign_color_orders(new_game, p1_order_layers, p2_order_layers)
        self.assign_orth_flip_groups(new_game, )

        return new_game

    @staticmethod
    def assign_bowl_tokens(new_game: Patterns,
                           p1_bowl_tokens: torch.tensor,
                           p2_bowl_tokens: torch.tensor) -> None:
        """ use the tensors to assign active and passive bowl tokens to the reconstructed game
        """
        p1_color = None
        p2_color = None

        for _color, (_entry1, _entry2) in enumerate(zip(p1_bowl_tokens, p2_bowl_tokens)):
            if _entry1 == 1:
                p1_color = _color

            if _entry2 == 1:
                p2_color = _color

            if (p1_color is not None) and (p2_color is not None):
                break

        new_game.active_bowl_token = p1_color
        new_game.passive_bowl_token = p2_color

    @staticmethod
    def populate_board(new_game: Patterns,
                       board_tensor: torch.tensor) -> None:
        """ use the board tensor (18, 8, 8) to populate the active and passive boards in the game
        """
        # populate the active and passive boards:
        new_game.active_board = np.zeros((8, 8), dtype=int)
        new_game.passive_board = np.zeros((8, 8), dtype=int)

        vals, iind, jind = np.where(board_tensor)
        new_game.active_board[iind, jind] = vals
        new_game.passive_board[iind, jind] = vals

        # masks for the active and passive tokens:
        active_mask = np.logical_and(new_game.passive_board >= 6, new_game.passive_board < 12)
        passive_mask = new_game.passive_board >= 12

        # the difference for the passive board and active board is simply that the piece representations
        # are 6-11 instead of 12-15 for the active, and reverse for passive:
        new_game.passive_board[active_mask] += 6
        new_game.passive_board[passive_mask] -= 6

        # can use either board for this step:
        unflipped_i, unflipped_j = np.where(new_game.active_board < 6)
        unflipped_locations = set(list(zip(unflipped_i, unflipped_j)))

        # subtract the unflipped locations from all the possible locations:
        new_game.flipped_locations = set(location_to_coordinates) - unflipped_locations

    @staticmethod
    def assign_color_orders(new_game: Patterns,
                            p1_order_layers: torch.tensor,
                            p2_order_layers: torch.tensor) -> None:
        """ use the 6 deep layers to assign the order that colors were taken for each player:
        NOTE must occur after the active board is correctly assigned:
        """
        new_game.active_placing_number, new_game.passive_placing_number = 1, 1

        for _color, (_p1layer, _p2layer) in enumerate(zip(p1_order_layers, p2_order_layers)):
            _p1order = round(_p1layer * 6)
            _p2order = round(_p2layer * 6)

            if _p1order > 0:
                new_game.active_color_order[_color] = _p1order

                # store the full color group:
                _i, _j = np.where(new_game.active_board == (_color + 6))
                new_game.active_color_groups[_color] = list(zip(_i, _j))
                new_game.active_placing_number += 1

            if _p2order > 0:
                new_game.passive_color_order[_color] = _p2order

                # store the full color group:
                _i, _j = np.where(new_game.active_board == (_color + 12))
                new_game.passive_color_groups[_color] = list(zip(_i, _j))
                new_game.passive_placing_number += 1

    @staticmethod
    def assign_orth_flip_groups(new_game: Patterns):
        """ assign the flipping and orthogonal groups so that the game is functional going forward
        """
        # populate the orthogonal groups and flipping groups from the color groups:
        for _color in range(6):
            active_group = new_game.active_color_groups[_color]
            passive_group = new_game.passive_color_groups[_color]

            active_orthogs = []
            passive_orthogs = []

            # find all the orthogonal locations then remove any that are flipped:
            for _coords in active_group:
                active_orthogs.extend(list(orthogonal_neighbors[_coords]))

            for _coords in passive_group:
                passive_orthogs.extend(list(orthogonal_neighbors[_coords]))

            new_game.active_orthogonal_groups[_color] = set(active_orthogs) - new_game.flipped_locations
            new_game.passive_orthogonal_groups[_color] = set(passive_orthogs) - new_game.flipped_locations

            new_game.active_flipping_groups[_color] = set([_x for _x in new_game.active_orthogonal_groups[_color]
                                                           if new_game.active_board[_x] == _color
                                                           if _x not in new_game.passive_orthogonal_groups[_color]])

            new_game.passive_flipping_groups[_color] = set([_x for _x in new_game.passive_orthogonal_groups[_color]
                                                            if new_game.active_board[_x] == _color
                                                            if _x not in new_game.active_orthogonal_groups[_color]])






#
#
# # todo today: rewrite the augmentor to make sure we are looking at the new state set up and TEST it before training!
# class StateAugmentor:
#     def __init__(self, state: torch.tensor, visit_counts: Optional[torch.tensor] = None) -> None:
#         """ perform the same augmentations but from the tensor rather than
#         the more natural state representation
#         """
#         self.state = state
#         self.visit_counts = visit_counts
#         self.board = state[:18, :, :].numpy()
#         self.p1order = state[18:h, 0, 0].numpy()
#         self.p2order = state[54:90, 0, 0].numpy()
#         self.p1bowl = state[90:96, 0, 0].numpy()
#         self.p2bowl = state[96:, 0, 0].numpy()
#
#     def full_augment(self) -> None:
#         self.flip()
#         self.rotate()
#         self.permute_colors()
#         self.combine()
#
#     def combine(self) -> None:
#         """ restack the state:
#         """
#         # if no more placing:
#         if self.state[-1, 0, 0] == 1:
#             no_more_placing = torch.ones((1, 8, 8))
#
#         else:
#             no_more_placing = torch.zeros((1, 8, 8))
#
#         p1order = torch.zeros((36, 8, 8))
#         p2order = torch.zeros((36, 8, 8))
#
#         p1order[self.p1order] = 1
#         p2order[self.p2order] = 1
#
#         p1bowl = torch.zeros((6, 8, 8))
#         p2bowl = torch.zeros((6, 8, 8))
#
#         p1bowl[self.p1bowl] = 1
#         p2bowl[self.p2bowl] = 1
#
#         # board must be copied, as the flip and rotate operations are views in numpy, and torch
#         # cannot cope with negative strides in these views:
#         self.state = torch.concat([torch.tensor(self.board.copy()),
#                                    p1order, p2order,
#                                    p1bowl, p2bowl,
#                                    no_more_placing])
#
#     def permute_visit_counts(self, loc_permutation: np.ndarray) -> None:
#         """ apply the permutation to the first 52 and then the second 52 of the action space:
#         """
#         if self.visit_counts is None:
#             return
#
#         self.visit_counts[:52] = self.visit_counts[loc_permutation]
#         self.visit_counts[52:104] = self.visit_counts[52:104][loc_permutation]
#
#     def flip(self) -> None:
#         """ randomly flip the board...
#         """
#         # 50% chance to flip the board:
#         if random.random() < 0.5:
#             return
#
#         self.board = np.flip(self.board, axis=2)
#
#         # flipping and placing actions are flipped:
#         self.permute_visit_counts(location_to_flip)
#
#     def rotate(self) -> None:
#         """ choose a random rotation from 4
#         """
#         _rand = random.random()
#
#         if _rand < 0.25:
#             return
#
#         if _rand < 0.5:
#             self.permute_visit_counts(location_to_rot1)
#             k = 1
#
#         elif _rand < 0.75:
#             self.permute_visit_counts(location_to_rot2)
#             k = 2
#
#         else:
#             self.permute_visit_counts(location_to_rot3)
#             k = 3
#
#         self.board = np.rot90(self.board, k=k, axes=(1, 2))
#
#     @staticmethod
#     def permute_order(order: np.ndarray, c1: int, c2: int) -> None:
#         """ permute the colors c1 and c2 in the color orders for a given player:
#
#         Note that if there is no color taken, there where will give nothing.
#         """
#         # get indices:
#         c1_order, = np.where(order[(6 * c1) : (6 * c1 + 6)])
#         c2_order, = np.where(order[(6 * c2): (6 * c2 + 6)])
#
#         # zero the old, True the new:
#         if len(c1_order) > 0:
#             order[6 * c1 + c1_order[0]] = 0
#             order[6 * c2 + c1_order[0]] = 1
#
#         if len(c2_order) > 0:
#             order[6 * c2 + c2_order[0]] = 0
#             order[6 * c1 + c2_order[0]] = 1
#
#     def permute_colors(self, p=0.5) -> None:
#         """ choose two colors to swap
#         """
#         if random.random() < p:
#             return
#
#         # select two colors:
#         c1, c2 = random.sample([0, 1, 2, 3, 4, 5], 2)
#
#         # swap axes in 1st board channel:
#         self.board[c1, :, :], self.board[c2, :, :] = self.board[c2, :, :], self.board[c1, :, :]
#         self.board[c1 + 6, :, :], self.board[c2 + 6, :, :] = self.board[c2 + 6, :, :], self.board[c1 + 6, :, :]
#         self.board[c1 + 12, :, :], self.board[c2 + 12, :, :] = self.board[c2 + 12, :, :], self.board[c1 + 12, :, :]
#
#         # swap the colors in the ordering for each player:
#         self.permute_order(self.p1order, c1, c2)
#         self.permute_order(self.p2order, c1, c2)
#
#         # swap the bowl tokens if necessary:
#         self.p1bowl[[c1, c2]] = self.p1bowl[[c2, c1]]
#         self.p2bowl[[c1, c2]] = self.p2bowl[[c2, c1]]
#
#     def create_game_from_state(self) -> Patterns:
#         """ Use the second formulation of tensor state to create the full patterns game:
#
#         here, there are 18 layers of the tensor for the board, as above (0-18)
#
#         The following 12 layers are float layers representing the order that a color was taken:
#
#         finally there are 2 (float) layers for score and for is taken
#
#         Note that the float layers are normalized to sit between 0 and 1:
#
#         [board_tensor, order_tensor, bowl_tensor, placing_tensor, score_tensor, token_value_tensor]
#         =
#         [18, 12, 12, 1, 2, 2] = 47
#         """
#         numpy_state = self.state.numpy()
#
#         board_layers = numpy_state[:18, :, :]
#
#         p1_order_layers = numpy_state[18:24, 0, 0]
#         p2_order_layers = numpy_state[24:30, 0, 0]
#
#         p1_bowl_tokens = numpy_state[30:36, 0, 0]
#         p2_bowl_tokens = numpy_state[36:42, 0, 0]
#
#         # create game to assign attributes to:
#         new_game = Patterns()
#
#         # also populate the is no more placing characteristic:
#         if numpy_state[42, 0, 0] == 1:
#             new_game._is_no_more_placing = True
#
#         else:
#             # set to NONE so that the property is checked!
#             new_game._is_no_more_placing = None
#
#         # assume that if we are validating, or wishing to create game from state, the game has started proper:
#         new_game.is_game_started = True
#         new_game.first_turn_passed = False
#
#         self.assign_bowl_tokens(new_game, p1_bowl_tokens, p2_bowl_tokens)
#         self.populate_board(new_game, board_layers)
#         self.assign_color_orders(new_game, p1_order_layers, p2_order_layers)
#         self.assign_orth_flip_groups(new_game, )
#
#         return new_game
#
#     @staticmethod
#     def assign_bowl_tokens(new_game: Patterns,
#                            p1_bowl_tokens: torch.tensor,
#                            p2_bowl_tokens: torch.tensor) -> None:
#         """ use the tensors to assign active and passive bowl tokens to the reconstructed game
#         """
#         p1_color = None
#         p2_color = None
#
#         for _color, (_entry1, _entry2) in enumerate(zip(p1_bowl_tokens, p2_bowl_tokens)):
#             if _entry1 == 1:
#                 p1_color = _color
#
#             if _entry2 == 1:
#                 p2_color = _color
#
#             if (p1_color is not None) and (p2_color is not None):
#                 break
#
#         new_game.active_bowl_token = p1_color
#         new_game.passive_bowl_token = p2_color
#
#     @staticmethod
#     def populate_board(new_game: Patterns,
#                        board_tensor: torch.tensor) -> None:
#         """ use the board tensor (18, 8, 8) to populate the active and passive boards in the game
#         """
#         # populate the active and passive boards:
#         new_game.active_board = np.zeros((8, 8), dtype=int)
#         new_game.passive_board = np.zeros((8, 8), dtype=int)
#
#         vals, iind, jind = np.where(board_tensor)
#         new_game.active_board[iind, jind] = vals
#         new_game.passive_board[iind, jind] = vals
#
#         # masks for the active and passive tokens:
#         active_mask = np.logical_and(new_game.passive_board >= 6, new_game.passive_board < 12)
#         passive_mask = new_game.passive_board >= 12
#
#         # the difference for the passive board and active board is simply that the piece representations
#         # are 6-11 instead of 12-15 for the active, and reverse for passive:
#         new_game.passive_board[active_mask] += 6
#         new_game.passive_board[passive_mask] -= 6
#
#         # can use either board for this step:
#         unflipped_i, unflipped_j = np.where(new_game.active_board < 6)
#         unflipped_locations = set(list(zip(unflipped_i, unflipped_j)))
#
#         # subtract the unflipped locations from all the possible locations:
#         new_game.flipped_locations = set(location_to_coordinates) - unflipped_locations
#
#     @staticmethod
#     def assign_color_orders(new_game: Patterns,
#                             p1_order_layers: torch.tensor,
#                             p2_order_layers: torch.tensor) -> None:
#         """ use the 6 deep layers to assign the order that colors were taken for each player:
#         NOTE must occur after the active board is correctly assigned:
#         """
#         new_game.active_placing_number, new_game.passive_placing_number = 1, 1
#
#         for _color, (_p1layer, _p2layer) in enumerate(zip(p1_order_layers, p2_order_layers)):
#             _p1order = round(_p1layer * 6)
#             _p2order = round(_p2layer * 6)
#
#             if _p1order > 0:
#                 new_game.active_color_order[_color] = _p1order
#
#                 # store the full color group:
#                 _i, _j = np.where(new_game.active_board == (_color + 6))
#                 new_game.active_color_groups[_color] = list(zip(_i, _j))
#                 new_game.active_placing_number += 1
#
#             if _p2order > 0:
#                 new_game.passive_color_order[_color] = _p2order
#
#                 # store the full color group:
#                 _i, _j = np.where(new_game.active_board == (_color + 12))
#                 new_game.passive_color_groups[_color] = list(zip(_i, _j))
#                 new_game.passive_placing_number += 1
#
#     @staticmethod
#     def assign_orth_flip_groups(new_game: Patterns):
#         """ assign the flipping and orthogonal groups so that the game is functional going forward
#         """
#         # populate the orthogonal groups and flipping groups from the color groups:
#         for _color in range(6):
#             active_group = new_game.active_color_groups[_color]
#             passive_group = new_game.passive_color_groups[_color]
#
#             active_orthogs = []
#             passive_orthogs = []
#
#             # find all the orthogonal locations then remove any that are flipped:
#             for _coords in active_group:
#                 active_orthogs.extend(list(orthogonal_neighbors[_coords]))
#
#             for _coords in passive_group:
#                 passive_orthogs.extend(list(orthogonal_neighbors[_coords]))
#
#             new_game.active_orthogonal_groups[_color] = set(active_orthogs) - new_game.flipped_locations
#             new_game.passive_orthogonal_groups[_color] = set(passive_orthogs) - new_game.flipped_locations
#
#             new_game.active_flipping_groups[_color] = set([_x for _x in new_game.active_orthogonal_groups[_color]
#                                                            if new_game.active_board[_x] == _color
#                                                            if _x not in new_game.passive_orthogonal_groups[_color]])
#
#             new_game.passive_flipping_groups[_color] = set([_x for _x in new_game.passive_orthogonal_groups[_color]
#                                                             if new_game.active_board[_x] == _color
#                                                             if _x not in new_game.active_orthogonal_groups[_color]])

#### using the old state system:
    # def _OLDcreate_game_from_state(self) -> Patterns:
    #     """ Use the tensor state to create the full patterns game
    #     """
    #     numpy_state = self.state.numpy()
    #
    #     # create game to assign attributes to:
    #     new_game = Patterns()
    #
    #     # assume that if we are validating, or wishing to create game from state, the game has started proper:
    #     new_game.is_game_started = True
    #     new_game.first_turn_passed = False
    #
    #     ##########################################
    #     ## set the bowl tokens for each player:
    #     _p1color = 0
    #     for _p1color, _is_color in enumerate(numpy_state[90:, 0, 0]):
    #         if _is_color == 1:
    #             break
    #
    #     new_game.active_bowl_token = _p1color
    #
    #     _p2color = 0
    #     for _p2color, _is_color in enumerate(numpy_state[96:, 0, 0]):
    #         if _is_color == 1:
    #             break
    #
    #     new_game.passive_bowl_token = _p2color
    #     ##########################################
    #
    #     ##########################################
    #     # populate the active and passive boards:
    #     new_game.active_board = np.zeros((8, 8), dtype=int)
    #     new_game.passive_board = np.zeros((8, 8), dtype=int)
    #
    #     vals, iind, jind = np.where(numpy_state[:18])
    #     new_game.active_board[iind, jind] = vals
    #     new_game.passive_board[iind, jind] = vals
    #
    #     # masks for the active and passive tokens:
    #     active_mask = np.logical_and(new_game.passive_board >= 6, new_game.passive_board < 12)
    #     passive_mask = new_game.passive_board >= 12
    #
    #     # the difference for the passive board and active board is simply that the piece representations
    #     # are 6-11 instead of 12-15 for the active, and reverse for passive:
    #     new_game.passive_board[active_mask] += 6
    #     new_game.passive_board[passive_mask] -= 6
    #
    #     # can use either board for this step:
    #     unflipped_i, unflipped_j = np.where(new_game.active_board < 6)
    #     unflipped_locations = set(list(zip(unflipped_i, unflipped_j)))
    #
    #     # subtract the unflipped locations from all the possible locations:
    #     new_game.flipped_locations = set(location_to_coordinates) - unflipped_locations
    #
    #     ##########################################
    #
    #     ##########################################
    #     # populate the color orders and active/ passive color groups:
    #     new_game.active_placing_number, new_game.passive_placing_number = 1, 1
    #
    #     for _ in range(18, 54):
    #         if numpy_state[_, 0, 0] == 1:
    #             color = (_ // 6) - 3
    #             new_game.active_color_order[color] = (_ % 6) + 1
    #
    #             # store the full color group:
    #             _i, _j = np.where(new_game.active_board == (color + 6))
    #             new_game.active_color_groups[color] = list(zip(_i, _j))
    #
    #             new_game.active_placing_number += 1
    #
    #     for _ in range(54, 90):
    #         if numpy_state[_, 0, 0] == 1:
    #             color = (_ // 6) - 9
    #             new_game.passive_color_order[color] = (_ % 6) + 1
    #
    #             # also store the full color group:
    #             _i, _j = np.where(new_game.active_board == (color + 12))
    #             new_game.passive_color_groups[color] = list(zip(_i, _j))
    #
    #             new_game.passive_placing_number += 1
    #
    #     ##########################################
    #     # populate the orthogonal groups and flipping groups from the color groups:
    #     for _color in range(6):
    #         active_group = new_game.active_color_groups[_color]
    #         passive_group = new_game.passive_color_groups[_color]
    #
    #         active_orthogs = []
    #         passive_orthogs = []
    #
    #         # find all the orthogonal locations then remove any that are flipped:
    #         for _coords in active_group:
    #             active_orthogs.extend(list(orthogonal_neighbors[_coords]))
    #
    #         for _coords in passive_group:
    #             passive_orthogs.extend(list(orthogonal_neighbors[_coords]))
    #
    #         new_game.active_orthogonal_groups[_color] = set(active_orthogs) - new_game.flipped_locations
    #         new_game.passive_orthogonal_groups[_color] = set(passive_orthogs) - new_game.flipped_locations
    #
    #         new_game.active_flipping_groups[_color] = set([_x for _x in new_game.active_orthogonal_groups[_color]
    #                                                        if new_game.active_board[_x] == _color
    #                                                        if _x not in new_game.passive_orthogonal_groups[_color]])
    #
    #         new_game.passive_flipping_groups[_color] = set([_x for _x in new_game.passive_orthogonal_groups[_color]
    #                                                    if new_game.active_board[_x] == _color
    #                                                    if _x not in new_game.active_orthogonal_groups[_color]])
    #
    #     ##########################################
    #
    #     # also populate the is no more placing characteristic:
    #     if numpy_state[-1, 0, 0] == 1:
    #         new_game._is_no_more_placing = True
    #
    #     else:
    #         new_game._is_no_more_placing = False
    #
    #     return new_game



#
#
# class Augmentor:
#     def __init__(self, game: Patterns):
#         self.game = game
#
#     def full_augment(self):
#         self.flip()
#         self.rotate()
#         self.permute_colors()
#
#     def flip(self):
#         """ randomly flip the board...
#         """
#         # 50% chance to flip the board:
#         if random.random() < 0.5:
#             return
#
#         self.game.active_board = np.fliplr(self.game.active_board)
#
#         # will also need to flip all the locations in the color groups to be able to plot:
#         for _color in range(6):
#             self.game.active_color_groups[_color] = [self.flip_coordinates(_x)
#                                                      for _x in self.game.active_color_groups[_color]]
#             self.game.passive_color_groups[_color] = [self.flip_coordinates(_x)
#                                                      for _x in self.game.passive_color_groups[_color]]
#
#     def rotate(self):
#         """ choose a random rotation from 4
#         """
#         _rand = random.random()
#
#         if _rand < 0.25:
#             return
#
#         if _rand < 0.5:
#             k = 1
#
#         elif _rand < 0.75:
#             k = 2
#
#         else:
#             k = 3
#
#         self.game.active_board = np.rot90(self.game.active_board, k=k)
#
#         for _color in range(6):
#             self.game.active_color_groups[_color] = [self.rotate_coordinates(_x, k)
#                                                      for _x in self.game.active_color_groups[_color]]
#             self.game.passive_color_groups[_color] = [self.rotate_coordinates(_x, k)
#                                                       for _x in self.game.passive_color_groups[_color]]
#
#     @staticmethod
#     def flip_coordinates(coordinates: tuple[int, int]) -> tuple[int, int]:
#         """ return the flipped coordinates. So (1, 3) -> (1, 4)
#         """
#         return coordinates[0], 7 - coordinates[1]
#
#     @staticmethod
#     def rotate_coordinates(coordinates: tuple[int, int], k: int) -> tuple[int, int]:
#         """ return the rotated coordinates by 90 degrees times k. (7-j, i)
#         """
#         tup = coordinates[0], coordinates[1]
#
#         for _ in range(k):
#             tup = 7 - tup[1], tup[0]
#
#         return tup
#
#     def permute_colors(self):
#         """ choose two colors to swap
#         """
#         if random.random() < 0.5:
#             return
#
#         c1, c2 = random.sample([0, 1, 2, 3, 4, 5], 2)
#
#         # swap all instances of the two colors:
#         self.game.active_color_order[c1], self.game.active_color_order[c2] = (
#             self.game.active_color_order[c2], self.game.active_color_order[c1])
#
#         self.game.passive_color_order[c1], self.game.passive_color_order[c2] = (
#             self.game.passive_color_order[c2], self.game.passive_color_order[c1]
#         )
#
#         # swap the bowl tokens if necessary:
#         if self.game.active_bowl_token == c1:
#             self.game.active_bowl_token = c2
#
#         if self.game.active_bowl_token == c2:
#             self.game.active_bowl_token = c1
#
#         if self.game.active_bowl_token == c1:
#             self.game.active_bowl_token = c2
#
#         if self.game.active_bowl_token == c2:
#             self.game.active_bowl_token = c1
#
#         c1_ind = self.game.active_board == c1
#         c2_ind = self.game.active_board == c2
#
#         self.game.active_board[c1_ind] = c2
#         self.game.active_board[c2_ind] = c1
#
#         # swap the color groups:
#         self.game.active_color_groups[c1], self.game.active_color_groups[c2] = (
#             self.game.active_color_groups[c2], self.game.active_color_groups[c1]
#         )
#
#         self.game.passive_color_groups[c1], self.game.passive_color_groups[c2] = (
#             self.game.passive_color_groups[c2], self.game.passive_color_groups[c1]
#         )
