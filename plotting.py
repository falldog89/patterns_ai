""" Plot the board provided in the game:


# C:\\Users\\Danie\\PycharmProjects\\patterns\\plotting_icons\\board
"""
import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches
import matplotlib.image as mpimg
import random

from game import Patterns
from int_to_board import location_to_coordinates


class PatternPlotter:
    def __init__(self,
                 game: Patterns,
                 fig_size: tuple[float, float]=(16, 8)) -> None:
        """ check thorughout for (x, y) versus (i, j)u
        """
        self.game = game
        self.fig_size = fig_size
        self.axis_xlimits = [-6.5, 13.5]
        self.axis_ylimits = [-4.5, 11.5]

        self.token_colors = [
            (0.9254901960784314, 0.6980392156862745, 0.4470588235294118),  # sand: 0
            (0.8901960784313725, 0.33725490196078434, 0.1568627450980392),  # red: 1
            (0.2549019607843137, 0.6941176470588235, 0.8392156862745098),  # light blue: 2
            (0.0196078431372549, 0.06274509803921569, 0.0392156862745098),  # black: 3
            (0.023529411764705882, 0.26666666666666666, 0.49019607843137253),  # dark blue: 4
            (0.050980392156862744, 0.44313725490196076, 0.047058823529411764),  # green: 5
            (0.7, 0.7, 0.7), # neutral colour before anything is taken.
        ]

        # centers for the bowls for each player:
        self.bowl_centers = [(-3, 7), (10, 0)]

        # where tokens start for each player:
        self.token_starts = [(4.5, 10), (-3.5, -2)]

    def get_star_token(self,
                       token_center: tuple[float, float],
                       num_points: int,
                       face_color: tuple[float, float, float]) -> patches.Patch:
        """ return the star token shaped patch
        """
        outrad = 0.38
        inrad = 0.31
        arc = 2. * np.pi / num_points

        patch_coords_1 = [
            [token_center[0] + outrad * np.cos(arc * _it),
             token_center[1] + outrad * np.sin(arc * _it)]
            for _it in range(num_points)
        ]

        patch_coords_2 = [
            [token_center[0] + inrad * np.cos((arc / 2) + arc * _it),
             token_center[1] + inrad * np.sin((arc / 2) + arc * _it)]
            for _it in range(num_points)
        ]

        patch_coords = []
        for p1, p2 in zip(patch_coords_1, patch_coords_2):
            patch_coords.append(p1)
            patch_coords.append(p2)

        return patches.Polygon(patch_coords,
                               alpha=0.8,
                               facecolor=face_color,
                               edgecolor=(0.0, 0.0, 0.0),
                               linewidth=0.5)

    def add_player_labels(self, axis: plt.axis) -> None:
        """ label the sides of the patterns board
        """
        # label each player:
        axis.text(self.bowl_centers[0][0], self.bowl_centers[0][1] + 2.7,
                'Player 1',
                fontsize=15.0,
                horizontalalignment='center',
                verticalalignment='center',
                )

        axis.text(self.bowl_centers[1][0], self.bowl_centers[1][1] + 2.7,
                  'Player 2',
                  fontsize=15.0,
                  horizontalalignment='center',
                  verticalalignment='center',
                  )

    def add_bowls(self, axis: plt.axis) -> None:
        """ draw the bowls for bowl tokens, and add those tokens if they have been chosen.
        Further, highlight the active players bowl with a golden outline:
        """
        # highlioht active player bowl, that is the player who will be playing next:
        active_center = self.bowl_centers[0] if self.game.active_player == 1 else self.bowl_centers[1]
        axis.add_patch(
            patches.Circle(
                active_center,
                radius=2.2,
                alpha=0.4,
                facecolor=(1.0, 0.8, 0)
            )
        )

        # draw the bowl:
        axis.add_patch(
            patches.Circle(self.bowl_centers[0],
                           radius=2.0,
                           alpha=0.5,
                           facecolor=(0.35, 0.3, 0.15))
        )

        axis.add_patch(
            patches.Circle(self.bowl_centers[1],
                           radius=2.0,
                           alpha=0.5,
                           facecolor=(0.35, 0.3, 0.15))
        )

        # assign the hand pieces to each player:
        p1_hand = self.game.active_bowl_token if self.game.active_player == 1 else self.game.passive_bowl_token
        p2_hand = self.game.active_bowl_token if self.game.active_player == -1 else self.game.passive_bowl_token

        # display the hand pieces for each player:
        if p1_hand is not None:
            random_center = (self.bowl_centers[0][0] - 0.5 + random.random(),
                             self.bowl_centers[0][1] - 0.5 + random.random())

            # player 1 hand patch:
            axis.add_patch(
                patches.Circle(random_center,
                               radius=0.5,
                               alpha=1.0,
                               facecolor=self.token_colors[p1_hand],
                               )
            )

        if p2_hand is not None:
            random_center = (self.bowl_centers[1][0] - 0.5 + random.random(),
                             self.bowl_centers[1][1] - 0.5 + random.random())

            # player 2 hand patch:
            axis.add_patch(
                patches.Circle(random_center,
                               radius=0.5,
                               alpha=1.0,
                               facecolor=self.token_colors[p2_hand],
                               )
                         )

    def draw_token_spaces(self, axis) -> None:
        """ the spaces in which the order tokens will sit
        """
        tholder_facecolor = (0.35, 0.3, 0.15)
        cholder_facecolors = [(0.5, 0.0, 0.5), (0.8, 0.8, 0.8)]
        vspaces = [[0.0, -1.2], [-1.2, 0.0]]

        # add locations for the color tokens taken and the order in which they are taken:
        for _hspace in range(6):
            for _ts, _vspace, _cfc in zip(self.token_starts, vspaces, cholder_facecolors):
                # the token holder:
                axis.add_patch(patches.Circle((_ts[0] + 1.2 * _hspace, _ts[1] + _vspace[0]),
                                              radius=0.5,
                                              alpha=0.4,
                                              facecolor=tholder_facecolor,
                                              edgecolor="black",
                                              linewidth=0.5)
                               )

                # the color token holder:
                axis.add_patch(patches.Circle((_ts[0] + 1.2 * _hspace, _ts[1] + _vspace[1]),
                                              radius=0.3,
                                              alpha=0.5,
                                              facecolor=_cfc,
                                              edgecolor="black",
                                              linewidth=0.5)
                               )

    def draw_board(self, axis: plt.axis) -> None:
        """ draw all the tokens, flipped or unflipped, on the board
        """
        # keep consistency by pointing at the relevant state vector:
        board = self.game.active_board if self.game.active_player == 1 else self.game.passive_board

        for _location in location_to_coordinates:
            _color = board[_location]
            face_color = self.token_colors[_color % 6]
            alpha = 0.3 if _color < 6 else 0.8
            tile_patch = patches.Circle((_location[1], 7 - _location[0]),
                                        radius=0.4,
                                        alpha=alpha,
                                        facecolor=face_color,
                                        edgecolor="black",
                                        linewidth=0.5)

            axis.add_patch(tile_patch)

    def add_order_tokens(self, axis: plt.axis) -> None:
        """ display the locations of the order tokens and color tokens which indicate the order in which
        the color groups were taken
        """
        # assign the color-order-taken to each player:
        p1_order = self.game.active_color_order if self.game.active_player == 1 else self.game.passive_color_order
        p2_order = self.game.active_color_order if self.game.active_player == -1 else self.game.passive_color_order
        player_orders = [p1_order, p2_order]

        # assign the color groups to each player:
        p1_color_groups = self.game.active_color_groups if self.game.active_player == 1 else self.game.passive_color_groups
        p2_color_groups = self.game.active_color_groups if self.game.active_player == -1 else self.game.passive_color_groups
        player_color_groups = [p1_color_groups, p2_color_groups]

        token_points = [8, 4]
        y_offsets = [[0.0, -1.2], [-1.2, 0.0]]
        face_colors = [(0.8, 0.1, 0.8), (0.8, 0.8, 0.9)]

        for porder, pcgroup, tstart, npoints, fcol, yoff in zip(player_orders, player_color_groups,
                                                                self.token_starts, token_points,
                                                                face_colors, y_offsets):
            # populate the ordering tiles for each player, from 1-6:
            untaken = 5

            for _color, _order in enumerate(porder):
                # if color group exists, place star token on board and indicate color order in token row:
                if _order > 0:
                    # add color token to token row:
                    color_center = (tstart[0] + 1.2 * (_order - 1), tstart[1] + yoff[0])
                    axis.add_patch(patches.Circle(color_center,
                                                  radius=0.35,
                                                  alpha=0.8,
                                                  facecolor=self.token_colors[_color],
                                                  edgecolor='black',
                                                  linewidth=0.5))

                    # place star token over the first color taken in the group:
                    token_coordsij = pcgroup[_color][0]
                    token_coords = (token_coordsij[1], 7 - token_coordsij[0])
                    token_string = str(_order)

                else:
                    token_coords = (tstart[0] + 1.2 * untaken, tstart[1] + yoff[1])
                    token_string = str(untaken + 1)
                    untaken -= 1

                star_patch = self.get_star_token(token_coords,
                                                 num_points=npoints,
                                                 face_color=fcol)
                axis.add_patch(star_patch)
                axis.text(token_coords[0], token_coords[1],
                          token_string,
                          fontsize=12.,
                          horizontalalignment='center',
                          verticalalignment='center', )

    def plot(self) -> None:
        """ plot the board according to the state:

        plot the player 1 and player 2 hand tiles somewhere for ease, as well as the colours they have taken so far!
        """
        fig, ax = plt.subplots(figsize=self.fig_size)

        self.add_player_labels(ax)
        self.add_bowls(ax)
        self.draw_token_spaces(ax)
        self.draw_board(ax)
        self.add_order_tokens(ax)

        ax.set_xlim(self.axis_xlimits)
        ax.set_ylim(self.axis_ylimits)
        ax.set_aspect('equal')

        ax.set_xticks([])
        ax.set_yticks([])
        plt.show()

#
# # todo correct this to make it work with the new board system in the patterns game:
# class PhotoPatternPlotter:
#     def __init__(self,
#                  game: Patterns,
#                  fig_size: tuple[float, float] = (16, 8)) -> None:
#         """ check thorughout for (x, y) versus (i, j)u
#         """
#         self.game = game
#         self.fig_size = fig_size
#
#         # for plotting on the photographic board:
#         self.top_left = (2440, 4700)  # centre of the first in the grid
#         self.top_right = (5800, 4760)
#         self.bott_left = (2410, 1210)
#         self.bott_right = (5790, 1310)
#
#         # initialize grid for plotting tiles themselves:
#         self.askew_grid = [[(None, None)] * 8 for _ in range(8)]
#
#         # grids for representing which colour was taken in order:
#         self.white_score_color_grid = [None] * 6
#         self.white_score_token_grid = [None] * 6
#         self.white_token_start = (7760, 5200)
#         self.white_token_end = (3890, 5170)
#         self.white_color_start = (7782, 5604)
#         self.white_color_end = (3864, 5604)
#
#         self.purple_score_color_grid = [None] * 6
#         self.purple_score_token_grid = [None] * 6
#         self.purple_token_start = (420, 800)
#         self.purple_token_end = (4330, 820)
#         self.purple_color_start = (435, 489)
#         self.purple_color_end = (4346, 407)
#
#         # bowl will be assigned the tile randomly. (center_x, center_y)
#         self.white_bowl = (1300, 4500)
#         self.purple_bowl = (6880, 1260)
#
#         self.bowl_radius = 1025.0
#         self.tile_radius = 160.
#         self.token_radius = 120.
#         self.color_radius = 120.
#
#         self.form_askew_grid()
#         self.form_counter_grids()
#
#         self.image_lims = [0, 0]
#
#     def form_counter_grids(self) -> None:
#         """ populate the coordinates for the counter grids and the token grids for
#         each player:
#         """
#         white_token_vec = (self.white_token_end[0] - self.white_token_start[0],
#                            self.white_token_end[1] - self.white_token_start[1])
#
#         white_color_vec = (self.white_color_end[0] - self.white_color_start[0],
#                            self.white_color_end[1] - self.white_color_start[1])
#
#         purple_token_vec = (self.purple_token_end[0] - self.purple_token_start[0],
#                             self.purple_token_end[1] - self.purple_token_start[1])
#
#         purple_color_vec = (self.purple_color_end[0] - self.purple_color_start[0],
#                             self.purple_color_end[1] - self.purple_color_start[1])
#
#         self.white_score_color_grid = [(self.white_color_start[0] + _i * white_color_vec[0] / 5,
#                                         self.white_color_start[1] + _i * white_color_vec[1] / 5)
#                                        for _i in range(6)]
#
#         self.white_score_token_grid = [(self.white_token_start[0] + _i * white_token_vec[0] / 5,
#                                         self.white_token_start[1] + _i * white_token_vec[1] / 5)
#                                        for _i in range(6)]
#
#         self.purple_score_color_grid = [(self.purple_color_start[0] + _i * purple_color_vec[0] / 5,
#                                          self.purple_color_start[1] + _i * purple_color_vec[1] / 5)
#                                         for _i in range(6)]
#
#         self.purple_score_token_grid = [(self.purple_token_start[0] + _i * purple_token_vec[0] / 5,
#                                          self.purple_token_start[1] + _i * purple_token_vec[1] / 5)
#                                         for _i in range(6)]
#
#     def form_askew_grid(self) -> None:
#         """ use the wonky lines to form a grid for the pieces, then
#         assign each location a coordinate. We do this with every placeable place,
#         except for the bowls which are random!
#         """
#         leftvec = (self.bott_left[0] - self.top_left[0], self.bott_left[1] - self.top_left[1])
#         rightvec = (self.bott_right[0] - self.top_right[0], self.bott_right[1] - self.top_right[1])
#
#         for _i in range(8):
#             # vertical start points for this row:
#             left_point = (self.top_left[0] + _i * leftvec[0] / 7., self.top_left[1] + _i * leftvec[1] / 7.)
#             right_point = (self.top_right[0] + _i * rightvec[0] / 7., self.top_right[1] + _i * rightvec[1] / 7.)
#
#             vec = (right_point[0] - left_point[0], right_point[1] - left_point[1])
#
#             for _j in range(8):
#                 self.askew_grid[_i][_j] = (left_point[0] + _j * vec[0] / 7., left_point[1] + _j * vec[1] / 7.)
#
#     def get_image_paths(self) -> None:
#         """ store all the paths to images:
#         """
#         white_token_paths = [os.path.join('plotting_icons', 'white_icons', f'{_num}.png') for _num in
#                              range(1, 7)]
#
#         self.white_token_images = self.get_images_from_paths(white_token_paths)
#
#         purple_token_paths = [os.path.join('plotting_icons', 'purple_icons', f'{_num}.png') for _num in
#                               range(1, 7)]
#
#         self.purple_token_images = self.get_images_from_paths(purple_token_paths)
#
#         colors = ['yellow', 'orange', 'light_blue', 'black', 'blue', 'green']
#
#         flipped_paths = [os.path.join('plotting_icons', 'color_token_markers', f'{_color}.png') for _color in
#                          colors]
#         self.flipped_images = self.get_images_from_paths(flipped_paths)
#
#         unflipped_paths = [os.path.join('plotting_icons', 'unflipped_tokens', f'{_color}.png') for _color in
#                            colors]
#
#         self.unflipped_images = self.get_images_from_paths(unflipped_paths)
#
#     @staticmethod
#     def get_images_from_paths(paths: list[str]) -> list:
#         """ return the images from the paths provided:
#         """
#         return [mpimg.imread(_path) for _path in paths]
#
#     @staticmethod
#     def get_extent(center: tuple[float, float], radius: float) -> list[float]:
#         return [center[0] - radius, center[0] + radius, center[1] - radius, center[1] + radius]
#
#     def plot_actual_patterns_board(self) -> None:
#         """
#         :return:
#         """
#         self.get_image_paths()
#         fig, ax = plt.subplots(figsize=self.fig_size)
#
#         self.plot_background_board(ax)
#         self.plot_tiles(ax)
#         self.plot_tokens(ax)
#         self.plot_bowl_tiles(ax)
#
#         ax.set_xlim([0., self.image_lims[1]])
#         ax.set_ylim([0., self.image_lims[0]])
#
#         plt.show()
#
#     def plot_background_board(self, ax: plt.axes) -> None:
#         """ plot the image of the board itself
#         """
#         # first plot the background board image:
#         board_path = os.path.join('plotting_icons', 'board', 'full_board_merged.png')
#         board_image = mpimg.imread(board_path)
#         ax.imshow(board_image, extent=[0, board_image.shape[1], 0, board_image.shape[0]])
#
#         self.image_lims = board_image.shape
#
#     def plot_tiles(self, ax) -> None:
#         """ plot the flipped or unflipped tiles
#         """
#         # iterate over state to populate the board:
#         for _loc, _tile in enumerate(self.game.active_state[:52]):
#             color = _tile % 6
#             _i, _j = list_vec[_loc]
#             tile_extent = self.get_extent(self.askew_grid[_i][_j], self.tile_radius)
#
#             # state is 52 locations, plus 12 for the color taken by each player in order, plus hand piece:
#             if _tile < 6:
#                 im = self.unflipped_images[color]
#
#             else:
#                 im = self.flipped_images[color]
#
#             ax.imshow(im, extent=tile_extent)
#
#     def plot_token(self, ax: plt.axes, color: int, order: int) -> None:
#         """ color here is between 0 and 11, and comes from self.game.state.
#         Order is the order at which that color was taken
#         """
#         # determine the player from the color provided:
#         player = 0 if (color < 6) else 1
#
#         # point to the correct player images;
#         token_images = self.white_token_images if player == 0 else self.purple_token_images
#
#         # grid locations on the board image for the token and color indicator:
#         color_grid = self.white_score_color_grid if player == 0 else self.purple_score_color_grid
#
#         # get the location of the first member of the color group taken:
#         color_groups = self.game.active_color_groups if color < 6 else self.game.passive_color_groups
#         location = color_groups[color % 6][0]
#         i, j = list_vec[location]
#         tile_center = self.askew_grid[i][j]
#
#         # determine the extents of each image:
#         token_extent = self.get_extent(tile_center, self.token_radius)
#         color_extent = self.get_extent(color_grid[order], self.color_radius)
#
#         # plot each image on the axis:
#         ax.imshow(self.flipped_images[color % 6], extent=color_extent)
#         ax.imshow(token_images[order], extent=token_extent)
#
#     def plot_untaken_token(self, axis: plt.axes, token_number: int, player: int):
#         """ plot the untaken token indicated for the relevant player
#         """
#         # get the relevant tokens and grid center locations:
#         token_images = self.white_token_images if player == 0 else self.purple_token_images
#         token_grid = self.white_score_token_grid if player == 0 else self.purple_score_token_grid
#         token_extent = self.get_extent(token_grid[token_number], self.token_radius)
#
#         axis.imshow(token_images[token_number], extent=token_extent)
#
#     def plot_tokens(self, ax: plt.axes) -> None:
#         """ plot the white and purple tokens that have been taken, on top of the tiles,
#         and plot the colours of those taken so far
#         """
#         untaken = 5
#
#         for _color, _order in enumerate(self.game.active_state[52:64]):
#             if _order > 0:
#                 self.plot_token(ax, _color, _order - 1)
#
#             else:
#                 self.plot_untaken_token(axis=ax, token_number=untaken, player=_color // 6)
#                 untaken -= 1
#
#             # at 5, reset untaken and move on to purple player:
#             if _color == 5:
#                 untaken = 5
#
#     def plot_bowl_tiles(self, ax: plt.axes) -> None:
#         """ plot the bowl token in a semi random location
#         """
#         if self.game.active_state[64] == 6:
#             return
#
#         white_im = self.flipped_images[self.game.active_state[64]]
#         purple_im = self.flipped_images[self.game.active_state[65]]
#
#         white_extent = self.get_extent(self.white_bowl, self.tile_radius)
#         purple_extent = self.get_extent(self.purple_bowl, self.tile_radius)
#
#         ax.imshow(white_im, extent=white_extent)
#         ax.imshow(purple_im, extent=purple_extent)
