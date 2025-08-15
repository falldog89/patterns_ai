""" Plot the patterns game, to show:

- The board, with flipped and unflipped color tokens,
- The order tokens for each player, indicating who owns which color groups, and in what order,
- The token that each player is able to place from their hand,
- Player information, including score, current active player, and name
- Possible legal moves for the active player

todo
1. scaling for text
3. split off the perim and shield code into other design files
4. check for final magic constants, tidy up code

Nomenclature:

Bowl: the bowl that bowl tokens sit in, 1 per player
bowl token: the token as it appears inside the bowl

color group: flipped tokens on the board
group patch: additional color to highlight the polygon that the color group represents

order_tokens: numbered tokens, different for each player, whether they be on the board or on the token holder spots

order_token_holders: the empty, labelled token spots that are revealed when the tokens are placed on the board
color_token_holders: the empty, small circles that will take minitokens

color_minitokens: the tiny sized color tokens used to highlight which color was taken in which order:
flipped_tokens: the board tokens that represent colors that belong to a taken color group
unflipped_tokens: the board tokens that are yet to be claimed.

"""
from collections import Counter

import matplotlib.pyplot as plt
import matplotlib.path as mpath

import numpy as np
from matplotlib import patches

from patterns.game import BlobComet
from patterns.game import Gem
from patterns.game import Patterns
from patterns.constants import set_location_to_coordinates
from patterns.constants import location_to_coordinates

from typing import Optional


class Plotter:
    def __init__(self, game: Patterns) -> None:
        """ Here we define a bunch of bespoke config that nicely arranges and displays the board game
        of Patterns.
        """
        self.game = game

        ### FIGURE params:
        self.set_fsize = 10.0

        # the limits of the display:
        self.axis_xlimits = [-7.0, 7.0]
        self.axis_ylimits = [-7.0, 7.0]

        ### General color scheme for the game, defining the 6 colors that will be used for the different color groups:
        global_colors = [self._color_rgb(_color) for _color in [(255, 173, 173),  # sand: 0
                                                                (253, 255, 182),  # red: 1
                                                                (202, 255, 191),  # steel blue: 2
                                                                (155, 246, 255),  # teal green: 3
                                                                (160, 196, 255),  # dark blue: 4
                                                                (223, 235, 235),  # green: 5
        ]]

        ### Player information, inside player shields, with score and bowl tokens:

        ### BOWLS: the holder for the bowl token for each player:
        self.bowl_config = {
            'token_locations': [(-4.75, 3.8), (4.75, -3.8)],
            'locations': [(-4.6, 4.6), (4.6, -4.6)],
            'holder_color': self._color_rgb( (232, 232, 232) ),
            'edgecolor': (125, 90, 50),
            'holder_linewidth': 0.5,
            'holder_radius': 0.55,
        }

        self.cross_config = {
            'offset_mults': [0.9, 0.8],
            'colors': [self._color_rgb((168, 25, 7)), self._color_rgb((48, 7, 163))],
            'radius1s': [[0.03, 0.06], [0.08, 0.09]],
            'radius2s': [[0.08, 0.1], [0.08, 0.07]],
            'clockwise_angles': [[0.3 + np.pi / 4.0, -0.3 + 3.0 * np.pi / 4.0],
                                 [0.1 + np.pi / 4.0, -0.3 + 3.0 * np.pi / 4.0]],
            'anticlockwise_angles': [[0.2 + np.pi / 4.0, -0.2 + 3.0 * np.pi / 4.0],
                                     [0.2 + np.pi / 4.0, -0.2 + 3.0 * np.pi / 4.0]],
            'linewidth': 0.0,
        }

        # scalables = {self.bowl_config: ['holder_linewidth', 'no_more_placing_size'], ]


        ### Player shields:
        self.shield_config = {
            'colors': [self._color_rgb((245, 222, 76)), self._color_rgb((210, 157, 245))],
            'locations': [(-6.5, 6.5), (3.0, -6.5)],
            'width': 3.5,
            'height': 4.2,
            'alphas': [1.0, 1.0],
            'border_colors': [self._color_rgb((255, 150, 79)), self._color_rgb((200, 200, 200))],
            'interior_offset': 0.275,
            'interior_alpha': 0.5,
            'linewidth': 0.2,# 0.25,
            'border_corner_curve': 0.1,
            'interior_corner_curve': 0.1,
            'interior_height_offset': 0.03,
        }

        ### Player scores and gems:
        self.player_score_config = {
            'locations': [(-4.75, 4.8), (4.75, -4.8)],
            'gem_scales': [0.4, 0.4], # ability to set different scale for active and passive gems
            'gem_colors': [['yellow_bright', 'yellow_dark'], ['purple_bright', 'purple_dark']],
            'gem_locations': [[(-3.9, 4.75), (-5.6, 4.75)], [(3.9, -4.85), (5.6, -4.85)]],
            'gem_linewidth': 0.0,
            'gem_aspect': (1.0, 0.9),
            'fontname': 'MingLiU-ExtB',
            'fontsize': 18.0,
            'fontweight': 400.0,
        }

        ### Player labels and appearance:
        self.player_label_config = {
            'labels': ['Player 1', 'Player 2'],
            'fontname': 'MingLiU-ExtB',
            'fontsize': 20.0,
            'fontweight': 500.0,
            'locations': [(-4.7, 5.65), (4.8, -5.65)],
        }

        bgcols = [self._color_rgb(_color) for _color in [ (139, 193, 223), (246, 217, 207),(254, 242, 212), #(139, 193, 223)
                                                          (179, 155, 160)]]#(244, 251, 243), ]]

        ### light decorative details for the background:
        self.background_config = {
            'circle_colors': bgcols,
            'radii': [6.2, 5.6, 5.1, 4.9],
            'linewidths': [0.0, 0.0, 0.0, 0.0],
            'background_color': self._color_rgb( (100, 100, 100)),#(179, 155, 160)),#(254, 242, 212)),
            'origin': (0.0, 0.0),
        }

        ### Order token holders:
        self.order_token_holder_config = {
            'locations': [ [(-1.75 + 1.45 * _x, 6.0) for _x in range(6)],
                           [(1.75 - 1.45 * _x, -6.0) for _x in range(6)]],
            'scale_multiplier': 1.5, #1.4,
            'linewidth_multiplier': 0.0,
            'alpha_multiplier': 0.6,

            'text_config':
                {
                    'fontsize': 15.0,
                    'fontweight': 50.0,
                    'color': (0.0, 0.0, 0.0),
                    'fontname': 'Palatino Linotype',
                    'alpha': 0.4,
                }
        }

        ### Mini token holder:
        self.mini_holder_config = {
            'radius': 0.15,
            'locations': [ [(-1.6 + 1.5 * _x, 5.0) for _x in range(6)],
                           [(1.6 - 1.5 * _x, -5.0) for _x in range(6)]],
            'facecolor': self._color_rgb( (227, 227, 222) ),
            'linewidth': 0.5,
            'edgecolor': self._color_rgb( (145, 145, 140) ),
        }

        ### (Main) TOKENS:
        self.token_config = {
            'radius_flipped': 0.44,
            'radius_unflipped': 0.25,
            'radius_unflipped_white': 0.01,#0.09,
            'radius_mini': 0.25,
            'radius_legal': 0.3,

            'edgecolor_flipped': self._color_rgb( (50, 50, 50) ),
            'edgecolor_unflipped': self._color_rgb( (50, 50, 50) ),
            'middle_color_unflipped': self._color_rgb( (255, 255, 255) ),
            'edgecolor_mini': self._color_rgb( (20, 20, 20) ),

            'linewidth_flipped': 0.0,#0.4,
            'linewidth_unflipped': 0.0,#0.3,
            'linewidth_mini': 0.5,

            'unflipped_colors': global_colors,
            'flipped_colors': global_colors,

            'legal_move_color': self._color_rgb( (255, 150, 79) ),
            'legal_move_linewidth': 2.4,

            'middle_color_alpha': 0.6,
        }

        ### ORDER TOKENS:
        self.order_token_config = {
            'fontsize': 10.0,
            'fontweight': 30.0,
            'numerals': {0: '1', 1: '2', 2: '3', 3: '4', 4: '5', 5: '6'},

            # fiddly constants that align the numeral with the center "better":
            'numeral_offsets': [(-0.015, -0.015), (-0.005, -0.02), (-0.005, -0.02),
                                (-0.01, -0.02), (-0.01, -0.02), (-0.015, -0.02)],

            'random_rotation_scale': 0.2,
            'rotations': [0.0, -np.pi / 10.0],

            # parameters to control order tokens and their holders:
            'num_petals': [7, 5],
            'extents_inner': [0.24, -0.24],
            'extents_outer': [0.25, -0.25],
            'scales_inner': [0.24, 0.27], #[0.20, 0.27],
            'scales_outer': [0.32, 0.38], #[0.28, 0.38]
            'alpha_inner': 0.5,
            'alpha_outer': 1.0,
            'color_inner': (1.0, 1.0, 1.0),
            'colors': [self._color_rgb( (250, 235, 135) ), self._color_rgb( (213, 187, 250) )],
            'linewidths_inner': [0.5, 1.5],
            'linewidths_outer': [1.5, 1.5],
            'edgecolor': (0, 0, 0),

            'text_config':
                {
                    'fontsize': 10.0,
                    'fontweight': 50.0,
                    'color': (0.0, 0.0, 0.0),
                    'fontname': 'Palatino Linotype',
            },
        }

        ### COLORGROUPS:
        self.colorgroup_config = {
            'offset': 0.32,
            'linewidth': 0.0,
            'edgecolor': (0.0, 0.0, 0.0),
        }

    @staticmethod
    def _color_rgb(color: tuple[int, int, int]) -> tuple[float, float, float]:
        return color[0] / 255.0, color[1] / 255.0, color[2] / 255.0

    def plot(self, fig_size: float = 10) -> None:
        """ plot the board according to the state:

        Just choose fig size as a single float, as the plot will always be square:
        """
        self.assert_scale_attributes(fig_size)
        self.determine_active_player()

        fig, ax = plt.subplots(figsize=(fig_size, fig_size) )

        # Introduce background elements first:
        self.add_background(ax)

        # draw the player information areas, with shields, hand tokens, scores, and gems:
        self.draw_player_shields(ax)
        self.draw_player_bowls(ax)
        self.add_player_scores(ax)
        self.add_player_gems(ax)
        self.add_player_labels(ax)

        # draw flipped and unflipped board tokens, and add the order tokens:
        self.draw_board(ax)
        self.add_order_tokens(ax)

        # final tinker to the plot axes:
        self.configure_axes(ax)

    def assert_scale_attributes(self, scale: float) -> None:
        """ check whether the requested plot size is different from the established one,
        and if so scale all attributes
        """
        if scale == self.set_fsize:
            return

        # the multiplier that all scalable fields will receive:
        scale_ratio = scale/ self.set_fsize

        # ensure that repeated calls will not continue to shrink things:
        self.set_fsize = scale

        # Scale all attributes that require scaling as fig size shifts:
        self.scale_scalable_attributes(scale_ratio)

    def scale_scalable_attributes(self, scale_ratio: float) -> None:
        """ go into the relevant attributes and scale them according to the scale factor
        """

        self.player_label_config['fontsize'] *= scale_ratio
        self.player_label_config['fontweight'] = min(self.player_label_config['fontweight']
                                                        * scale_ratio, 1000.0)

        self.colorgroup_config['linewidth'] *= scale_ratio

        self.token_config['linewidth_unflipped'] *= scale_ratio
        self.token_config['linewidth_mini'] *= scale_ratio
        self.token_config['linewidth_flipped'] *= scale_ratio

        # self.order_token_holder_constants['fontsize'] *= scale_ratio
        # self.order_token_holder_constants['linewidth'] *= scale_ratio
        # self.order_token_holder_constants['fontweight'] = min(self.order_token_holder_constants['fontweight']
        #                                                       * scale_ratio, 1000.0)


        self.order_token_config['fontsize'] *= scale_ratio
        self.order_token_config['fontweight'] = min(self.order_token_config['fontweight']
                                                       * scale_ratio, 1000.0)

        # self.order_token_constants['flower_configs'][0]['linewidth'] *= scale_ratio
        # self.order_token_constants['flower_configs'][1]['linewidth'] *= scale_ratio
        # self.order_token_constants['flower_configs'][0]['fontsize'] *= scale_ratio
        # self.order_token_constants['flower_configs'][1]['fontsize'] *= scale_ratio

        # self.order_token_constants['flower_configs'][0]['fontweight'] = min(self.order_token_constants['flower_configs'][0]
        #                                                              ['fontweight'] * scale_ratio, 1000.0)
        # self.order_token_constants['flower_configs'][1]['fontweight'] = min(self.order_token_constants['flower_configs'][1]
        #                                                               ['fontweight'] * scale_ratio, 1000.0)

    def determine_active_player(self):
        """ handy selection tool, as a lot of configs access the first entry for active, second for passive
        """
        self.player_select = (0, 1) if self.game.active_player == 1 else (1, 0)

    def add_background(self, axis: plt.axis) -> None:
        """
        add teardrops and other geometric shapes to make the board look a little nicer:
        """
        # first add in the background color for the square patch:
        axis.add_patch(patches.Rectangle((self.axis_xlimits[0], self.axis_ylimits[0]),
                                         width=self.axis_xlimits[1] - self.axis_xlimits[0],
                                         height=self.axis_ylimits[1] - self.axis_ylimits[0],
                                         facecolor=self.background_config['background_color'],
                                         )
                       )

        for _color, _radius, _linewidth in zip(self.background_config['circle_colors'],
                                               self.background_config['radii'],
                                               self.background_config['linewidths']):
            circle_patch = patches.Circle(self.background_config['origin'],
                                          radius=_radius,
                                          facecolor=_color,
                                          linewidth=_linewidth,
                                          edgecolor='black')

            axis.add_patch(circle_patch)

    def add_player_scores(self, axis: plt.axis) -> None:
        """ keep track of guaranteed score for each player ie score of flipped tokens
        """
        scores = self.game.calculate_score()
        for _player, _score in enumerate(scores):
            axis.text(self.player_score_config['locations'][_player][0],
                      self.player_score_config['locations'][_player][1],
                      str(_score),
                      fontname=self.player_score_config['fontname'],
                      fontsize=self.player_score_config['fontsize'],
                      horizontalalignment='center', verticalalignment='center',
                      )

    def add_player_labels(self, axis: plt.axis) -> None:
        """ label the active and passive players:
        """
        for (_x, _y), _text in zip(self.player_label_config['locations'],
                                   self.player_label_config['labels']):
            axis.text(_x, _y, _text,
                fontsize=self.player_label_config['fontsize'],
                horizontalalignment='center', verticalalignment='center',
                fontname=self.player_label_config['fontname'],
                fontweight=self.player_label_config['fontweight']
            )

    def add_player_gems(self, axis: plt.axis) -> None:
        """ Draw gems either side of the player score:
        """
        for _locations, _colors, _select in zip(self.player_score_config['gem_locations'],
                                                self.player_score_config['gem_colors'],
                                                self.player_select):
            # gems either side of the score:
            for _location in _locations:
                Gem().gem_3(_colors[_select], axis, _location,
                            aspect=self.player_score_config['gem_aspect'],
                            scale=self.player_score_config['gem_scales'][_select],
                            rotation=0.0,
                            linewidth=self.player_score_config['gem_linewidth'],
                            )

    def draw_player_shields(self, axis: plt.axis) -> None:
        """ Draw the shields, add the player labels, gems, scores and bowl tokens:
        """
        for _it, (_select, _flip) in enumerate(zip(self.player_select, [False, True])):
            # add the shields to the axis:
            self.shield(axis,
                        self.shield_config['locations'][_it],
                        width=self.shield_config['width'],
                        height=self.shield_config['height'],
                        interior_offset=self.shield_config['interior_offset'],
                        interior_facecolor=self.shield_config['colors'][_it],
                        interior_alpha=self.shield_config['interior_alpha'],
                        border_facecolor=self.shield_config['border_colors'][_select],
                        border_alpha=self.shield_config['alphas'][_select],
                        linewidth=self.shield_config['linewidth'],
                        border_corner_curve=self.shield_config['border_corner_curve'],
                        interior_corner_curve=self.shield_config['interior_corner_curve'],
                        interior_height_offset=self.shield_config['interior_height_offset'],
                        is_flipped=_flip,
            )

    def draw_player_bowls(self, axis: plt.axis) -> None:
        """ draw the hand tokens, the holding spot, and indicate whether placing is allowed."""
        # assign the hand pieces to each player:
        bowl_tokens = [self.game.active_bowl_token, self.game.passive_bowl_token]

        for (cx, cy), _select in zip(self.bowl_config['token_locations'], self.player_select):
            axis.add_patch(patches.Circle( (cx, cy),
                                          radius=self.bowl_config['holder_radius'],
                                          facecolor=self.shield_config['border_colors'][_select],
                                          edgecolor=self.token_config['edgecolor_flipped'],
                                          linewidth=self.bowl_config['holder_linewidth'],
                                          )
                           )

            axis.add_patch(patches.Circle((cx, cy),
                                          radius=self.token_config['radius_flipped'],
                                          facecolor=self.token_config['flipped_colors'][bowl_tokens[_select]],
                                          edgecolor=self.token_config['edgecolor_flipped'],
                                          linewidth=self.token_config['linewidth_flipped'],
                                          )
                           )

        if self.game.is_no_more_placing:
            self.add_crosses(axis)

    def add_crosses(self, axis: plt.axis) -> None:
        """ draw blobby crosses to indicate no more placing:
        """
        for (cx, cy), r1s, r2s, cw_angles, acw_angles, _col in zip(self.bowl_config['token_locations'],
                                                                   self.cross_config['radius1s'],
                                                                   self.cross_config['radius2s'],
                                                                   self.cross_config['clockwise_angles'],
                                                                   self.cross_config['anticlockwise_angles'],
                                                                   self.cross_config['colors']
                                                                   ):
            _patch1 = BlobComet().get_blobcomet_patch(
                        center_1 = (cx - self.cross_config['offset_mults'][0] * self.bowl_config['holder_radius'],
                                    cy - self.cross_config['offset_mults'][0] * self.bowl_config['holder_radius']),
                        center_2 = (cx + self.cross_config['offset_mults'][0] * self.bowl_config['holder_radius'],
                                    cy + self.cross_config['offset_mults'][0] * self.bowl_config['holder_radius']),
                        radius_1=r1s[0], radius_2=r2s[0],
                        cw_angle=cw_angles[0], acw_angle=acw_angles[0],
                        color=_col,
                        linewidth=self.cross_config['linewidth'],
                    )
            _patch2 = BlobComet().get_blobcomet_patch(
                center_1=(cx + self.cross_config['offset_mults'][1] * self.bowl_config['holder_radius'],
                          cy - self.cross_config['offset_mults'][1] * self.bowl_config['holder_radius']),
                center_2=(cx - self.cross_config['offset_mults'][1] * self.bowl_config['holder_radius'],
                          cy + self.cross_config['offset_mults'][1] * self.bowl_config['holder_radius']),
                radius_1=r1s[1], radius_2=r2s[1],
                cw_angle=cw_angles[1], acw_angle=acw_angles[1],
                color=_col,
                linewidth=self.cross_config['linewidth'],
            )

            axis.add_patch(_patch1)
            axis.add_patch(_patch2)

    def draw_board(self, axis: plt.axis) -> None:
        """ draw all the tokens, flipped or unflipped, on the board
        """
        self.draw_unflipped_tokens(axis)
        self.draw_flipped_tokens(axis)

    def draw_flipped_tokens(self, axis: plt.axis) -> None:
        """ draw and populate the color groups for each player:
        """
        color_groups = [self.game.active_color_groups, self.game.passive_color_groups]

        # First, iterate over the color groups:
        for color_group in color_groups:
            for _color, _locations in color_group.items():
                if len(_locations) > 0:
                    # get the perimeter patch for the whole color group in one object:
                    perimeter_patch = self.get_colorgroup_patch(_locations)
                    self.set_patch_attributes(perimeter_patch, _color)
                    axis.add_patch(perimeter_patch)

                    # also draw the tokens for the color group:
                    face_color = self.token_config['flipped_colors'][_color]

                    for _i, _j in _locations:
                        tokx, toky = self._convert_ij_xy((_i, _j))
                        _token = patches.Circle((tokx, toky),
                                                    radius=self.token_config['radius_flipped'],
                                                    facecolor=face_color,
                                                    edgecolor=self.token_config['edgecolor_flipped'],
                                                    linewidth=self.token_config['linewidth_flipped'],
                                                    )

                        axis.add_patch(_token)

    def set_patch_attributes(self, patch: patches.Patch, color: int):
        """ change the attributes to match global settings
        """
        # set parameters for the group patch:
        patch.set_facecolor(self.token_config['flipped_colors'][color])
        patch.set_edgecolor(self.colorgroup_config['edgecolor'])
        patch.set_linewidth(self.colorgroup_config['linewidth'])

    def draw_unflipped_tokens(self,axis: plt.axis) -> None:
        """ different style and drawing for the unflipped locations

        NEW LOGIC

        - if color group of bowl token not taken, only plot little crosses on where you cannot place
        the tile. Plot tiles as normal,

        - if color group IS taken, assign an arrow from the color group to all tiles that can be taken!
            plot all other tiles as normal

        - if a flip is ever possible, indicate with curved arrow inside the tile.
        """

        self.arrow_config = {'colors': [self._color_rgb(_col) for _col in [
            (227, 80, 64), # red
            (245, 157, 5), # yellow
            (55, 191, 108), # green
            (29, 141, 245), # turq
            (16, 64, 235),  # dark blue
            (64, 66, 74), # gray
        ]]}
        # first iterate over and plot all unflipped tiles normally
        for _ij in (set_location_to_coordinates - self.game.flipped_locations):
            tokx, toky = self._convert_ij_xy(_ij)
            _color = self.game.active_board[_ij]
            axis.add_patch(patches.Circle((tokx, toky),
                                          facecolor=self.token_config['unflipped_colors'][_color],
                                          radius=self.token_config['radius_unflipped'],
                                          linewidth=self.token_config['linewidth_unflipped'],
                                          edgecolor=self.token_config['edgecolor_unflipped'],
                                          )
                           )

        # now iterate over placing and flipping locations as necessary:
        game_actions = self.game.get_actions()
        flip_locs = set(location_to_coordinates[_action % 52]
                        for _action in game_actions if 52 <= _action < 104)

        place_locs = set(location_to_coordinates[_action % 52]
                         for _action in game_actions if _action < 52)

        # if the color group is taken:
        if self.game.active_color_order[self.game.active_bowl_token] > 0:
            cgroup = self.game.active_color_groups[self.game.active_bowl_token]
            for _i, _j in place_locs:
                rotation = 0.0
                offx, offy = 0.0, 0.0
                tokx, toky = self._convert_ij_xy((_i, _j))
                osize = 0.3

                # down:
                if (_i - 1, _j) in cgroup:
                    rotation = 0.0
                    offy = osize

                # up:
                elif (_i + 1, _j) in cgroup:
                    rotation = 180.0
                    offy = -osize

                # right:
                elif (_i, _j - 1) in cgroup:
                    rotation = 90.0
                    offx = -osize

                # left:
                elif (_i, _j + 1) in cgroup:
                    rotation = 270.0
                    offx = osize

                # now we know which direction to point in with the arrow!
                arrow_patch = self.get_placing_arrow_patch(
                    point=(tokx + offx, toky + offy),
                    rotation=rotation,
                    head_width = 0.25,
                    head_height = 0.17,
                    body_width = 0.1,
                    body_height = 0.9,
                    color=self.token_config['flipped_colors'][self.game.active_bowl_token],#(0.0, 0.0, 0.0),
                    )
                axis.add_patch(arrow_patch)

        # if the color group is not taken, add little crosses over illegal moves!
        else:
            # first find what those illegal moves are:
            illegal_locations = set_location_to_coordinates - self.game.flipped_locations - place_locs
            cw_angles = self.cross_config['clockwise_angles'][0]
            acw_angles = self.cross_config['anticlockwise_angles'][0]
            _col = self.cross_config['colors'][0]

            for _i, _j in illegal_locations:
                tokx, toky = self._convert_ij_xy((_i, _j))
                off = 0.2
                bo = 0.0 if self.game.active_player == 1 else  -0.0
                _patch1 = BlobComet().get_blobcomet_patch(
                    center_1=(tokx - off * self.token_config['radius_unflipped'] - bo,
                              toky - off * self.token_config['radius_unflipped'] + bo),
                    center_2=(tokx + off * self.token_config['radius_unflipped'] - bo,
                              toky + off * self.token_config['radius_unflipped'] + bo),
                    radius_1=0.02, radius_2=0.02,
                    cw_angle=cw_angles[0], acw_angle=acw_angles[0],
                    color=_col,
                    linewidth=self.cross_config['linewidth'],
                )

                _patch2 = BlobComet().get_blobcomet_patch(
                    center_1=(tokx + off * self.token_config['radius_unflipped'] - bo,
                              toky - off * self.token_config['radius_unflipped'] + bo),
                    center_2=(tokx - off * self.token_config['radius_unflipped'] - bo,
                              toky + off * self.token_config['radius_unflipped'] + bo),
                    radius_1=0.02, radius_2=0.02,
                    cw_angle=cw_angles[1], acw_angle=acw_angles[1],
                    color=_col,
                    linewidth=self.cross_config['linewidth'],
                )

                axis.add_patch(_patch1)
                axis.add_patch(_patch2)

        # finally, add in the rotation symbol for flippers: NOTE that we do not consider flipping actions
        # when they occur in both placing and flipping form.
        for _ij in flip_locs:
            tokx, toky = self._convert_ij_xy(_ij)

            w1 = 0.68
            w2 = 0.91
            toff = 0.25
            _patch1 = BlobComet().get_blobcomet_patch(
                    center_1=(tokx - w1 * self.token_config['radius_unflipped'], toky),
                    center_2=(tokx + w2 * self.token_config['radius_unflipped'], toky),
                    radius_1=(1.0 - w1) * self.token_config['radius_unflipped'],
                    radius_2=(1.0 - w2) * self.token_config['radius_unflipped'],
                    cw_angle=np.pi / 2.0, acw_angle=np.pi /2.0 + toff,
                    color=self.arrow_config['colors'][self.game.active_board[_ij]],
                    linewidth=0.0,
                )

            _patch2 = BlobComet().get_blobcomet_patch(
                center_1=(tokx + w1 * self.token_config['radius_unflipped'], toky),
                center_2=(tokx - w2 * self.token_config['radius_unflipped'], toky),
                radius_1=(1.0 - w1) * self.token_config['radius_unflipped'],
                radius_2=(1.0 - w2) * self.token_config['radius_unflipped'],
                cw_angle=-np.pi / 2.0, acw_angle=-np.pi / 2.0 + toff,
                color=self.arrow_config['colors'][self.game.active_board[_ij]],
                linewidth=0.0,
            )

            axis.add_patch(_patch1)
            axis.add_patch(_patch2)

    def draw_order_token(self,
                         axis: plt.axis,
                         location: tuple[float, float],
                         player: int,
                         number: int,
                         is_holder: bool,
                         is_rotation: bool,
                         inner_color: Optional[tuple[float, float, float]]=None,
                   ) -> None:
        """ A token is two flowers overlaid with the given player config, placed at a given location,
        at a given size to represent the token or the token holder, and with a given numeral
        """

        text_config = (self.order_token_holder_config['text_config']
                       if is_holder else self.order_token_config['text_config'])

        inner_scale = self.order_token_config['scales_inner'][player]
        outer_scale = self.order_token_config['scales_outer'][player]

        inner_alpha = self.order_token_config['alpha_inner']
        outer_alpha = self.order_token_config['alpha_outer']

        inner_linewidth = self.order_token_config['linewidths_inner'][player]
        outer_linewidth = self.order_token_config['linewidths_outer'][player]

        if is_holder:
            inner_scale *= self.order_token_holder_config['scale_multiplier']
            outer_scale *= self.order_token_holder_config['scale_multiplier']

            inner_alpha *= self.order_token_holder_config['alpha_multiplier']
            outer_alpha *= self.order_token_holder_config['alpha_multiplier']

            inner_linewidth *= self.order_token_holder_config['linewidth_multiplier']
            outer_linewidth *= self.order_token_holder_config['linewidth_multiplier']

        if not is_rotation:
            random_rotation = 0.0

        else:
            random_rotation = (-self.order_token_config['random_rotation_scale']
                               + 2.0 * self.order_token_config['random_rotation_scale'] * np.random.rand())

        # only add rotation if they are on a color group
        _rotation = random_rotation + self.order_token_config['rotations'][player]
        inner_alpha = self.order_token_config['alpha_inner'] if inner_color is None else 1.0
        outer_alpha = self.order_token_config['alpha_outer'] if inner_color is None else 1.0
        inner_color = self.order_token_config['color_inner'] if inner_color is None else inner_color

        # two patches for a flower token, one inside the other!
        outer_token_patch = self.flower_patch(
            center=location, rotation=_rotation,
            num_petals=self.order_token_config['num_petals'][player],
            color=self.order_token_config['colors'][player],
            extent=self.order_token_config['extents_outer'][player],
            scale=outer_scale,
            alpha=outer_alpha,
            edgecolor=self.order_token_config['edgecolor'],
            linewidth=outer_linewidth,
        )

        inner_token_patch = self.flower_patch(
            center=location, rotation=_rotation,
            num_petals=self.order_token_config['num_petals'][player],
            color=inner_color,
            extent=self.order_token_config['extents_inner'][player],
            scale=inner_scale,
            alpha=inner_alpha,
            edgecolor=self.order_token_config['edgecolor'],
            linewidth=inner_linewidth,
        )

        axis.add_patch(outer_token_patch)
        axis.add_patch(inner_token_patch)

        offx, offy = self.order_token_config['numeral_offsets'][number - 1]
        if not is_holder:
            axis.text(location[0] + offx, location[1] + offy, str(number),
                      rotation=random_rotation * 180.0 / np.pi,
                      **text_config,
                      horizontalalignment='center', verticalalignment='center',
                      zorder=1,
                      )

    def add_order_tokens(self, axis: plt.axis) -> None:
        """ display the locations of the order tokens and color tokens which indicate the order in which
        the color groups were taken:

        iterate over and add the token and the holder for each one. Don't worry about drawing over! Just
        make sure to draw the holder first.
        """
        color_orders = [self.game.active_color_order, self.game.passive_color_order]
        color_groups = [self.game.active_color_groups, self.game.passive_color_groups]

        if self.game.active_player == -1:
            color_orders = reversed(color_orders)
            color_groups = reversed(color_groups)

        for player, (_color_order, _color_group) in enumerate(zip(color_orders, color_groups)):
            untaken = 5
            # _order is a list of 0s for untaken colors and ints 1-6 to indicate the order in which that color was taken:
            for _color, _order in enumerate(_color_order):
                # draw the holders in reverse order:
                inner_color = self.token_config['flipped_colors'][_color] if _order > 0 else None
                self.draw_order_token(axis=axis,
                                      location=self.order_token_holder_config['locations'][player][_order - 1],
                                      player=player,
                                      number=_color + 1,
                                      is_holder=True,
                                      is_rotation=False,
                                      inner_color=inner_color,
                                      )

                # if the color group is taken, plot the token on the board, else plot it on top of the holder:
                if _order > 0:
                    token_location = self._convert_ij_xy(_color_group[_color][0])
                    number = _order
                    is_rotation = True
                    circle_location = self.order_token_holder_config['locations'][player][_order - 1]
                    #circle_location = self.mini_holder_config['locations'][player][_order - 1]
                    circle_radius = self.token_config['radius_mini']
                    circle_facecolor = self.token_config['flipped_colors'][_color]
                    circle_edgecolor = self.token_config['edgecolor_mini']
                    circle_linewidth = self.token_config['linewidth_mini']

                    offx, offy = self.order_token_config['numeral_offsets'][number - 1]
                    axis.text(circle_location[0] + offx, circle_location[1] + offy, str(number),
                              rotation=0,
                              **self.order_token_config['text_config'],
                              horizontalalignment='center', verticalalignment='center',
                              zorder=10,
                              )

                    # axis.add_patch(
                    #     patches.Circle(
                    #         xy=circle_location,
                    #         radius=circle_radius,
                    #         facecolor=circle_facecolor,
                    #         edgecolor=circle_edgecolor,
                    #         linewidth=circle_linewidth,
                    #         zorder=2,
                    #
                    #     )
                    # )

                else:
                    token_location = self.order_token_holder_config['locations'][player][untaken]
                    number = untaken + 1
                    is_rotation = False
                    circle_location = self.mini_holder_config['locations'][player][untaken]
                    circle_radius = self.mini_holder_config['radius']
                    circle_facecolor = self.mini_holder_config['facecolor']
                    circle_edgecolor = self.mini_holder_config['edgecolor']
                    circle_linewidth = self.mini_holder_config['linewidth']
                    untaken -= 1

                self.draw_order_token(axis=axis,
                                      location=token_location,
                                      player=player,
                                      number=number,
                                      is_holder=False,
                                      is_rotation=is_rotation,
                                      )

                # axis.add_patch(
                #     patches.Circle(
                #         xy=circle_location,
                #         radius=circle_radius,
                #         facecolor=circle_facecolor,
                #         edgecolor=circle_edgecolor,
                #         linewidth=circle_linewidth,
                #         zorder=2,
                #
                #     )
                # )

    @staticmethod
    def _convert_ij_xy(pij: tuple[int, int]) -> tuple[float, float]:
        """ change the ij view from the matrix representing the board to our
        spatial plotting xy view
        """
        return pij[1] - 3.5, 3.5 - pij[0]

    def get_colorgroup_patch(self, color_group: list[tuple[int, int]]) -> patches.Patch:
        """ Return the perimeter path surrounding the provided continuous locations:
        """
        locations = set(color_group)

        # integer offsets to give 4 surrounding lattice points for a location:
        offsets = [(1, 1), (0, 1), (0, 0), (1, 0)]
        all_poly_points = [(_i + _offi, _j + _offj) for (_offi, _offj) in offsets for (_i, _j) in locations]

        # Any point that occurs fewer than 4 times belongs to a perimeter:
        my_counter = Counter(all_poly_points)
        all_perimeter_points = {x for x, y in my_counter.items() if y < 4}

        # track possibly disjoint perimeters (eg internal holes):
        all_perimeters_xy = []

        # The first perimeter is guaranteed to be external as bottom-right corner is well-defined through max:
        perimeter1_xy, perimeter1_ij = self.extract_perimeter(start_point=max(all_perimeter_points),
                                                              perimeter_points=all_perimeter_points,
                                                              locations=locations,
                                                              # the outermost perimeter, facing the global "out":
                                                              is_outer=True,
                                                              )

        all_perimeters_xy.append(perimeter1_xy)

        # Points not accounted for through external perimeter:
        remaining_points = all_perimeter_points - set(perimeter1_ij)

        while len(remaining_points) > 0:
            # the remaining perimeters must all be internal:
            inner_perimeter_xy, inner_perim_ij = self.extract_perimeter(start_point=max(remaining_points),
                                                                        perimeter_points=all_perimeter_points,
                                                                        locations=locations,
                                                                        is_outer=False)

            all_perimeters_xy.append(inner_perimeter_xy)
            remaining_points -= inner_perim_ij

        ### if only one perimeter, we have a simply connected polygon:
        if len(all_perimeters_xy) == 1:
            return patches.Polygon(all_perimeters_xy[0])

        ### otherwise, create the path using MOVETO and LINETO codes:
        path_data = []
        code_data = []

        for _perim in all_perimeters_xy:
            path_data.extend(_perim + [_perim[0]])
            code_data.extend( [mpath.Path.MOVETO] + [mpath.Path.LINETO] * len(_perim) )

        path = mpath.Path(path_data, code_data)
        return patches.PathPatch(path)

    def extract_perimeter(self,
                          start_point: tuple[int, int],
                          perimeter_points: set[tuple[int, int]],
                          locations: set[tuple[int, int]],
                          is_outer: bool) -> tuple[list[tuple[float, float]], list[tuple[int, int]]]:
        """ Find the anticlockwise-oriented perimeter that contains the starting point.
        """
        (pi, pj) = start_point

        # (ij) offsets for the directions 0: right, 1: up, 2: left, 3: down
        dir_offsets = [(0, 1), (-1, 0), (0, -1), (1, 0)]

        # Spatial offsets are determined by the change in direction , ie by the orientation of the corner:
        change_dir_offset_dict = {
            (0, 1): (-1, 1), (1, 2): (-1, -1), (2, 3): (1, -1), (3, 0): (1, 1),
            (1, 0): (-1, 1), (2, 1): (-1, -1), (3, 2): (1, -1), (0, 3): (1, 1),
            # include same directions for the terminal points:
            (0, 0): (-1, 0), (1, 1): (-1, 0), (2, 2): (1, 0), (3, 3): (0, 1)
        }

        # The four lattice sites surrounding the starting perimeter point:
        spatial_start_points = [(pi, pj), (pi - 1, pj), (pi - 1, pj - 1), (pi, pj - 1), (pi, pj)]

        start_direction = 0

        # look for (empty, full) pattern to determine starting perimeter line segment:
        for start_direction, (p1, p2) in enumerate(zip(spatial_start_points, spatial_start_points[1:])):
            if (p1 not in locations) and (p2 in locations):
                break

        # determine the initial offset here, from the initial direction:
        prev_direction = start_direction
        offi, offj = dir_offsets[prev_direction]
        _i, _j = pi + offi, pj + offj
        prev_trial_point = (_i, _j)
        curr_direction = (prev_direction - 1) % 4

        # Continue until the initial point is seen again:
        terminal_point = (pi, pj)

        # track both the spatial coordinates and the integer lattice coordinates:
        perimeter_xy = []
        perimeter_ij = [(_i, _j)]

        # Furthermore, they are different again if the perimeter is internal:
        if not is_outer:
            change_dir_offset_dict = {_key: (-_i, -_j) for _key, (_i, _j) in change_dir_offset_dict.items()}

        # Offsets to recover the lattice sites from the vertices under investigation:
        check_empty_locs = [(0, 0), (-1, 0), (-1, -1), (0, -1), (0, 0)]

        while (_i, _j) != terminal_point:
            offi, offj = dir_offsets[curr_direction]
            trial_ij = (_i + offi, _j + offj)

            # Find points that are within the perimeter and where the (full, empty) pattern is conserved:
            locoff1_i, locoff1_j = check_empty_locs[curr_direction]
            locoff2_i, locoff2_j = check_empty_locs[curr_direction + 1]

            is_empty1 = (_i + locoff1_i, _j + locoff1_j) in locations
            is_empty2 = (_i + locoff2_i, _j + locoff2_j) in locations

            # Perimeter lines connect perimeter points without passing between two empty or two occupied lattice sites:
            if (trial_ij in perimeter_points) and (not is_empty1 == is_empty2):
                # Track all perimeter points encountered:
                perimeter_ij.append( trial_ij )

                # However, only add spatial points when the direction changes:
                if curr_direction != prev_direction:
                    _x, _y = self._convert_ij_xy(prev_trial_point)
                    _offx, _offy = change_dir_offset_dict[(prev_direction, curr_direction)]
                    perimeter_xy.append(( _x + _offx * self.colorgroup_config['offset'] - 0.5,
                                          _y + _offy * self.colorgroup_config['offset'] + 0.5))

                # Progress around the perimeter:
                _i += offi
                _j += offj

                # change direction by 1/4 turn clockwise:
                prev_direction = curr_direction
                curr_direction = (curr_direction - 1) % 4
                prev_trial_point = trial_ij

            else:
                # Keep scanning the surrounding four neighbors:
                curr_direction = (curr_direction + 1) % 4

        # reintroduce the spatial terminal point, using the changes in direction to determine the offset.
        _x, _y = self._convert_ij_xy(terminal_point)
        offx, offy = change_dir_offset_dict[(prev_direction, start_direction)]
        perimeter_xy.append( (_x + offx * self.colorgroup_config['offset'] - 0.5,
                              _y + offy * self.colorgroup_config['offset'] + 0.5) )

        return perimeter_xy, perimeter_ij

    @staticmethod
    def flower_patch(num_petals: int,
                     center: tuple[float, float],
                     rotation: float,
                     color: tuple[float, float, float],
                     extent: float,
                     scale: float = 1.0,
                     alpha: float = 1.0,
                     edgecolor: tuple[float, float, float] = (0.0, 0.0, 0.0),
                     linewidth: float = 0.5) -> patches.PathPatch:
        """ define a petal patch for tokens
        """
        # offset for center of the flower:
        _cx, _cy = center

        # specifies the protuberance of the petals:
        rads = [1.0 * scale, (1.0 + extent) * scale, (1.0 + extent) * scale, 1.0 * scale]

        # total angle arc that each petal will occupy
        petal_theta = 2. * np.pi / num_petals
        start_theta = rotation - petal_theta / 2.0

        # angle distances for regular spacing:
        theta_linspace = np.linspace(0.0, petal_theta, 4)

        # store the path and code data as we go
        path_data = []

        for _it in range(num_petals):
            mini_path = [(_r * np.cos(start_theta + _theta) + _cx, _r * np.sin(start_theta + _theta) + _cy)
                         for _r, _theta in zip(rads, theta_linspace)]

            path_data.extend(mini_path)
            start_theta += petal_theta

        # finish the path with the starting point:
        path_data.append(path_data[0])

        # Code presents move to and curves:
        code_data = [mpath.Path.LINETO, mpath.Path.CURVE4, mpath.Path.CURVE4, mpath.Path.CURVE4] * num_petals
        code_data.append(mpath.Path.CLOSEPOLY)
        code_data[0] = mpath.Path.MOVETO

        path = mpath.Path(path_data, code_data)
        patch = patches.PathPatch(path, facecolor=color, alpha=alpha, edgecolor=edgecolor, linewidth=linewidth)

        return patch

    def configure_axes(self, axis: plt.axis):
        axis.set_xlim(self.axis_xlimits)
        axis.set_ylim(self.axis_ylimits)
        axis.set_aspect('equal')

        axis.set_xticks([])
        axis.set_yticks([])
        plt.show()

    def shield(self,
               axis: plt.axis,
               xy: tuple[float, float],
               width: float,
               height: float,
               interior_offset: float,
               interior_facecolor: tuple[float, float, float],
               border_facecolor: tuple[float, float, float],
               interior_alpha: float,
               border_alpha: float,
               linewidth: float,
               is_flipped: bool = False,
               border_corner_curve: float=0.0,
               interior_corner_curve: float=0.0,
               interior_height_offset: float=0.1,
               ) -> None:
        """ A shield is actually two patches. One is the outline, which is a contained polygon, the other is the
        interior. In this way, separate alpha values may be used for each.

        if is_flipped, shield is upside down, and xy refers to bottom left not top left.
        """
        border_vertices = self._get_shield_boundary(xy, width, height, is_flipped=is_flipped,
                                                    corner_curve=border_corner_curve)

        interiorx = xy[0] + interior_offset
        interiory = xy[1] + interior_offset if is_flipped else xy[1] - interior_offset

        interior_vertices = self._get_shield_boundary( (interiorx, interiory),
                                                       width - 2.0 * interior_offset,
                                                       height - 2.0 * (interior_offset + interior_height_offset),
                                                       is_flipped=is_flipped, corner_curve=interior_corner_curve)

        curve_segment_codes = [mpath.Path.LINETO] + [mpath.Path.CURVE4] * 3

        # 5 shield lines, including curved top corners, plus final connecting line to finish:
        clockwise_codes = 4 * curve_segment_codes + [mpath.Path.LINETO]
        clockwise_codes[0] = mpath.Path.MOVETO # change first point to MOVETO object:
        anticlockwise_codes = [mpath.Path.MOVETO] + 4 * curve_segment_codes

        interior_path = mpath.Path(interior_vertices, clockwise_codes)
        border_path = mpath.Path(border_vertices + interior_vertices[::-1], clockwise_codes + anticlockwise_codes)

        interior_patch = patches.PathPatch(interior_path,
                                           facecolor=interior_facecolor, linewidth=linewidth, alpha=interior_alpha)
        border_patch = patches.PathPatch(border_path,
                                         facecolor=border_facecolor, linewidth=linewidth, alpha=border_alpha)

        # order shouldn't matter as the two patches are disjoint:
        axis.add_patch(interior_patch)
        axis.add_patch(border_patch)

    @staticmethod
    def _get_shield_boundary(xy: tuple[float, float],
                             width: float,
                             height: float,
                             is_flipped: bool=False,
                             corner_curve=0.0) -> list[tuple[float, float]]:
        """ get the list of vertices making a shield shape

        do it all in (0, 0) and move from there, including flipping.

        Corner curve controls the roundness of the shields:
        """

        max_off = 5.0
        off = max_off * corner_curve

        verts = [
            # first corner, top right:
            (0.0, -2.0 * off), (0.0, -off), (off, 0.0), (2.0 * off, 0.0),

            # line across and curve
            (20.0 - 2.0 * off, 0.0), (20.0 - off, 0.0), (20.0, -off), (20.0, -2.0 * off),

            # curved right bottom of shield:
            (20.0 , -15.0), (18.0, -22.0), (15.0, -27.0), (10.3, -29.5),

            # curved left bottom of shield:
            (9.7, -29.5), (5.0, -27.0), (2.0, -22.0), (0.0, -15.0),

            # finish:
            (0.0, -2.0 * off)
        ]

        _sgn = -1.0 if is_flipped else 1.0

        verts = [(xy[0] + width * (_x / 20.0), xy[1] + height * (_sgn * _y / 30.0)) for (_x, _y) in verts]
        return verts

    @staticmethod
    def get_placing_arrow_patch(point: tuple[float, float],
                                rotation: float,
                                head_width: float,
                                head_height: float,
                                body_width: float,
                                body_height: float,
                                color: tuple[float, float, float] = (1.0, 1.0, 1.0),
                                linewidth: float = 0.0,
                                alpha: float = 1.0,
                                ):
        """ vertical arrow pointing down to indicate that a spot may be placed in
        """

        verts = np.array([
            (0.0, 0.0),
            (0.5 * head_width , head_height), (0.5 * body_width, head_height),
            (0.5 * body_width, body_height), (-0.5 * body_width, body_height),
            (-0.5 * body_width, head_height), (-0.5 * head_width, head_height),
            (0.0, 0.0),
        ])

        ### Rotate:
        rotation_rads = rotation * np.pi / 180.0
        new_verts_x = np.cos(rotation_rads) * verts[:, 0] - np.sin(rotation_rads) * verts[:, 1]
        new_verts_y = np.sin(rotation_rads) * verts[:, 0] + np.cos(rotation_rads) * verts[:, 1]

        ### Translate:
        new_verts_x += point[0]
        new_verts_y += point[1]

        verts = np.zeros((len(new_verts_x), 2))
        verts[:, 0] = new_verts_x
        verts[:, 1] = new_verts_y

        codes = [mpath.Path.MOVETO] + (len(verts) - 1) * [mpath.Path.LINETO]

        arrow_path = mpath.Path(verts, codes)
        return patches.PathPatch(arrow_path, facecolor=color, linewidth=linewidth, alpha=alpha)

    @staticmethod
    def get_flip_arrow_patch(circle_center: tuple[float, float], # point of the arrow
                             radius: float,
                             head_height: float,
                             head_width: float,
                             body_width: float,
                             arc_angle: float, # degrees
                             rotation: float, # degrees
                             is_flipped: bool,
                             color: tuple[float, float, float] = (1.0, 1.0, 1.0),
                             linewidth: float = 0.0,
                             alpha: float = 1.0,
                             ):
        """ arrow around the circumference of the circle to indicate that it can be flipped.

        Circular arrow, around circle center point and radius r
        """
        rotation_rads = rotation * np.pi / 180.0

        # start off with radius 1 version, and head width is < 1.0
        arrow_head_verts = np.array([
            (1.0 + body_width / 2.0, 0.0),
            (1.0 + head_width / 2.0, 0.0),
            (1.0, head_height),
            (1.0 - head_width / 2.0, 0.0),
            (1.0 - body_width / 2.0, 0.0),
                            ])

        arrow_head_codes = [mpath.Path.MOVETO] + 4 * [mpath.Path.LINETO]

        # now do the clockwise arc first:
        arc1 = patches.Arc( (0.0, 0.0), 2.0 - body_width,
                            2.0 - body_width, theta1 = -arc_angle, theta2 = 0.0)
        arc2 = patches.Arc( (0.0, 0.0), 2.0 + body_width,
                            2.0 + body_width, theta1 = -arc_angle, theta2 = 0.0)

        arc_1_vertices = arc1.get_patch_transform().transform(arc1.get_path().vertices)[::-1]
        arc_2_vertices = arc2.get_patch_transform().transform(arc2.get_path().vertices)

        arc_1_codes = [mpath.Path.LINETO] + (len(arc_1_vertices) - 1) * [mpath.Path.CURVE4]
        arc_2_codes = [mpath.Path.LINETO] + (len(arc_2_vertices) - 1) * [mpath.Path.CURVE4]

        all_vertices = np.concatenate([arrow_head_verts, arc_1_vertices, arc_2_vertices,
                                       [(1.0 + body_width / 2.0, 0.0)]])
        all_codes = arrow_head_codes + arc_1_codes + arc_2_codes + [mpath.Path.CLOSEPOLY]

        if is_flipped:
            all_vertices[:, 0] *= -1.0

            ### ROTATE:
        new_all_verts_x = np.cos(rotation_rads) * all_vertices[:, 0] - np.sin(rotation_rads) * all_vertices[:, 1]
        new_all_verts_y = np.cos(rotation_rads) * all_vertices[:, 1] + np.sin(rotation_rads) * all_vertices[:, 0]

        ### SCALE:
        new_all_verts_x *= radius
        new_all_verts_y *= radius

        ### TRANSLATE:
        new_all_verts_x += circle_center[0]
        new_all_verts_y += circle_center[1]

        all_vertices = np.zeros((len(new_all_verts_x), 2))
        all_vertices[:, 0] = new_all_verts_x
        all_vertices[:, 1] = new_all_verts_y

        arrow_path = mpath.Path(all_vertices, all_codes)

        return patches.PathPatch(arrow_path, facecolor=color, linewidth=linewidth, alpha=alpha)
