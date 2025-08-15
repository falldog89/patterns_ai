""" create pretty commets, characterized by two circles connected by circular arcs


BUG

BlobComet().get_blobcomet_patch(
            (-0.1, -0.6),
            (-0.3, -0.6),
            0.03, 0.03,
            np.pi,
            np.pi,
            scale=aspect,
            translation=location,
            color=colors[-1],
            linewidth=0.0,
            edgecolor=(0.0, 0.0, 0.0),
        )

        issue is perfect tangents, which give infinite size to the circle.
"""
import matplotlib.path as mpath
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches
from typing import Optional

class Gem:
    def __init__(self) -> None:
        self.colors = {
            'red_bright': [(209, 52, 13), (240, 118, 62), (252, 144, 28), (237, 193, 33), (255, 248, 232)],
            'red_dark': [(94, 27, 21), (128, 36, 28), (219, 83, 24), (245, 190, 51), (255, 248, 232)],

            'yellow_bright': [(255,116,0), (255,141,0), (255,167,0), (255,222,26), (255, 255, 240)],
            'yellow_dark': [(105, 35, 9), (135, 75, 1), (255,116,0), (255,141,0), (255,206,0)],

            'purple_bright': [(128, 79, 179), (153, 105, 199), (181, 137, 214), (242, 146, 252), (240, 240, 255)],
            'purple_dark': [(20, 0, 40),  (85, 37, 134), (106, 53, 156), (128, 79, 179), (153, 105, 199)],

            'red_pastel': [(163, 81, 78), (189, 103, 100), (201, 141, 101), (227, 192, 138), (255, 248, 232)],

            'pastel_terra': [(221, 179, 152), (248, 197, 173), (255, 203, 188), (251, 224, 210), (249, 238, 226)],
            'pastel_gray': [(179, 179, 179), (187, 174, 165), (219, 212, 206), (221, 211, 210), (236, 231, 225)],

            'green': [(9, 64, 8), (13, 94, 11), (25, 179, 21), (135, 247, 59), (245, 250, 242)],

            'blue_bright': [(17, 23, 194), (73, 78, 235), (73, 130, 235), (89, 192, 247), (247, 252, 255)],
            'blue_dark': [(8, 6, 69), (17, 13, 133), (34, 27, 224), (36, 157, 227), (247, 252, 255)],

            'purple': [(44, 10, 74), (76, 17, 128), (132, 48, 207), (203, 70, 240), (246, 242, 247)],
            'lime': [(49, 64, 6), (92, 120, 11), (169, 214, 36), (216, 227, 116), (246, 242, 247)],
            'magic': [(54, 5, 24), (128, 19, 100), (162, 134, 247), (49, 218, 247), (246, 242, 247)],
        }

    @staticmethod
    def _rgb(color: tuple[int, int, int]) -> tuple[float, float, float]:
        return color[0] / 255.0, color[1] / 255.0, color[2] / 255.0

    def gem_1(self,
              color: str,
              axis: plt.axis,
              location: tuple[float, float] = (0.0, 0.0),
              aspect: tuple[float, float] = (1.0, 1.0),
              scale: Optional[float] = None,
              rotation: Optional[float] = None,
              ) -> None:
        """ first style of eldar gem
        """
        colors = self.colors[color.lower()]
        aspect = (scale * aspect[0], scale * aspect[1]) if scale is not None else aspect

        # base, 3 color highlights, 1 white highlight:
        center_radii = [0.95, 0.95, 0.95]
        first_blob_radii = [_mult * (1.0 - _r) for _mult, _r in zip([0.8, 0.5, 0.4], center_radii)]
        second_blob_radii = [_mult * (1.0 - _r) for _mult, _r in zip([0.6, 0.35, 0.2], center_radii)]
        theta_starts = [np.pi + 0.1, np.pi + 0.5, np.pi + 1.2]
        theta_ends = [4.0 * np.pi / 9.0, np.pi / 3.0, np.pi / 6.0]
        cw_offs = [0.4, 0.3, 0.2] # offset from just being perpendicular:
        acw_offs = [0.0, 0.0, 0.0]
        linewidths = [0.0, 0.0, 0.0]

        ### BASE GEM:
        gem_patches = [patches.Ellipse(location, aspect[0] * 2.0, aspect[1] * 2.0,
                                   facecolor=self._rgb(colors[0]), edgecolor='black', linewidth= 2.0)]

        for _r, _r1, _r2, _start, _end, _cw_off, _acw_off, _col, _lw in zip(center_radii,
                                                                            first_blob_radii,
                                                                            second_blob_radii,
                                                                            theta_starts,
                                                                            theta_ends,
                                                                            cw_offs,
                                                                            acw_offs,
                                                                            colors[1:-1],
                                                                            linewidths,):
            gem_patches.append(
                BlobComet().get_blobcomet_patch(
                    (_r * np.cos(_start), _r * np.sin(_start)),
                    (_r * np.cos(_end), _r * np.sin(_end)),
                    _r1,
                    _r2,
                    _start + np.pi / 2.0 + _cw_off,
                    _start + np.pi / 2.0 + _acw_off,
                    scale=aspect,
                    translation=location,
                    color=self._rgb(_col),
                    linewidth=0.0,
                    rotation=rotation,
                )
            )

        # add on final highlight:
        gem_patches.append(
            BlobComet().get_blobcomet_patch(
                (0.4, 0.35),
                (0.2, 0.55),
                0.04, 0.07, np.pi-1.5, np.pi - 0.5,
                scale=aspect, translation=location, color=self._rgb(colors[-1]), linewidth=0.0
            )
        )

        # add all the collected patches in order:
        for _patch in gem_patches:
            axis.add_patch(_patch)

    def gem_2(self,
              color: str,
              axis: plt.axis,
              location: tuple[float, float] = (0.0, 0.0),
              aspect: tuple[float, float] = (1.0, 1.0),
              scale: Optional[float] = None,
              rotation: Optional[float] = None,
              ) -> None:
        """ first style of eldar gem
        """
        color_order = [0, 1, 1, 2, 2, 3, 3, 4, 4]
        colors = [self.colors[color.lower()][_x] for _x in color_order]
        aspect = (scale * aspect[0], scale * aspect[1]) if scale is not None else aspect

        # base, 3 color highlights, 1 white highlight:
        center_radii = [0.9, 0.9, 0.93, 0.93, 0.94, 0.94, 0.95, 0.95]
        first_blob_radii = [_mult * (1.0 - _r) for _mult, _r in zip([0.9, 0.6, 0.4, 0.5, 0.4, 0.4, 0.4, 0.3], center_radii)]
        second_blob_radii = [_mult * (1.0 - _r) for _mult, _r in zip([1.0, 0.6, 0.4, 0.5, 0.4, 0.4, 0.4, 0.3], center_radii)]
        theta_starts = [
            np.pi / 2.0 + 0.5,
            0.2,
            np.pi / 2.0 + 0.6,
            0.25,
            np.pi / 2.0 + 0.7,
            0.3,
            np.pi / 2.0 + 1.2,
            0.55]
        theta_ends = [
            3.0 * np.pi / 2.0 + 0.6, np.pi / 2.0 - 0.2,
            3.0 * np.pi / 2.0 + 0.5, np.pi / 2.0 - 0.2 -0.05,
            3.0 * np.pi / 2.0 + 0.4, np.pi / 2.0 - 0.2 - 0.1,
            3.0 * np.pi / 2.0 + 0.3, np.pi / 2.0 - 0.2 - 0.3,
        ]

        cw_offs = [0.3, 0.9, 0.2, 0.8, 0.2, 0.8, 0.1, 0.5]
        acw_offs = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        linewidths = [0.0] * len(cw_offs)

        ### BASE GEM:
        gem_patches = [patches.Ellipse(location, aspect[0] * 2.0, aspect[1] * 2.0,
                                       facecolor=self._rgb(colors[0]), edgecolor='black', linewidth=2.0)]

        for _r, _r1, _r2, _start, _end, _cw_off, _acw_off, _col, _lw in zip(center_radii,
                                                                            first_blob_radii,
                                                                            second_blob_radii,
                                                                            theta_starts,
                                                                            theta_ends,
                                                                            cw_offs,
                                                                            acw_offs,
                                                                            colors[1:],
                                                                            linewidths, ):
            gem_patches.append(
                BlobComet().get_blobcomet_patch(
                    (_r * np.cos(_start), _r * np.sin(_start)),
                    (_r * np.cos(_end), _r * np.sin(_end)),
                    _r1,
                    _r2,
                    _start + np.pi / 2.0 + _cw_off,
                    _start + np.pi / 2.0 + _acw_off,
                    scale=aspect,
                    translation=location,
                    color=self._rgb(_col),
                    linewidth=0.0,
                    rotation=rotation,
                )
            )

        # add on final highlight:
        gem_patches.append(
            BlobComet().get_blobcomet_patch(
                (0.0, 0.8),
                (0.0, 0.87),
                0.04, 0.01, np.pi - 0.4, 0.4,
                scale=aspect, translation=location, color=self._rgb(colors[-1]), linewidth=0.0
            )
        )

        # add all the collected patches in order:
        for _patch in gem_patches:
            axis.add_patch(_patch)

    def gem_3(self,
              color: str,
              axis: plt.axis,
              location: tuple[float, float] = (0.0, 0.0),
              aspect: tuple[float, float] = (1.0, 1.0),
              scale: Optional[float] = None,
              rotation: Optional[float] = None,
              linewidth: float=0.0,
              ) -> None:
        """ first style of eldar gem
        """
        colors = self.colors[color.lower()]
        aspect = (scale * aspect[0], scale * aspect[1]) if scale is not None else aspect

        ### BASE GEM:
        top_center = (0.0, 1.0)
        bottom_center = (0.0, 0.0)

        gem_patches = [BlobComet().get_blobcomet_patch(
            top_center, bottom_center,
            0.1, 0.6,
            -np.pi / 4.0, 5.0 * np.pi / 4.0,
            scale=aspect,
            rotation=rotation,
            translation=location,
            color=self._rgb(colors[0]), linewidth=linewidth, edgecolor=(0.0,0.0,0.0))]

        # highlight 1
        tstart = 6.0 * np.pi / 5.0
        tend = 0.0
        _r = 0.55
        gem_patches.append(
            BlobComet().get_blobcomet_patch(
                (_r * np.cos(tstart), _r * np.sin(tstart)),
                (_r * np.cos(tend), _r * np.sin(tend)),
                0.02, 0.02,
                tstart + np.pi / 2.0 + 0.5,
                tstart + np.pi / 2.0 + 0.0,
                scale=aspect,
                translation=location,
                color=self._rgb(colors[1]),
                linewidth=0.0,
                rotation=rotation,
            )
        )

        # highlight 2
        tstart = 6.0 * np.pi / 5.0 + 0.2
        tend = 0.0 - 0.2
        _r = 0.55
        gem_patches.append(
            BlobComet().get_blobcomet_patch(
                (_r * np.cos(tstart), _r * np.sin(tstart)),
                (_r * np.cos(tend), _r * np.sin(tend)),
                0.02, 0.02,
                tstart + np.pi / 2.0 + 0.3,
                tstart + np.pi / 2.0 + 0.0,
                scale=aspect,
                translation=location,
                color=self._rgb(colors[2]),
                linewidth=0.0,
                rotation=rotation,
            )
        )

        # highlight 3
        tstart = 6.0 * np.pi / 5.0 + 0.5
        tend = 0.0 - 0.4
        _r = 0.55
        gem_patches.append(
            BlobComet().get_blobcomet_patch(
                (_r * np.cos(tstart), _r * np.sin(tstart)),
                (_r * np.cos(tend), _r * np.sin(tend)),
                0.01, 0.01,
                tstart + np.pi / 2.0 + 0.15,
                tstart + np.pi / 2.0 + 0.0,
                scale=aspect,
                translation=location,
                color=self._rgb(colors[3]),
                linewidth=0.0,
                rotation=rotation,
            )
        )

        # highlight 4
        tstart = 6.0 * np.pi / 5.0 + 1.0
        tend = 0.0 - 1.0
        _r = 0.55
        gem_patches.append(
            BlobComet().get_blobcomet_patch(
                (_r * np.cos(tstart), _r * np.sin(tstart)),
                (_r * np.cos(tend), _r * np.sin(tend)),
                0.01, 0.01,
                tstart + np.pi / 2.0 + 0.15,
                tstart + np.pi / 2.0 + 0.0,
                scale=aspect,
                translation=location,
                color=self._rgb(colors[4]),
                linewidth=0.0,
                rotation=rotation,
            )
        )

        # highlight 5
        gem_patches.append(
            BlobComet().get_blobcomet_patch(
                (-0.3, 0.0), (-0.3, 0.1),
                0.05, 0.01,
                np.pi / 2.0 + 0.3, np.pi / 2.0 + 0.3,
                scale=aspect,
                translation=location,
                color=self._rgb(colors[4]),
                linewidth=0.0,
                rotation=rotation,
            )
        )

        # highlight 6
        gem_patches.append(
            BlobComet().get_blobcomet_patch(
                (0.45, 0.3), (0.33, 0.6),
                0.03, 0.01,
                np.pi / 2.0 + 0.3, np.pi / 2.0 + 0.3,
                scale=aspect,
                translation=location,
                color=self._rgb(colors[2]),
                linewidth=0.0,
                rotation=rotation,
            )
        )

        # highlight 7
        gem_patches.append(
            BlobComet().get_blobcomet_patch(
                (0.0, 1.04), (-0.25, 0.75),
                0.03, 0.01,
                5.0 * np.pi / 4.0 - 0.2, 5.0 * np.pi / 4.0 - 0.2,
                scale=aspect,
                translation=location,
                color=self._rgb(colors[4]),
                linewidth=0.0,
                rotation=rotation,
            )
        )

        # highlight 8
        gem_patches.append(
            BlobComet().get_blobcomet_patch(
                (0.0, 1.04), (0.1, 0.95),
                0.03, 0.01,
                -0.6, -0.8,
                scale=aspect,
                translation=location,
                color=self._rgb(colors[4]),
                linewidth=0.0,
                rotation=rotation,
            )
        )

        for _patch in gem_patches:
            axis.add_patch(_patch)

    def gem_4(self,
                      axis: plt.axis,
                      location: tuple[float, float],
                      scale: float) -> None:
        cx, cy = location
        #### COlors for the gems:
        colors = [
            (5.0 / 255.0, 10.0 / 255.0, 5.0 / 255.0), # black
            (44.0 / 255.0, 38.0 / 255.0, 163.0 / 255.0), # dark blue
            (62.0 / 255.0, 105.0 / 255.0, 214.0 / 255.0), # light blue
            (215.0 / 255.0, 247.0 / 255.0, 252.0 / 255.0), # lightest blue
            (242.0 / 255.0, 198.0 / 255.0, 136.0 / 255.0), # orangey hilight
        ]

        ### BASE GEM:
        aspect = (scale * 1.0, scale * 1.5)
        rs = [0.94, 0.94, 0.93]
        mini_rs = [_mult * (1.0 - _r) for _mult, _r in zip([0.6, 0.5, 0.1], rs)]
        theta_starts = [3.0 * np.pi / 2.0 - 0.8,
                        3.0 * np.pi / 2.0 - 0.7,
                        3.0 * np.pi / 2.0 - 0.1,
                        ]
        theta_ends = [np.pi / 4.0,
                      np.pi / 4.0 - 0.1,
                      np.pi / 4.0 - 0.4,
                      ]
        cw_offs = [0.7, 0.5, 0.4]
        acw_offs = [0.0, 0.0, 0.0]
        linewidths = [0.0, 0.0, 0.0]
        gem_base = patches.Ellipse((cx, cy), scale * (1.03 * 2.0), scale * (1.53 * 2.0),
                                   facecolor=colors[0], edgecolor='black', linewidth=2.0)
        axis.add_patch(gem_base)

        for _r, _mini_r, _theta_start, _theta_end, _cw_off, _acw_off, _col, _lw in zip(rs, mini_rs,
                                                                                       theta_starts,
                                                                                       theta_ends, cw_offs,
                                                                                       acw_offs,
                                                                                       colors[1:],
                                                                                       linewidths):
            _gem_patch = BlobComet().get_blobcomet_patch(
                (_r * np.cos(_theta_start), _r * np.sin(_theta_start)),
                (_r * np.cos(_theta_end), _r * np.sin(_theta_end)),
                _mini_r, _mini_r,
                _theta_start + np.pi / 2.0 + _cw_off,
                _theta_start + np.pi / 2.0 + _acw_off,
                scale=aspect,
                translation=location,
                color=_col,
                linewidth=0.0,#_lw * scale,
                edgecolor=(0.0, 0.0, 0.0),
            )
            axis.add_patch(_gem_patch)

        # ADDITIONAL IRRELGULAR HIGHLIGHTS
        # upper right orange streak:
        _gem_highlight_1 = BlobComet().get_blobcomet_patch(
            (0.6, 0.05),
            (0.5, 0.4),
            0.04, 0.02,
            1.0 * np.pi / 3.0 - 0.2,
            1.0 * np.pi / 3.0 + 0.1,
            scale=aspect,
            translation=location,
            color=colors[-1],
            linewidth=0.0, #1.0 * scale,
            edgecolor=(0.0, 0.0, 0.0),
        )

        # Lower left orange streak:
        _gem_highlight_2 = BlobComet().get_blobcomet_patch(
            (-0.2, -0.8),
            (-0.3, -0.77),
            0.03, 0.03,
            1.1 * np.pi,
            1.0 * np.pi,
            scale=aspect,
            translation=location,
            color=colors[-1],
            linewidth=0.0, #1.0 * scale,
            edgecolor=(0.0, 0.0, 0.0),
        )

        # middle white:
        _gem_highlight_3 = BlobComet().get_blobcomet_patch(
            (0.03, -0.35),
            (0.07, -0.3),
            0.05, 0.05,
            np.pi / 6.0,
            np.pi / 2.0,
            scale=aspect,
            translation=location,
            color=colors[-2],
            linewidth=0.0, #1.0 * scale,
            edgecolor=(0.0, 0.0, 0.0),
        )

        # upper left
        _gem_highlight_4 = BlobComet().get_blobcomet_patch(
            (-0.65, 0.7),
            (-0.5, 0.8),
            0.05, 0.05,
            np.pi / 4.0,
            np.pi / 2.0,
            scale=aspect,
            translation=location,
            color=colors[-2],
            linewidth=0.0, #1.0 * scale,
            edgecolor=(0.0, 0.0, 0.0),
        )
        axis.add_patch(_gem_highlight_1)
        axis.add_patch(_gem_highlight_2)
        axis.add_patch(_gem_highlight_3)
        axis.add_patch(_gem_highlight_4)



class BlobComet:

    def get_blobcomet_chain(self,
                            center_list: list[tuple[float, float]],
                            radius_list: list[float],
                            start_cw_angle: float,
                            start_acw_angle: float,
                            rotation: Optional[float] = None
                            ):
        """ The idea here is to chain together a bunch of different centers with blobs!
        todo either do as overlapping patches or something more involved...
        """
        return

    def get_blobcomet_patch(self,
                            center_1: tuple[float, float],
                            center_2: tuple[float, float],
                            radius_1: float,
                            radius_2: float,
                            cw_angle: float,
                            acw_angle: float,
                            rotation: Optional[float] = None,
                            scale: Optional[tuple[float, float]] = None,
                            translation: Optional[tuple[float, float]] = None,
                            color: tuple[float, float, float] = (0.1, 0.4, 0.7),
                            alpha: float=1.0,
                            edgecolor: tuple[float, float, float] = (0.2, 0.5, 0.9),
                            linewidth: float = 4.0,
                            ) -> patches.PathPatch:
        """ Return a matplotlib patch of a blob comet:
        """
        blob_comet_path = self.get_blobcomet_path(center_1, center_2,
                                                  radius_1, radius_2,
                                                  cw_angle, acw_angle,
                                                  rotation,
                                                  scale,
                                                  translation)

        # populate the actual patch:
        blob_comet_patch = patches.PathPatch(blob_comet_path,
                                             facecolor=color,
                                             alpha=alpha,
                                             edgecolor=edgecolor,
                                             linewidth=linewidth,)

        return blob_comet_patch

    def get_blobcomet_path(self,
                           center_1: tuple[float, float],
                           center_2: tuple[float, float],
                           radius_1: float,
                           radius_2: float,
                           cw_angle: float,
                           acw_angle: float,
                           rotation: Optional[float] = None,
                           scale: Optional[tuple[float, float]] = None,
                           translation: Optional[tuple[float, float]] = None,
                           ) -> mpath.Path:
        """ Two circles, (center_i, radius_i), and the clockwise, anticlockwise leaving angles
        from circle 1.
        """
        # Rotate the starting data if necessary:
        if rotation is not None:
            # rotate around the origin for now:
            center_1, center_2, cw_angle, acw_angle = self._rotate_start(center_1, center_2,
                                                                         cw_angle, acw_angle, rotation)

        # Use formula for cos(theta +- pi/2 ), sin(theta +- pi/2):
        cw_point = (center_1[0] - radius_1 * np.sin(cw_angle), center_1[1] + radius_1 * np.cos(cw_angle))
        acw_point = (center_1[0] + radius_1 * np.sin(acw_angle), center_1[1] - radius_1 * np.cos(acw_angle))

        ### CHECK 1:
        # cw and acw points cannot be contained within circle 2:
        dist_cw = self._dist_xy(cw_point, center_2)
        dist_acw = self._dist_xy(acw_point, center_2)

        if (dist_cw < radius_2) or (dist_acw < radius_2):
            print(f"The tangent points sit within the target circle!")
            raise ValueError(f"The tangent points sit within the target circle!")
        ####

        ####
        # Determine the center, radius and bounding angle for the circular arcs connecting the circle one tangent
        # points with circle two.
        ###
        arc_1_center, arc_1_radius, arc_1_theta_1, arc_1_theta_2, arc_1_is_major = self._get_arc_circle(cw_point,
                                                                                                       cw_angle,
                                                                                                       center_2,
                                                                                                       radius_2,
                                                                                                       True)

        arc_2_center, arc_2_radius, arc_2_theta_1, arc_2_theta_2, arc_2_is_major = self._get_arc_circle(acw_point,
                                                                                                       acw_angle,
                                                                                                       center_2,
                                                                                                       radius_2,
                                                                                                       False)
        ####

        ### The path must be traversed anticlockwise, which dictates which arc angles come first:
        circle_1_theta_start = arc_1_theta_1 if arc_1_is_major else arc_1_theta_1 + 180.0
        circle_1_theta_end = arc_2_theta_1 if arc_2_is_major else arc_2_theta_1 + 180.0

        circle_2_theta_start = arc_2_theta_2 if arc_2_is_major else arc_2_theta_2 + 180.0
        circle_2_theta_end = arc_1_theta_2 if arc_1_is_major else arc_1_theta_2 + 180.0

        # arc defined by the theta approach angles
        arc_1_theta_start = arc_1_theta_2 if arc_1_is_major else arc_1_theta_1
        arc_1_theta_end = arc_1_theta_1 if arc_1_is_major else arc_1_theta_2

        arc_2_theta_start = arc_2_theta_1 if arc_2_is_major else arc_2_theta_2
        arc_2_theta_end = arc_2_theta_2 if arc_2_is_major else arc_2_theta_1

        ### CHECK 2: Major and minor arcs cannot intersect, and circle 1 and circle 2 arcs cannot intersect:
        is_intersect_1 = self._check_if_arcs_intersect(arc_1_center, arc_1_radius, arc_1_theta_start, arc_1_theta_end,
                                                       arc_2_center, arc_2_radius, arc_2_theta_start, arc_2_theta_end)
        if is_intersect_1:
            print(f"The two arcs intersect one another...")
            raise ValueError(f"The two arcs intersect one another...")

        # check that the arcs don't intersect one another!
        is_intersect_2 = self._check_if_arcs_intersect(center_1, radius_1, circle_1_theta_start, circle_1_theta_end,
                                                       center_2, radius_2, circle_2_theta_start, circle_2_theta_end, )

        if is_intersect_2:
            print(f"The two circles intersect one another...")
            raise ValueError(f"The two circles intersect one another...")

        ####

        ### Define the four circular paths in turn: circle 1 -> arc 2 -> circle 2 -> arc 1

        ### Define the four circular paths, in counter clockwise orientation:
        circle_1_arc = patches.Arc(center_1, 2.0 * radius_1, 2.0 * radius_1,
                                   theta1=circle_1_theta_start, theta2=circle_1_theta_end)
        circle_1_vertices = circle_1_arc.get_patch_transform().transform(circle_1_arc.get_path().vertices)
        circle_1_codes = [mpath.Path.MOVETO] + [mpath.Path.CURVE4] * (len(circle_1_vertices) - 1)

        circle_2_arc = patches.Arc(center_2, 2.0 * radius_2, 2.0 * radius_2,
                                   theta1=circle_2_theta_start, theta2=circle_2_theta_end)
        circle_2_vertices = circle_2_arc.get_patch_transform().transform(circle_2_arc.get_path().vertices)
        circle_2_codes = [mpath.Path.LINETO] + [mpath.Path.CURVE4] * (len(circle_2_vertices) - 1)

        arc_1 = patches.Arc(arc_1_center, 2.0 * arc_1_radius, 2.0 * arc_1_radius,
                            theta1=arc_1_theta_start, theta2=arc_1_theta_end)
        arc_1_vertices = arc_1.get_patch_transform().transform(arc_1.get_path().vertices)

        # reverse a MINOR first arc to preserve counter-clockwise:
        if not arc_1_is_major:
            arc_1_vertices = arc_1_vertices[::-1]

        arc_1_codes = [mpath.Path.LINETO] + [mpath.Path.CURVE4] * (len(arc_1_vertices) - 1)

        arc_2 = patches.Arc(arc_2_center, 2.0 * arc_2_radius, 2.0 * arc_2_radius,
                            theta1=arc_2_theta_start, theta2=arc_2_theta_end)
        arc_2_vertices = arc_2.get_patch_transform().transform(arc_2.get_path().vertices)

        # reverse a MAJOR second arc to preserve counter-clockwise:
        if not arc_2_is_major:  # reverse it:
            arc_2_vertices = arc_2_vertices[::-1]

        arc_2_codes = [mpath.Path.LINETO] + [mpath.Path.CURVE4] * (len(arc_2_vertices) - 1)

        #### Join the arcs together as circle 1 -> arc 2 -> circle 2 -> arc 2:
        vertices = np.concatenate([circle_1_vertices, arc_2_vertices, circle_2_vertices, arc_1_vertices])
        code_data = np.concatenate([circle_1_codes, arc_2_codes, circle_2_codes, arc_1_codes])

        # scale the vertices if scale is not None:
        if scale is not None:
            vertices[:, 0] *= scale[0]
            vertices[:, 1] *= scale[1]

        if translation is not None:
            vertices[:, 0] += translation[0]
            vertices[:, 1] += translation[1]

        my_path = mpath.Path(vertices, code_data)

        return my_path

    @staticmethod
    def _rotate_start(center_1: tuple[float, float],
                      center_2: tuple[float, float],
                      cw_angle: float,
                      acw_angle: float,
                      theta: Optional[float]) -> tuple[tuple[float, float], tuple[float, float], float, float]:
        """ rotate the starting locations and angles by the provided quantity
        """
        if theta is None:
            return center_1, center_2, cw_angle % (2.0 * np.pi), acw_angle % (2.0 * np.pi)

        ctheta, stheta = np.cos(theta), np.sin(theta)

        new_center_1 = (ctheta * center_1[0] - stheta * center_1[1],
                        stheta * center_1[0] + ctheta * center_1[1])

        new_center_2 = (ctheta * center_2[0] - stheta * center_2[1],
                        stheta * center_2[0] + ctheta * center_2[1])

        return new_center_1, new_center_2, (cw_angle + theta) % (2.0 * np.pi), (acw_angle + theta) % (2.0 * np.pi)

    def _get_arc_circle(self,
                        tangent_point: tuple[float, float],
                        tangent_angle: float,
                        circle_center: tuple[float, float],
                        circle_radius: float,
                        is_clockwise: bool) -> tuple[tuple[float, float], float, float, float, bool]:
        """
        :param tangent_point: point that circular arc must pass through,
        :param tangent_angle: angle with +ve x axis that the circular arc must pass through p1 at,
        :param circle_center: center of circle to which the circular arc will touch,
        :param circle_radius: radius of circle
        :param is_clockwise: orientation of tangent leaving circle.
        :return: arc center, radius, angle from center to p1, angle from center to tangent circle point,
         whether arc is major or minor
        """
        ### Distance between the point and the circle center, for rescaling the coordinate system:
        scale_radius = self._dist_xy(tangent_point, circle_center)

        # angle from point to circle center:
        theta = self._get_angle_between_points(tangent_point, circle_center, is_degrees=False)

        # rotate the start angle relative to the rotated bulb:
        start_angle = (tangent_angle - theta) % (2.0 * np.pi)
        is_flipped = False

        # if start angle is greater than pi, flip it so that the angle is in the range [0, pi]:
        if start_angle > np.pi:
            start_angle = 2.0 * np.pi - start_angle
            is_flipped = True

        # scaled radius of circle
        r = circle_radius / scale_radius

        # the angle formed between the tangent of the circle passing through the origin:
        tangent_theta = np.arcsin(r)

        # Approach angles cannot fall within a cone that would prohibit a tangent:
        if start_angle > np.pi - tangent_theta:
            print(f"Illegal angle - tail cannot form tangent teardrop with bulb.")
            raise ValueError(f"Illegal angle - tail cannot form tangent teardrop with bulb.")

        # whether to use major or minor tangent arcs: can tidy up later:
        if is_clockwise:
            if is_flipped:
                is_major = False

            else:
                if start_angle < tangent_theta:
                    is_major = False

                else:
                    is_major = True

        else:
            if is_flipped:
                if start_angle > tangent_theta:
                    is_major = True

                else:
                    is_major = False

            else:
                is_major = False

        # Flip orientation if the arc is flipped:
        if is_flipped:
            is_clockwise = not is_clockwise

        # Define tangent circle center and radius:
        (arc_center_x, arc_center_y), arc_radius = self._get_tangent_circle(r, start_angle, is_major=is_major,
                                                                           is_clockwise=is_clockwise)

        # Arc interval, in radians:
        point_theta = self._get_angle_between_points((arc_center_x, arc_center_y), (0, 0), is_degrees=True)
        circle_theta = self._get_angle_between_points((arc_center_x, arc_center_y), (1, 0), is_degrees=True)

        # Rescale, flip, translate, rotate back to original coordinate system:
        if is_flipped:
            # flip the angles and the arc center y value:
            point_theta = (point_theta * -1) % 360.0
            circle_theta = (circle_theta * -1) % 360.0
            arc_center_y *= -1

        # theta is treated in degrees for mpl path:
        point_theta += theta * 180.0 / np.pi
        circle_theta += theta * 180.0 / np.pi

        # scale back:
        arc_center_x *= scale_radius
        arc_center_y *= scale_radius
        arc_radius *= scale_radius

        # rotate back:
        new_arc_center_x = np.cos(theta) * arc_center_x - np.sin(theta) * arc_center_y
        new_arc_center_y = np.sin(theta) * arc_center_x + np.cos(theta) * arc_center_y

        # translate back:
        new_arc_center_x += tangent_point[0]
        new_arc_center_y += tangent_point[1]

        return (new_arc_center_x, new_arc_center_y), arc_radius, point_theta % 360.0, circle_theta % 360.0, is_major

    def _check_if_arcs_intersect(self,
                                 arc_1_center: tuple[float, float],
                                 arc_1_radius: float,
                                 arc_1_theta_1: float,
                                 arc_1_theta_2: float,
                                 arc_2_center: tuple[float, float],
                                 arc_2_radius: float,
                                 arc_2_theta_1: float,
                                 arc_2_theta_2: float) -> bool:
        """ Determine whether two circular arcs intersect or not.

        It is assumed that arc i theta 1 comes prior to arc i theta 2, in a counter-clockwise sense.
        """
        # Determine the possible intersection points:
        i1, i2 = self._get_circle_intersections(arc_1_center, arc_1_radius, arc_2_center, arc_2_radius)

        if i1 is None:
            # if not intersection, because one circle is within the other or they do not overlap:
            return False

        # Arc angles for the intersects, from each circle center:
        arc_1_int_angle_1 = self._get_angle_between_points(arc_1_center, i1)
        arc_1_int_angle_2 = self._get_angle_between_points(arc_1_center, i2)
        arc_2_int_angle_1 = self._get_angle_between_points(arc_2_center, i1)
        arc_2_int_angle_2 = self._get_angle_between_points(arc_2_center, i2)

        # conditions for the angles to fall within the defined arc provided:
        if arc_1_theta_1 > arc_1_theta_2:
            arc_1_int_1_is_bad = True if ((arc_1_theta_1 < arc_1_int_angle_1)
                                          or (arc_1_int_angle_1 < arc_1_theta_2)) else False
            arc_1_int_2_is_bad = True if ((arc_1_theta_1 < arc_1_int_angle_2)
                                          or (arc_1_int_angle_2 < arc_1_theta_2)) else False

        else:
            arc_1_int_1_is_bad = True if (arc_1_theta_1 < arc_1_int_angle_1 < arc_1_theta_2) else False
            arc_1_int_2_is_bad = True if (arc_1_theta_1 < arc_1_int_angle_2 < arc_1_theta_2) else False

        if arc_2_theta_1 > arc_2_theta_2:
            arc_2_int_1_is_bad = True if ((arc_2_theta_1 < arc_2_int_angle_1)
                                          or (arc_2_int_angle_1 < arc_2_theta_2)) else False
            arc_2_int_2_is_bad = True if ((arc_2_theta_1 < arc_2_int_angle_2)
                                          or (arc_2_int_angle_2 < arc_2_theta_2)) else False

        else:
            arc_2_int_1_is_bad = True if (arc_2_theta_1 < arc_2_int_angle_1 < arc_2_theta_2) else False
            arc_2_int_2_is_bad = True if (arc_2_theta_1 < arc_2_int_angle_2 < arc_2_theta_2) else False

        if arc_1_int_1_is_bad and arc_2_int_1_is_bad:
            return True

        if arc_1_int_2_is_bad and arc_2_int_2_is_bad:
            return True

        return False

    def _get_circle_intersections(self,
                                  center_1: tuple[float, float],
                                  radius_1: float,
                                  center_2: tuple[float, float],
                                  radius_2: float) -> tuple[Optional[tuple[float, float]], Optional[tuple[float, float]]]:
        """ return the two intersection points between the circles
        """
        # Confirm that the distance between the two centers permits an intersection:
        center_distance = self._dist_xy(center_1, center_2)

        # we are not interested in zero or 1 intersection:
        if center_distance >= (radius_1 + radius_2):
            return None, None

        # Rotate the centers to fall along the x-axis:
        theta = self._get_angle_between_points(center_1, center_2, is_degrees=False)

        # (common) x intersection point for both points in this coordinate system:
        x = (radius_1 ** 2.0 - radius_2 ** 2.0 + center_distance ** 2.0) / (2.0 * center_distance)

        # imaginary solutions if x > radius_1 => one circle within the other entirely:
        if x > radius_1:
            return None, None

        # mirror image y intersections, due to symmetry of coordinate system:
        y1 = (radius_1 ** 2.0 - x ** 2.0) ** 0.5
        y2 = -y1

        # Rotate the points back:
        rot_x1 = np.cos(theta) * x - np.sin(theta) * y1
        rot_x2 = np.cos(theta) * x - np.sin(theta) * y2
        rot_y1 = np.sin(theta) * x + np.cos(theta) * y1
        rot_y2 = np.sin(theta) * x + np.cos(theta) * y2

        # Translate the points back away from center 1, which was assumed to sit at (0, 0):
        int1 = (rot_x1 + center_1[0], rot_y1 + center_1[1])
        int2 = (rot_x2 + center_1[0], rot_y2 + center_1[1])

        return int1, int2

    @staticmethod
    def _get_tangent_circle(radius: float,
                            start_theta: float,
                            is_major: bool,
                            is_clockwise: bool) -> tuple[tuple[float, float], float]:
        """ This function determines the circle that is tangent to the point (0, 0) at an angle of start_theta,
        and tangent to the circle with center at (1, 0), with given radius:
        The point is at (0, 0)
        The center is at (1, 0)
        the start theta is between 0 and pi.
        Return the center and radius of this circle.
        """
        _theta = (start_theta - np.pi / 2.0 if is_major or (not is_major and not is_clockwise)
                  else start_theta + np.pi / 2.0)
        _rsign = -1.0 if is_major else 1.0

        arc_radius = 0.5 * (1.0 - radius ** 2.0) / (np.cos(_theta) + _rsign * radius)

        # arc center:
        arc_center = (arc_radius * np.cos(_theta), arc_radius * np.sin(_theta))

        return arc_center, np.abs(arc_radius)

    @staticmethod
    def _get_angle_between_points(point1: tuple[float, float],
                                  point2: tuple[float, float],
                                  is_degrees=True) -> float:
        """ return the angle formed with the x-axs when drawing a line STARTING at point 1 and traveling to point 2:
        """
        c1x, c1y = point1
        c2x, c2y = point2
        dx, dy = c2x - c1x, c2y - c1y

        if dx == 0:
            acute_theta = 3.0 * np.pi / 2.0

        else:
            acute_theta = np.abs(np.arctan(dy / dx))

        if dx < 0:
            if dy < 0:
                theta = np.pi + acute_theta

            else:
                theta = np.pi - acute_theta

        else:
            if dy < 0:
                theta = 2.0 * np.pi - acute_theta

            else:
                theta = acute_theta

        if is_degrees:
            theta *= 180.0 / np.pi

        return theta

    @staticmethod
    def _dist_xy(point1: tuple[float, float], point2: tuple[float, float]) -> float:
        return ((point1[0] - point2[0]) ** 2.0 + (point1[1] - point2[1]) ** 2.0) ** 0.5