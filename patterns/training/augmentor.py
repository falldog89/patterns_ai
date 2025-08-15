""" Class to exploit symmetries in patterns states and augment the training data.

A Patterns board has 8 dihedral group symmetries; bool flip and 4 rotations.

Further, the colors may be permuted, giving a further 6! = 720 equivalent states.

Therefore, each state can represent 720 * 8 = 5000 equivalent games (!)

We do not apply augmentations to permute colors for the first turns, as the game is restricted
to set up with colors 0 and 1 in the bowls, and 2-5 in the middle 4 locations.
"""

import numpy as np
import random
from typing import Optional

from patterns.constants import d4_permutation_indices
from patterns.constants import loci, locj


def augment(state: np.ndarray,
            visit_counts: Optional[np.ndarray] = None,
            ) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """ Apply a random symmetry and a random color permutation to the state. If provided,
    respect the symmetry augmentations in the visit counts array as well, as the actions represented
    refer to different board locations under symmetry.
    """
    # copy the arrays provided as augmentation will occur in place:
    aug_state = np.array(state)
    aug_counts = np.array(visit_counts) if visit_counts is not None else None

    # in place augmentation functions: we return indices and apply all at once at the end:
    _d4_permutation(aug_state, aug_counts)
    _color_permutation(aug_state)

    return aug_state, aug_counts

def _d4_permutation(state: np.ndarray, counts: Optional[np.ndarray]) -> None:
    """ choose a random dihedral permutation, and rearrange the visit counts and board
    accordingly:
    """
    # choose a member of the d4 group at random:
    permutation = random.choice(d4_permutation_indices)

    # Rearrange visit counts:
    if counts is not None:
        counts[:52] = counts[permutation]
        counts[52:104] = counts[permutation]

    perm_i = loci[permutation]
    perm_j = locj[permutation]

    state[:18, perm_i, perm_j] = state[:18, loci, locj]

def _color_permutation(state: np.ndarray) -> None:
    """ permute the colors so that all possible rearrangements of (0, 1, 2, 3, 4, 5) are
    equally likely
    """
    # randomly assign a permutation of the 6 colors:
    shuffled_colors = [0, 1, 2, 3, 4, 5]
    random.shuffle(shuffled_colors)
    full_index = (
        [_i for _i in shuffled_colors]
        + [_i + 6 for _i in shuffled_colors]
        + [_i + 12 for _i in shuffled_colors]
        + [_i + 18 for _i in shuffled_colors]
        + [_i + 24 for _i in shuffled_colors]
        + [_i + 30 for _i in shuffled_colors]
        + [_i + 36 for _i in shuffled_colors]
    )

    # each set of 6 slices is permuted in the same way:
    state[:42] = state[:42][full_index]
