"""pytest tests for module spatial_fitting.py (Phase 2 methods)"""
import os
import pickle
from pathlib import Path

from astropy.io import fits

import numpy as np

ROOT = Path(os.path.realpath(__file__)).parents[1]


# def test_spatial_fitting_phase_2():
#     from ..spatial_fitting.spatial_fitting import SpatialFitting
#     sp = SpatialFitting()
#     sp.decomposition = {'N_components': [3, 2, 2, 3, 3, 2, 1, 3, 3, 3, 2, 2, 3, 2, 3, 2, 3, 3, 2, 3, 3, 2, 3, 2, 3]}
#     sp.w_1 = 1/3
#     sp.w_2 = 1/6
#     indices_neighbors, chosen_weights = sp._get_weights(relative_positional_indices=np.array([1, 4]) - 2,
#                                                         direction='diagonal_ur')
#     print(indices_neighbors, chosen_weights)


if __name__ == "__main__":
    test_spatial_fitting_phase_2()
