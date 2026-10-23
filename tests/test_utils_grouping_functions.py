"""Tests for the get_neighbors function in spatial_fitting/grouping.py."""

import numpy as np

from gausspyplus.spatial_fitting.grouping import get_neighbors


def test_get_neighbors():
    expected_results = [
        np.array([[1, 1, 2], [0, 1, 1]]).T,
        np.array([[1, 1, 2, 2], [0, 1, 0, 1]]).T,
        np.array([3, 4, 6, 7]),
        #  default call: returns flat indices (return_indices=True); the
        #  central location (4) is excluded
        np.array([0, 1, 2, 3, 5, 6, 7, 8]),
        np.array([[0, 0, 0, 1, 1, 1, 2, 2, 2], [0, 1, 2, 0, 1, 2, 0, 1, 2]]).T,
        np.array([False, True, True, False, True, True, False, False, False]),
        np.array([[1, 2], [2, 2]]),
    ]
    actual_results = [
        get_neighbors((2, 0), shape=(3, 3), return_indices=False, return_coordinates=True),
        get_neighbors(
            (2, 0),
            exclude_location=False,
            shape=(3, 3),
            return_indices=False,
            return_coordinates=True,
        ),
        get_neighbors((2, 0), exclude_location=False, shape=(3, 3)),
        get_neighbors((1, 1), shape=(3, 3)),
        get_neighbors(
            (0, 0),
            exclude_location=False,
            shape=(3, 3),
            n_neighbors=2,
            return_indices=False,
            return_coordinates=True,
        ),
        get_neighbors(
            (2, 0),
            exclude_location=False,
            shape=(3, 3),
            return_indices=False,
            return_mask=True,
        ),
        get_neighbors(
            (2, 2),
            exclude_location=False,
            shape=(3, 3),
            direction="vertical",
            return_indices=False,
            return_coordinates=True,
        ),
    ]
    for expected, actual in zip(expected_results, actual_results):
        np.testing.assert_array_equal(np.asarray(expected), np.asarray(actual))
