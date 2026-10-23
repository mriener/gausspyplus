"""Tests for the remove_elements_at_indices function in utils/misc.py."""
import numpy as np
import pytest

from gausspyplus.utils.misc import remove_elements_at_indices


def test_remove_elements_at_indices():
    assert remove_elements_at_indices([1, 2, 3, 4, 5], [1, 3]) == [1, 3, 5]
    # a scalar index is accepted as well
    assert remove_elements_at_indices([1, 2, 3, 4, 5], 0) == [2, 3, 4, 5]
    # numpy arrays are accepted as input; the result is always a plain list
    assert remove_elements_at_indices(np.array([1, 2, 3, 4]), [1]) == [1, 3, 4]


def test_remove_elements_at_indices_edge_cases():
    # removing no elements returns all elements
    assert remove_elements_at_indices([1, 2, 3], []) == [1, 2, 3]
    # removing all elements returns an empty list
    assert remove_elements_at_indices([1, 2], [0, 1]) == []


def test_remove_elements_at_indices_multiple_subarrays():
    # the array is split into n_subarrays and the indices are applied to each subarray
    assert remove_elements_at_indices([1, 2, 3, 4], [0], n_subarrays=2) == [2, 4]
    assert remove_elements_at_indices(list(range(8)), [0, 2], n_subarrays=2) == [1, 3, 5, 7]


def test_remove_elements_at_indices_unequal_split_raises():
    # np.split requires the array to be divisible into n_subarrays equal parts
    with pytest.raises(ValueError):
        remove_elements_at_indices([1, 2, 3, 4, 5], [0], n_subarrays=2)
