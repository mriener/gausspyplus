"""Tests for the pure helper functions in training/gradient_descent.py."""

import numpy as np
import pytest

from gausspyplus.training.gradient_descent import GradientDescent, compare_parameters, count_ones_in_row


def test_count_ones_in_row():
    np.testing.assert_array_equal(count_ones_in_row([0, 1, 1, 1, 0, 1]), [0, 1, 2, 3, 0, 1])
    # runs of '1's that reach back to the first element of the array are counted correctly
    np.testing.assert_array_equal(count_ones_in_row([0, 1, 1]), [0, 1, 2])
    np.testing.assert_array_equal(count_ones_in_row([0, 1]), [0, 1])


def test_count_ones_in_row_edge_cases():
    np.testing.assert_array_equal(count_ones_in_row([0, 0, 0]), [0, 0, 0])
    np.testing.assert_array_equal(count_ones_in_row([]), [])


@pytest.mark.xfail(
    reason="known bug: count_ones_in_row accesses a negative index for i=0, which wraps around to the "
    "end of the array and counts trailing '1's that are not part of a continuous run",
    strict=True,
)
def test_count_ones_in_row_leading_ones():
    # the first element is its own run: a single leading '1' counts as 1, regardless of the array tail
    np.testing.assert_array_equal(count_ones_in_row([1, 0, 0, 1]), [1, 0, 0, 1])
    np.testing.assert_array_equal(count_ones_in_row([1]), [1])
    np.testing.assert_array_equal(count_ones_in_row([1, 1]), [1, 2])


def test_compare_parameters_correct_match():
    # guess parameters identical to the true parameters
    assert compare_parameters([1, 10, 50], [1, 10, 50]) == (1.0, 1, 1)
    # two guesses and two true components that match with swapped mean positions
    assert compare_parameters([1, 1, 10, 10, 80, 50], [1, 1, 10, 10, 50, 80]) == (2.0, 2, 2)


def test_compare_parameters_no_match():
    # mean position too far away (more than 1 sigma)
    assert compare_parameters([1, 10, 60], [1, 10, 50]) == (0.0, 1, 1)
    # FWHM outside the allowed range (0.3 x - 2.5 x the true FWHM)
    assert compare_parameters([1, 30, 50], [1, 10, 50]) == (0.0, 1, 1)
    # non-positive amplitude guesses are never matched
    assert compare_parameters([0, 10, 50], [1, 10, 50]) == (0.0, 1, 1)


def test_compare_parameters_unique_matches():
    # a true component can only be matched by a single guess component
    # (two identical guesses for the same true component count only once)
    assert compare_parameters([1, 1, 10, 10, 50, 50], [1, 1, 10, 10, 50, 80]) == (1.0, 2, 2)


def test_gradient_descent_bookkeeping():
    gd = GradientDescent(iterations=3)
    assert gd.alpha1_trace.shape == (4,)
    assert gd.alpha2_trace.shape == (4,)
    assert gd.accuracy_trace.shape == (3,)
    assert gd.D_alpha1_trace.shape == (3,)
    assert gd.D_alpha2_trace.shape == (3,)
    assert np.isnan(gd.iter_of_convergence)
    assert np.all(np.isnan(gd.alpha1_trace))
