"""Unit tests for the functions in spatial_fitting/ndimage_functions.py."""
import numpy as np

from gausspyplus.spatial_fitting.ndimage_functions import (
    broad_components,
    number_of_component_jumps,
    weighted_median,
)


def test_weighted_median_all_neighbors_valid():
    #  central value (index 4) is excluded; with all 8 neighbors valid the
    #  cumulative weight hits the midpoint exactly between the two central
    #  neighbors, so their mean is returned
    data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=float)
    assert weighted_median(data) == 5.0


def test_weighted_median_unsorted_neighbors_strict_crossing():
    #  same set of neighbor values as above but shuffled such that the sorted
    #  cumulative weights cross the midpoint strictly between two neighbors
    data = np.array([9, 1, 8, 2, 5, 3, 7, 4, 6], dtype=float)
    assert weighted_median(data) == 4.0


def test_weighted_median_two_equal_weights():
    #  two direct neighbors (weight 1 each): the midpoint is hit exactly, so the
    #  mean of the two values is returned (even-length behaviour)
    data = np.array([np.nan, np.nan, np.nan, 10, 5, 30, np.nan, np.nan, np.nan])
    assert weighted_median(data) == 20.0


def test_weighted_median_two_different_weights():
    #  diagonal neighbor (weight 1/sqrt(2)) and direct neighbor (weight 1):
    #  the direct neighbor carries more than half of the total weight
    data = np.array([np.nan, np.nan, np.nan, np.nan, 5, np.nan, 10, 20, np.nan])
    assert weighted_median(data) == 20.0


def test_weighted_median_four_direct_neighbors():
    #  four direct neighbors (weight 1 each): mean of the two central values
    data = np.array([np.nan, 10, np.nan, 20, 5, 30, np.nan, 40, np.nan])
    assert weighted_median(data) == 25.0


def test_weighted_median_four_diagonal_neighbors():
    #  four diagonal neighbors (weight 1/sqrt(2) each): mean of the two central
    #  values; the midpoint is hit exactly (2 * 1/sqrt(2) == half of 4 * 1/sqrt(2))
    data = np.array([10, np.nan, 20, np.nan, 5, np.nan, 30, np.nan, 40])
    assert weighted_median(data) == 25.0


def test_weighted_median_single_valid_neighbor():
    #  only the direct neighbor to the left is available
    data = np.array([np.nan, np.nan, np.nan, 7, 5, np.nan, np.nan, np.nan, np.nan])
    assert weighted_median(data) == 7.0


def test_weighted_median_masked_central_spectrum():
    #  central spectrum masked out
    data = np.array([1, 2, 3, 4, np.nan, 6, 7, 8, 9])
    assert weighted_median(data) == 0


def test_weighted_median_no_valid_neighbors():
    #  central spectrum valid but all neighbors masked out
    data = np.array([np.nan, np.nan, np.nan, np.nan, 5, np.nan, np.nan, np.nan, np.nan])
    assert weighted_median(data) == 0


def test_number_of_component_jumps():
    #  central spectrum (index 4) has 3 components; neighbors with 1, 5 and 10
    #  components differ by more than the maximum allowed jump of 1
    values = np.array([3, 3, 3, 3, 3, 1, 3, 5, 10])
    assert number_of_component_jumps(values, max_jump_comps=1) == 3
    #  masked-out neighbors do not count towards the jumps
    values = np.array([3, 3, 3, 3, 3, 1, np.nan, 5, 10])
    assert number_of_component_jumps(values, max_jump_comps=1) == 3
    #  a difference of exactly 'max_jump_comps' is not a jump (strict inequality)
    values = np.array([3, 3, 3, 3, 3, 4, 3, 2, 10])
    assert number_of_component_jumps(values, max_jump_comps=1) == 1
    #  larger allowed jump value
    values = np.array([3, 3, 3, 3, 3, 1, 3, 5, 10])
    assert number_of_component_jumps(values, max_jump_comps=2) == 1


def test_number_of_component_jumps_masked_central_spectrum():
    values = np.array([3, 3, 3, 3, np.nan, 1, 3, 5, 10])
    assert number_of_component_jumps(values, max_jump_comps=1) == 0
    #  central spectrum valid but all neighbors masked out
    values = np.array([np.nan, np.nan, np.nan, np.nan, 3, np.nan, np.nan, np.nan, np.nan])
    assert number_of_component_jumps(values, max_jump_comps=1) == 0


def test_broad_components():
    #  central FWHM (index 4) exceeds all neighbors by more than a factor of 2
    #  and by more than the FWHM separation
    values = np.array([5, 5, 5, 5, 20, 5, 5, 5, 5], dtype=float)
    assert broad_components(values, fwhm_factor=2.0, fwhm_separation=4.0, broad_neighbor_fraction=0.5) == 20.0
    #  only 4 of 8 neighbors are sufficiently narrower; 4 is not more than half
    values = np.array([5, 5, 5, 5, 20, 20, 20, 20, 20], dtype=float)
    assert broad_components(values, fwhm_factor=2.0, fwhm_separation=4.0, broad_neighbor_fraction=0.5) == 0
    #  ... but 4 out of 8 exceeds a fraction of 0.4
    assert broad_components(values, fwhm_factor=2.0, fwhm_separation=4.0, broad_neighbor_fraction=0.4) == 20.0
    #  a factor of exactly 'fwhm_factor' does not qualify (strict inequality)
    values = np.array([5, 5, 5, 5, 10, 5, 5, 5, 5], dtype=float)
    assert broad_components(values, fwhm_factor=2.0, fwhm_separation=4.0, broad_neighbor_fraction=0.5) == 0
    #  a difference of exactly 'fwhm_separation' does not qualify (strict inequality)
    values = np.array([4, 4, 4, 4, 10, 4, 4, 4, 4], dtype=float)
    assert broad_components(values, fwhm_factor=2.0, fwhm_separation=6.0, broad_neighbor_fraction=0.5) == 0
    #  masked-out neighbors are skipped and do not count towards the fraction
    values = np.array([5, np.nan, 5, 5, 20, 5, np.nan, 5, 5], dtype=float)
    assert broad_components(values, fwhm_factor=2.0, fwhm_separation=4.0, broad_neighbor_fraction=0.5) == 20.0


def test_broad_components_masked_central_spectrum():
    values = np.array([5, 5, 5, 5, np.nan, 5, 5, 5, 5], dtype=float)
    assert broad_components(values, fwhm_factor=2.0, fwhm_separation=4.0, broad_neighbor_fraction=0.5) == 0
    #  central spectrum valid but all neighbors masked out
    values = np.array([np.nan, np.nan, np.nan, np.nan, 20, np.nan, np.nan, np.nan, np.nan], dtype=float)
    assert broad_components(values, fwhm_factor=2.0, fwhm_separation=4.0, broad_neighbor_fraction=0.5) == 0
