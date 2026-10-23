"""Unit tests for phase 2 methods (continuity of centroid positions) of the SpatialFitting class.

The heavy orchestration methods (_check_continuity, refit_spectrum_phase_2, _refitting) are
covered end-to-end by tests/test_pipeline.py (stage 5); this file covers the decision logic.
"""
import numpy as np

from gausspyplus.spatial_fitting.spatial_fitting import SpatialFitting

#  Default weights corresponding to weight_factor=2 (see SpatialFitting set-up routine):
#  w_1 for immediate neighbors, w_2 for neighbors two pixels away.
W_1 = 1 / 3
W_2 = 1 / 6


def make_spatial_fitting(**attributes):
    sp = SpatialFitting()
    sp.w_1 = W_1
    sp.w_2 = W_2
    for key, value in attributes.items():
        setattr(sp, key, value)
    return sp


def test_weights():
    """The weights property assigns neighbor weights by relative position and direction."""
    sp = make_spatial_fitting()
    np.testing.assert_allclose(list(sp.weights["horizontal"].values()), [W_2, W_1, W_1, W_2])
    np.testing.assert_allclose(list(sp.weights["vertical"].values()), [W_2, W_1, W_1, W_2])
    # diagonal neighbors carry a reduced weight (w / sqrt(2) for immediate, w / sqrt(8) for distant)
    np.testing.assert_allclose(
        list(sp.weights["diagonal_ul"].values()),
        [W_2 / np.sqrt(8), W_1 / np.sqrt(2), W_1 / np.sqrt(2), W_2 / np.sqrt(8)],
    )
    np.testing.assert_allclose(
        list(sp.weights["diagonal_ur"].values()),
        [W_2 / np.sqrt(8), W_1 / np.sqrt(2), W_1 / np.sqrt(2), W_2 / np.sqrt(8)],
    )


def test_get_n_centroid():
    """Expected number of centroids per interval: majority vote of immediate neighbors,
    falling back to the weighted sum over all neighbors."""
    sp = make_spatial_fitting()
    # majority vote among the immediate neighbors (weight w_1): two neighbors with one
    # component outweigh one distant neighbor (weight w_2) with two components
    assert sp._get_n_centroid(np.array([1, 1, 2]), np.array([W_1, W_1, W_2])) == 1
    # no majority among immediate neighbors -> weighted sum over all neighbors
    # (choice 1: w_1 = 1/3; choice 2: w_1 + w_2 = 1/2)
    assert sp._get_n_centroid(np.array([1, 2, 2]), np.array([W_1, W_1, W_2])) == 2
    # no neighbor has a component in the interval
    assert sp._get_n_centroid(np.array([0, 0]), np.array([W_1, W_1])) == 0


def test_stopping_criterion():
    """Refitting iterations stop when the maximum is reached, the same spectra would be
    refit again, or the number of spectra to refit did not decrease."""
    sp = make_spatial_fitting()
    sp.max_refitting_iteration = 5
    sp.refitting_iteration = 5
    assert sp._stopping_criterion([1, 2]) is True

    sp.refitting_iteration = 1
    sp.list_n_refit = [[1, 2], [3, 4]]
    # the same number of spectra would be refit as in a previous iteration
    assert sp._stopping_criterion([1, 2]) is True
    # all counts are >= the minimum of the previous iterations
    assert sp._stopping_criterion([3, 5]) is True
    # a count below the minimum of the previous iterations -> continue refitting
    assert sp._stopping_criterion([0, 5]) is False

    # no criterion matched: the method falls through without an explicit return
    # statement and returns None (falsy for the caller, but an implicit API smell)
    sp.refitting_iteration = 0
    sp.list_n_refit = []
    assert sp._stopping_criterion([1, 2]) is None


def test_get_indices_and_weights_of_valid_neighbors():
    """Neighbors without fit components are excluded; weights follow the relative position."""
    sp = make_spatial_fitting()
    sp.shape = (1, 5)
    # neighbor at index 3 has no fit components and must be excluded
    sp.decomposition = {"N_components": [1, 2, 3, 0, 1]}
    indices, weights = sp._get_indices_and_weights_of_valid_neighbors(loc=(0, 2), idx=2, direction="horizontal")
    np.testing.assert_array_equal(indices, [0, 1, 4])
    np.testing.assert_allclose(weights, [W_2, W_1, W_2])

    # no valid neighbor -> empty results
    sp.decomposition = {"N_components": [0, 0, 3, 0, 0]}
    indices, weights = sp._get_indices_and_weights_of_valid_neighbors(loc=(0, 2), idx=2, direction="horizontal")
    assert indices.size == 0
    assert weights.size == 0


def test_get_indices_and_weights_of_valid_neighbors_diagonal():
    sp = make_spatial_fitting()
    sp.shape = (3, 3)
    sp.decomposition = {"N_components": [1, 1, 1, 1, 1, 1, 1, 1, 1]}
    indices, weights = sp._get_indices_and_weights_of_valid_neighbors(loc=(1, 1), idx=4, direction="diagonal_ur")
    # diagonal neighbors of the central spectrum (index 4) in a 3x3 grid: indices 2 and 6
    np.testing.assert_array_equal(indices, [2, 6])
    np.testing.assert_allclose(weights, [W_1 / np.sqrt(2), W_1 / np.sqrt(2)])


def test_check_continuity_centroids():
    """Spatial coherence check of centroid positions on a 1d row of three spectra."""
    sp = make_spatial_fitting(mean_separation=8.0, fwhm_separation=4.0, min_p=0.2)
    sp.shape = (1, 3)
    sp.decomposition = {
        "N_components": [1, 2, 1],
        "amplitudes_fit": [[1.0], [1.5, 0.5], [1.2]],
        "means_fit": [[10.0], [28.0, 11.0], [12.0]],
        "fwhms_fit": [[5.0], [5.0, 6.0], [5.0]],
    }
    checks = sp._check_continuity_centroids(idx=1, loc=(0, 1))

    # only the horizontal direction has valid neighbors in a single-row grid
    np.testing.assert_array_equal(checks["indices_neighbors"], [0, 2])
    np.testing.assert_allclose(checks["weights"], [W_1, W_1])
    # the neighbors' centroid positions (10, 12) form one group; its interval
    # [10 - mean_separation/2, 12 + mean_separation/2] = [6, 16] is extended by
    # mean_separation/2 again in the combination of the directions -> [2, 20]
    assert checks["means_interval"] == {"1": [2, 20]}
    # each neighbor has one centroid position inside the interval [2, 20]
    assert checks["n_comps"] == {"1": [1, 1]}
    # both immediate neighbors agree on one centroid -> expected number of centroids is 1
    assert checks["n_centroids"] == {"1": 1}


def test_check_for_required_components():
    """A spectrum is flagged for refitting if its number of centroids inside an interval
    differs from the expected number of centroids."""
    sp = make_spatial_fitting()
    sp.decomposition = {"means_fit": [[10.0], [10.0, 30.0], [12.0]]}
    spatial_coherence_checks = {
        "indices_neighbors": [0, 2],
        "weights": [W_1, W_1],
        "means_interval": {"1": [2, 20]},
        "n_centroids": {"1": 1},
    }
    # the central spectrum has exactly one centroid position inside the interval -> no refit
    checks = sp._check_for_required_components(idx=1, spatial_coherence_checks=spatial_coherence_checks)
    assert checks["means_interval"] == {}
    assert checks["n_centroids"] == {}

    # the central spectrum has two centroid positions inside the interval -> refit flagged
    sp.decomposition = {"means_fit": [[10.0], [10.0, 15.0], [12.0]]}
    checks = sp._check_for_required_components(idx=1, spatial_coherence_checks=spatial_coherence_checks)
    assert checks["means_interval"] == {"1": [2, 20]}
    assert checks["n_centroids"] == {"1": 1}
