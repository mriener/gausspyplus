"""Unit tests for the grouping functions in spatial_fitting/grouping.py.

Covers to_edges, to_graph, group_fit_solutions and determine_average_values;
the get_neighbors function is covered by tests/test_utils_grouping_functions.py.
"""

import collections

import numpy as np
from networkx.algorithms.components.connected import connected_components

from gausspyplus.spatial_fitting.grouping import (
    determine_average_values,
    group_fit_solutions,
    to_edges,
    to_graph,
)


def test_to_edges():
    assert list(to_edges(["a", "b", "c", "d"])) == [("a", "b"), ("b", "c"), ("c", "d")]
    #  a single element does not imply any edges
    assert list(to_edges([1])) == []
    assert list(to_edges([1, 2])) == [(1, 2)]


def test_to_graph():
    graph = to_graph([[1, 2], [2, 3], [10]])
    assert set(graph.nodes) == {1, 2, 3, 10}
    assert set(graph.edges) == {(1, 2), (2, 3)}
    assert sorted(graph.degree, key=lambda node_degree: node_degree[0]) == [(1, 1), (2, 2), (3, 1), (10, 0)]
    #  connected components merge all sublists that share common elements
    assert sorted(map(sorted, connected_components(graph))) == [[1, 2, 3], [10]]


def test_to_graph_empty_input():
    graph = to_graph([])
    assert graph.number_of_nodes() == 0
    assert graph.number_of_edges() == 0


def test_group_fit_solutions_means_only():
    """Grouping by mean positions only (split_fwhm=False)."""
    amps = np.array([1.0, 2.0, 3.0, 4.0])
    means = np.array([10.0, 11.0, 20.0, 50.0])
    fwhms = np.array([5.0, 5.0, 5.0, 5.0])
    result = group_fit_solutions(amps, means, fwhms, split_fwhm=False)
    assert isinstance(result, collections.OrderedDict)
    assert list(result.keys()) == ["1", "2", "3"]
    np.testing.assert_array_equal(result["1"]["amps"], [1.0, 2.0])
    np.testing.assert_array_equal(result["1"]["means"], [10.0, 11.0])
    np.testing.assert_array_equal(result["1"]["fwhms"], [5.0, 5.0])
    np.testing.assert_array_equal(result["2"]["amps"], [3.0])
    np.testing.assert_array_equal(result["2"]["means"], [20.0])
    np.testing.assert_array_equal(result["3"]["amps"], [4.0])
    np.testing.assert_array_equal(result["3"]["means"], [50.0])


def test_group_fit_solutions_mean_separation_strict():
    """A mean separation of exactly 'mean_separation' does not split the group (strict inequality)."""
    amps = np.array([1.0, 2.0])
    means = np.array([10.0, 12.0])
    fwhms = np.array([5.0, 5.0])
    result = group_fit_solutions(amps, means, fwhms, split_fwhm=False)
    assert list(result.keys()) == ["1"]
    np.testing.assert_array_equal(result["1"]["amps"], [1.0, 2.0])
    np.testing.assert_array_equal(result["1"]["means"], [10.0, 12.0])


def test_group_fit_solutions_split_fwhm_singletons():
    """Components with similar mean positions but dissimilar FWHM values are split up."""
    amps = np.array([1.0, 2.0, 3.0])
    means = np.array([10.0, 11.0, 12.0])
    fwhms = np.array([5.0, 50.0, 5.0])
    result = group_fit_solutions(amps, means, fwhms, mean_separation=2.0, fwhm_separation=4.0)
    assert list(result.keys()) == ["1", "2", "3"]
    for key, (amp, mean, fwhm) in zip(["1", "2", "3"], [(1.0, 10.0, 5.0), (2.0, 11.0, 50.0), (3.0, 12.0, 5.0)]):
        np.testing.assert_array_equal(result[key]["amps"], [amp])
        np.testing.assert_array_equal(result[key]["means"], [mean])
        np.testing.assert_array_equal(result[key]["fwhms"], [fwhm])


def test_group_fit_solutions_split_fwhm_merged():
    """Components with similar mean positions and similar FWHM values are grouped together."""
    amps = np.array([1.0, 2.0])
    means = np.array([10.0, 11.0])
    fwhms = np.array([5.0, 6.0])
    result = group_fit_solutions(amps, means, fwhms)
    assert list(result.keys()) == ["1"]
    np.testing.assert_array_equal(result["1"]["amps"], [1.0, 2.0])
    np.testing.assert_array_equal(result["1"]["means"], [10.0, 11.0])
    np.testing.assert_array_equal(result["1"]["fwhms"], [5.0, 6.0])
    #  an FWHM difference of exactly 'fwhm_separation' does not qualify for grouping
    fwhms = np.array([5.0, 9.0])
    result = group_fit_solutions(amps, means, fwhms, fwhm_separation=4.0)
    assert list(result.keys()) == ["1", "2"]
    np.testing.assert_array_equal(result["1"]["amps"], [1.0])
    np.testing.assert_array_equal(result["2"]["amps"], [2.0])
    #  ... whereas a larger 'fwhm_separation' groups them together
    result = group_fit_solutions(amps, means, fwhms, fwhm_separation=5.0)
    assert list(result.keys()) == ["1"]
    np.testing.assert_array_equal(result["1"]["amps"], [1.0, 2.0])


def test_group_fit_solutions_ordering_by_size():
    """Groups are renumbered according to the number of contained components (descending)."""
    amps = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    means = np.array([10.0, 10.5, 11.0, 30.0, 60.0, 60.5])
    fwhms = np.array([5.0, 5.0, 5.0, 5.0, 5.0, 5.0])
    result = group_fit_solutions(amps, means, fwhms, split_fwhm=False)
    assert list(result.keys()) == ["1", "2", "3"]
    #  largest group first
    np.testing.assert_array_equal(result["1"]["amps"], [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(result["1"]["means"], [10.0, 10.5, 11.0])
    #  second-largest group
    np.testing.assert_array_equal(result["2"]["amps"], [5.0, 6.0])
    np.testing.assert_array_equal(result["2"]["means"], [60.0, 60.5])
    #  smallest group last
    np.testing.assert_array_equal(result["3"]["amps"], [4.0])
    np.testing.assert_array_equal(result["3"]["means"], [30.0])


def test_group_fit_solutions_single_component():
    amps = np.array([1.0])
    means = np.array([10.0])
    fwhms = np.array([5.0])
    result = group_fit_solutions(amps, means, fwhms)
    assert list(result.keys()) == ["1"]
    np.testing.assert_array_equal(result["1"]["amps"], [1.0])
    np.testing.assert_array_equal(result["1"]["means"], [10.0])
    np.testing.assert_array_equal(result["1"]["fwhms"], [5.0])


def _fit_components(amps, means, fwhms):
    return collections.OrderedDict(
        {
            "1": {"amps": np.asarray(amps), "means": np.asarray(means), "fwhms": np.asarray(fwhms)},
        }
    )


def test_determine_average_values():
    intensity_values = np.full(30, 0.1)
    intensity_values[10] = 5.0
    fit_components = collections.OrderedDict(
        {
            #  group with two components: kept and enriched with average values
            "1": {
                "amps": np.array([2.0, 3.0]),
                "means": np.array([10.0, 11.0]),
                "fwhms": np.array([4.0, 5.0]),
            },
            #  group with a single component: removed
            "2": {"amps": np.array([4.0]), "means": np.array([20.0]), "fwhms": np.array([5.0])},
            #  group with a maximum amplitude below the intensity threshold: removed
            "3": {
                "amps": np.array([1.0, 2.0]),
                "means": np.array([24.0, 25.0]),
                "fwhms": np.array([4.0, 4.0]),
            },
        }
    )
    result = determine_average_values(intensity_values, fit_components, intensity_threshold=1.0)
    assert list(result.keys()) == ["1"]
    component = result["1"]
    assert component["amp_ini"] == 2.5
    assert component["mean_ini"] == 10.5
    assert component["fwhm_ini"] == 4.5
    #  upper limit for the amplitude is the maximum of the spectrum in the
    #  interval covered by the averaged component (channels 8 to 13)
    assert component["amp_bounds"] == [0.0, 1.1 * 5.0]
    #  mean bounds are at least 'mean_separation' channels away from the average
    assert component["mean_bounds"] == [8.5, 12.5]
    assert component["fwhm_bounds"] == [0, None]


def test_determine_average_values_constrain_fwhm():
    intensity_values = np.full(30, 0.1)
    intensity_values[10] = 5.0
    fit_components = _fit_components([2.0, 3.0], [10.0, 11.0], [4.0, 5.0])
    result = determine_average_values(intensity_values, fit_components, intensity_threshold=1.0, constrain_fwhm=True)
    #  fwhm bounds are at least 'fwhm_separation' channels away from the average
    assert result["1"]["fwhm_bounds"] == [0.5, 8.5]


def test_determine_average_values_mean_bound_clamped_to_zero():
    intensity_values = np.full(30, 0.1)
    intensity_values[1] = 3.0
    fit_components = _fit_components([2.0, 3.0], [0.5, 1.0], [4.0, 4.0])
    result = determine_average_values(intensity_values, fit_components, intensity_threshold=1.0)
    component = result["1"]
    assert component["amp_ini"] == 2.5
    assert component["mean_ini"] == 0.75
    assert component["fwhm_ini"] == 4.0
    #  lower mean bound is clamped to 0; upper bound is 'mean_separation' away
    assert component["mean_bounds"] == [0, 2.75]
    #  upper limit for the amplitude is taken from channels 0 to 3
    assert component["amp_bounds"] == [0.0, 1.1 * 3.0]
