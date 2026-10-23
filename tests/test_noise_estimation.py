"""Tests for the noise_estimation.py module."""

import itertools
import random

import numpy as np
import pytest

from gausspyplus.preparation.noise_estimation import (
    calculate_average_rms_noise,
    determine_maximum_consecutive_channels,
    determine_noise,
    determine_peaks,
    intervals_where_mask_is_true,
    mask_channels,
    pad_intervals,
)

RMS = 0.10634302494716603  # noise level of the test spectrum at position (31, 40); see test_get_rms_noise


def test_determine_maximum_consecutive_channels():
    # characterization test (deterministic function; any change here indicates a bug, not dependency drift)
    assert determine_maximum_consecutive_channels(100, 0.05) == 11
    assert determine_maximum_consecutive_channels(200, 0.02) == 14
    assert determine_maximum_consecutive_channels(1000, 0.01) == 17


def test_get_rms_noise(grs_test_field):
    # characterization test (deterministic function; any change here indicates a bug, not dependency drift)
    spectrum = grs_test_field[:, 31, 40]
    rms_noise = determine_noise(spectrum)
    assert np.allclose(rms_noise, RMS)


def test_intervals_where_mask_is_true(grs_test_field):
    # characterization test (deterministic function; any change here indicates a bug, not dependency drift)
    spectrum = grs_test_field[:, 31, 40]
    assert intervals_where_mask_is_true(spectrum > 0.5) == [
        [180, 192],
        [227, 229],
        [230, 238],
    ]


def test_determine_peaks():
    spectrum = np.array([0.0, 2.0, 5.0, 2.0, 0.0, -1.0, -4.0, 0.0])
    amp_vals, ranges = determine_peaks(spectrum, peak="positive", amp_threshold=1.0)
    np.testing.assert_allclose(amp_vals, [5.0])
    np.testing.assert_array_equal(ranges, [[1, 4]])
    amp_vals, ranges = determine_peaks(spectrum, peak="negative", amp_threshold=1.0)
    np.testing.assert_allclose(amp_vals, [-4.0])
    np.testing.assert_array_equal(ranges, [[5, 7]])


def test_determine_peaks_threshold_filtering():
    spectrum = np.array([0.0, 2.0, 5.0, 2.0, 0.0, -1.0, -4.0, 0.0])
    # no peak feature exceeds the threshold
    amp_vals, ranges = determine_peaks(spectrum, peak="positive", amp_threshold=6.0)
    assert len(amp_vals) == 0
    assert len(ranges) == 0
    # without a threshold all positive peak intervals are returned
    amp_vals, ranges = determine_peaks(spectrum, peak="positive")
    np.testing.assert_allclose(amp_vals, [5.0])
    np.testing.assert_array_equal(ranges, [[1, 4]])


def test_determine_peaks_grs(grs_test_field):
    # characterization test (deterministic function; any change here indicates a bug, not dependency drift):
    # the peak amplitudes and ranges below were recorded from the current implementation on the reference stack
    spectrum = grs_test_field[:, 31, 40]
    amp_vals, ranges = determine_peaks(spectrum, peak="positive", amp_threshold=0.3)
    # properties independent of the recorded values: each amplitude is the
    # maximum of the spectrum within its range and exceeds the threshold, and
    # the ranges are sorted and non-overlapping
    for amp, (start, end) in zip(amp_vals, ranges):
        assert spectrum[start:end].max() == amp
        assert amp > 0.3
    assert all(ranges[i + 1][0] >= ranges[i][1] for i in range(len(ranges) - 1))
    np.testing.assert_allclose(amp_vals, [0.8587118, 0.7724102], rtol=1e-6)
    np.testing.assert_array_equal(ranges, [[147, 202], [217, 247]])
    # the spectrum contains no negative peak that exceeds 3 * rms
    amp_vals, ranges = determine_peaks(spectrum, peak="negative", amp_threshold=0.3)
    assert len(amp_vals) == 0
    assert len(ranges) == 0


def test_pad_intervals():
    assert pad_intervals([[10, 20]], pad_channels=5, upper_limit=100) == [[5, 25]]
    # the padded intervals are clipped to the given limits
    assert pad_intervals([[1, 98]], pad_channels=5, upper_limit=100) == [[0, 100]]
    assert pad_intervals([[12, 20]], pad_channels=5, lower_limit=10, upper_limit=100) == [[10, 25]]
    # for pad_channels=None the intervals are returned unchanged
    intervals = [[10, 20]]
    assert pad_intervals(intervals, pad_channels=None) is intervals


def test_mask_channels():
    mask = mask_channels(n_channels=20, ranges=[[5, 10], [15, 16]])
    np.testing.assert_array_equal(mask, [False] * 5 + [True] * 5 + [False] * 5 + [True] + [False] * 4)
    # pad_channels extend the ranges before the mask is built
    mask = mask_channels(n_channels=20, ranges=[[5, 10], [15, 16]], pad_channels=2)
    np.testing.assert_array_equal(mask, [False] * 3 + [True] * 9 + [False] + [True] * 5 + [False] * 2)
    # remove_intervals are excluded from the mask
    mask = mask_channels(n_channels=20, ranges=[[5, 10], [15, 16]], remove_intervals=[[8, 10]])
    np.testing.assert_array_equal(mask, [False] * 5 + [True] * 3 + [False] * 7 + [True] + [False] * 4)


def test_determine_noise_edge_cases():
    # a spectrum containing only NaNs is masked out
    assert np.isnan(determine_noise(np.full(10, np.nan)))
    # a spectrum containing only values >= 0 is masked out
    assert np.isnan(determine_noise(np.abs(np.random.RandomState(1).normal(0, 1, 100))))


def test_calculate_average_rms_noise(grs_test_field):
    # characterization test (deterministic given the fixed random seed; any change here indicates a bug,
    # not dependency drift): value recorded from the current implementation on the reference stack
    assert np.isclose(calculate_average_rms_noise(grs_test_field, 10), 0.09361524408773854)


def test_calculate_average_rms_noise_samples_requested_number_of_spectra():
    # cube with four noise spectra of very different rms and four spectra that are all NaN
    rng = np.random.RandomState(0)
    n_channels = 100
    cube = np.empty((n_channels, 1, 8))
    for x, sigma in enumerate([0.1, 0.2, 0.4, 0.8]):
        cube[:, 0, x] = rng.normal(0.0, sigma, n_channels)
    cube[:, 0, 4:] = np.nan

    number_rms_spectra = 3
    random_seed = 1
    actual = calculate_average_rms_noise(cube, number_rms_spectra, random_seed=random_seed)

    # reference implementation of the intended behaviour: sample number_rms_spectra locations.
    # With random_seed=1 the sampled locations include one of the all-NaN spectra, so the
    # average is computed from only two valid rms values.
    random.seed(random_seed)
    locations = list(itertools.product(np.arange(cube.shape[1]), np.arange(cube.shape[2])))
    sampled_locations = random.sample(locations, number_rms_spectra)
    rms_values = [
        determine_noise(cube[:, y, x], max_consecutive_channels=14, pad_channels=5) for y, x in sampled_locations
    ]
    expected = np.nanmean(rms_values)
    assert np.isclose(actual, expected)
