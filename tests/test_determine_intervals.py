"""Tests for the determine_intervals.py module."""
import numpy as np

from gausspyplus.preparation.determine_intervals import (
    _add_buffer_to_intervals,
    check_if_intervals_contain_signal,
    get_noise_spike_ranges,
    get_signal_ranges,
    get_slice_indices_for_interval,
    indices_of_fit_components_in_interval,
    merge_overlapping_intervals,
)


def test_get_signal_ranges(grs_test_field):
    spectrum = grs_test_field[:, 31, 40]
    signal_ranges = get_signal_ranges(spectrum=spectrum, rms=0.1)
    assert signal_ranges == [[142, 207], [212, 252]]


def test_get_signal_ranges_without_padding(grs_test_field):
    # without padding, the raw intervals of the significant signal peaks are returned
    spectrum = grs_test_field[:, 31, 40]
    signal_ranges = get_signal_ranges(spectrum=spectrum, rms=0.1, pad_channels=0)
    assert signal_ranges == [[147, 202], [217, 247]]


def test_get_signal_ranges_with_remove_intervals(grs_test_field):
    # channels within the removed intervals are excluded from the signal ranges;
    # the remaining ranges are padded until min_channels channels are contained
    spectrum = grs_test_field[:, 31, 40]
    signal_ranges = get_signal_ranges(spectrum=spectrum, rms=0.1, remove_intervals=[[150, 220]])
    assert signal_ranges == [[112, 150], [220, 282]]


def test_get_slice_indices_for_interval():
    assert get_slice_indices_for_interval(50, 5) == (45, 57)
    # the lower bound is clipped at 0
    assert get_slice_indices_for_interval(2, 5) == (0, 9)
    assert get_slice_indices_for_interval(0, 5) == (0, 7)
    # non-integer interval centers are truncated
    assert get_slice_indices_for_interval(50.7, 5.2) == (45, 57)


def test_merge_overlapping_intervals():
    # note: the input list is sorted in place
    assert merge_overlapping_intervals([[5, 10], [1, 3], [12, 15], [2, 6]]) == [[1, 10], [12, 15]]
    # fully contained intervals are absorbed
    assert merge_overlapping_intervals([[1, 10], [2, 3]]) == [[1, 10]]
    # adjacent intervals (shared boundary) are merged as well
    assert merge_overlapping_intervals([[1, 5], [5, 10]]) == [[1, 10]]
    assert merge_overlapping_intervals([]) == []


def test_add_buffer_to_intervals():
    # intervals are padded and overlapping padded intervals are merged
    assert _add_buffer_to_intervals([[10, 20], [22, 30]], n_channels=100, pad_channels=5) == [[5, 35]]
    # the padded intervals are clipped to the spectrum boundaries
    assert _add_buffer_to_intervals([[0, 3], [95, 99]], n_channels=100, pad_channels=5) == [[0, 8], [90, 100]]
    assert _add_buffer_to_intervals([], n_channels=100) == []


def test_check_if_intervals_contain_signal():
    spectrum = np.zeros(100)
    spectrum[45:55] = 10.0
    # the off-signal interval [70, 80] is removed
    assert check_if_intervals_contain_signal(spectrum, rms=1.0, ranges=[[40, 60], [70, 80]]) == [[40, 60]]
    spectrum[75] = 4.0
    # the narrow spike in [70, 80] exceeds snr * rms, but fails the significance criterion
    assert check_if_intervals_contain_signal(spectrum, rms=1.0, ranges=[[40, 60], [70, 80]]) == [[40, 60]]


def test_check_if_intervals_contain_signal_grs(grs_test_field):
    spectrum = grs_test_field[:, 31, 40]
    # the signal intervals are kept; the off-signal interval [0, 50] is removed
    ranges = check_if_intervals_contain_signal(spectrum, rms=0.1, ranges=[[142, 207], [212, 252], [0, 50]])
    assert ranges == [[142, 207], [212, 252]]


def test_get_noise_spike_ranges():
    spectrum = np.zeros(100)
    spectrum[30] = -1.0
    assert get_noise_spike_ranges(spectrum, rms=0.1) == [[30, 31]]
    # the spike does not exceed the required signal-to-noise ratio
    assert get_noise_spike_ranges(spectrum, rms=0.1, snr_noise_spike=15.0) == []


def test_get_noise_spike_ranges_grs(grs_test_field):
    # characterization test: the 16 ranges below were recorded from the current
    # implementation on the reference stack (regression detection, not
    # independent ground truth)
    # the established test spectrum contains no significant negative noise spikes
    spectrum = grs_test_field[:, 31, 40]
    assert get_noise_spike_ranges(spectrum, rms=0.1) == []
    # the spectrum at (y=0, x=0) contains several significant negative noise spikes
    spectrum = grs_test_field[:, 0, 0]
    ranges = get_noise_spike_ranges(spectrum, rms=0.1)
    # properties independent of the recorded values: the ranges are sorted and
    # non-overlapping, and each contains at least one channel below the
    # noise-spike threshold of -snr_noise_spike * rms
    assert all(ranges[i + 1][0] >= ranges[i][1] for i in range(len(ranges) - 1))
    assert all(np.min(spectrum[start:end]) <= -5.0 * 0.1 for start, end in ranges)
    assert ranges == [
        [28, 29],
        [39, 45],
        [48, 52],
        [77, 81],
        [84, 86],
        [117, 122],
        [127, 130],
        [132, 137],
        [154, 156],
        [269, 271],
        [281, 284],
        [288, 290],
        [322, 327],
        [333, 338],
        [348, 351],
        [388, 395],
    ]


def test_indices_of_fit_components_in_interval():
    # component 0 (mean=50, fwhm=4) lies fully within the interval; component 1 (mean=80) lies outside
    indices, interval_new = indices_of_fit_components_in_interval(fwhms=[4, 20], means=[50, 80], interval=[45, 55])
    assert indices == [0]
    assert interval_new == [45, 55]
    # component 0 (mean=50, fwhm=10) overlaps the interval with its upper bound;
    # the interval is expanded to contain the full extent of the component
    indices, interval_new = indices_of_fit_components_in_interval(fwhms=[10], means=[50], interval=[45, 65])
    assert indices == [0]
    assert interval_new == [40, 65]
    # the lower bound of a component is clipped at channel 0
    indices, interval_new = indices_of_fit_components_in_interval(fwhms=[10], means=[3], interval=[0, 5])
    assert indices == [0]
    assert interval_new == [0, 13]


def test_indices_of_fit_components_in_interval_no_match():
    # neither the lower nor the upper bound of the component lies within the interval
    indices, interval_new = indices_of_fit_components_in_interval(fwhms=[10], means=[50], interval=[45, 55])
    assert indices == []
    assert interval_new == [45, 55]
