"""Functions for interval determination."""
import itertools
import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

import sys

ROOT = Path(os.path.realpath(__file__)).parents[2]
sys.path.append(str(ROOT))

from gausspyplus.preparation.noise_estimation import (
    determine_peaks,
    mask_channels,
    intervals_where_mask_is_true,
    pad_intervals,
)


def get_slice_indices_for_interval(interval_center, interval_half_width):
    return (
        max(0, int(interval_center - interval_half_width)),  # index for lower bound
        int(interval_center + interval_half_width) + 2,  # index for upper bound
    )


def merge_overlapping_intervals(intervals: List[List]) -> List[List]:
    """Merge overlapping intervals (Credit: https://stackoverflow.com/a/43600953)."""
    intervals.sort(key=lambda interval: interval[0])
    try:
        merged_intervals = [intervals[0]]
    except IndexError:  # in case intervals is an empty list
        merged_intervals = []
    for current in intervals:
        previous = merged_intervals[-1]
        # test for intersection between previous and current: we know via sorting that previous[0] <= current[0]
        if current[0] <= previous[1]:
            previous[1] = max(previous[1], current[1])
        else:
            merged_intervals.append(current)
    return merged_intervals


def _add_buffer_to_intervals(
    ranges: List[Optional[List]], n_channels: int, pad_channels: int = 5
) -> List[Optional[List]]:
    """Pad intervals on lower and upper sides and merge overlapping intervals."""
    # TODO: this needs to be tested and compared time-wise with the previous
    if not ranges:  # in case ranges is an empty list
        return ranges
    intervals = pad_intervals(intervals=ranges, pad_channels=pad_channels, upper_limit=n_channels)
    return merge_overlapping_intervals(intervals)


def check_if_intervals_contain_signal(
    spectrum: np.ndarray,
    rms: float,
    ranges: List[Optional[List]],
    snr: float = 3.0,
    significance: float = 5.0,
) -> List[Optional[List]]:
    """Check if selected intervals contain significant positive signal peaks.

    If the maximum intensity value of an interval (low, upp) is smaller than snr * rms, the interval gets removed
    from 'ranges'. The required minimum significance threshold helps to remove narrow noise spikes or insignificant
    positive intensity peaks.

    Parameters
    ----------
    spectrum : Intensity values of the spectrum.
    rms : Root-mean-square noise of the spectrum.
    ranges : List of intervals [(lower, upper), ...] that were identified as containing positive signal.
    snr : Required minimum signal-to-noise ratio for data peak.
    significance : Required minimum value for significance criterion.

    Returns
    -------
    Updated intervals [(low, upp), ...] that contain only significant positive signal peaks.

    """
    # TODO: ranges should be np.ndarray
    # TODO: rename this function (function is also used in gp_plus, where it is used for a conditional check)
    intervals = [[lower, upper] for lower, upper in ranges if np.max(spectrum[lower:upper]) > snr * rms]
    return [
        [lower, upper]
        for lower, upper in intervals
        if np.sum(spectrum[lower:upper]) / (np.sqrt(upper - lower) * rms) > significance
    ]


def get_signal_ranges(
    spectrum: np.ndarray,
    rms: float,
    pad_channels: int = 5,
    snr: float = 3.0,
    significance: float = 5.0,
    min_channels: int = 100,
    remove_intervals: Optional[List[Optional[List]]] = None,
) -> List[Optional[List]]:
    """Determine ranges in the spectrum that could contain signal.

    Parameters
    ----------
    spectrum : Intensity values of the spectrum.
    rms : Root-mean-square noise of the spectrum.
    pad_channels : Number of additional channels that get masked out on both sides of an identified (signal?) feature.
    snr : Required minimum signal-to-noise ratio for data peak.
    significance : Required minimum value for significance criterion.
    min_channels : Required minimum number of spectral channels that the signal ranges should contain.
    remove_intervals : Nested list containing info about ranges of the spectrum that should be masked out.

    Returns
    -------
    Nested list containing info about ranges of the spectrum that were estimated to contain signal.
        The goodness-of-fit calculations are only performed for the spectral channels within these ranges.

    """
    n_channels = spectrum.size

    #  TODO: max_amp_vals is calculated but not used -> can determine_peaks be simplified if max_amp_vals is not needed?
    _, ranges = determine_peaks(spectrum, peak="positive", amp_threshold=snr * rms)

    ranges = check_if_intervals_contain_signal(spectrum, rms, ranges, snr=snr, significance=significance)

    if len(ranges) == 0 or pad_channels <= 0:
        return ranges

    # TODO: can there be a more efficient implementation of the following buffering of intervals?

    # TODO: compared to the previous implementation this should give new results because of a bugfix; previously,
    #  in the ranges got padded by 1 x pad_channels in the 1st iteration, 2 x pad_channels in the 2nd iteration,
    #  3 x pad_channels in the 3rd iteration, and so on
    for i in itertools.count():
        ranges = _add_buffer_to_intervals(ranges, n_channels, pad_channels=pad_channels)
        mask_signal = mask_channels(n_channels, ranges, remove_intervals=remove_intervals)
        ranges = intervals_where_mask_is_true(mask_signal)
        # TODO: find something better for the second break condition
        if (np.count_nonzero(mask_signal) >= min_channels) or (2 * i * pad_channels >= spectrum.size):
            break
    return ranges


def get_noise_spike_ranges(spectrum: np.ndarray, rms: float, snr_noise_spike: float = 5.0) -> List[Optional[List]]:
    """Determine intervals in the spectrum potentially containing noise spikes.

    Parameters
    ----------
    spectrum : Intensity values of the spectrum.
    rms : Root-mean-square noise of the spectrum.
    snr_noise_spike : Required signal-to-noise ratio that a negative intensity value within an interval has to have
        to be counted as a potential noise spike.

    Returns
    -------
    Nested list containing slice information about intervals of the spectrum that potentially contain noise
        spike features. These intervals are neglected from goodness-of-fit calculations.

    """
    # TODO: change name of function to determine_noise_spike_intervals
    _, ranges = determine_peaks(spectrum, peak="negative", amp_threshold=snr_noise_spike * rms)
    return ranges.tolist()


def indices_of_fit_components_in_interval(fwhms: List, means: List, interval: List) -> Tuple[List, List]:
    """Find indices of components overlapping with the interval and update the interval range to accommodate full extent of the components.

    Component i is selected if means[i] +/- fwhms[i] overlaps with the
    interval.

    The interval is updated to accommodate all spectral channels contained in the range means[i] +/- fwhms[i].

    Parameters
    ----------
    fwhms : List of FWHM values of fit components.
    means : List of mean position values of fit components.
    interval : List specifying the interval of spectral channels containing the flagged feature in the form of
        [lower, upper].

    Returns
    -------
    indices : List with indices of components overlapping with interval.
    interval_new : Updated interval that accommodates all spectral channels contained in the range
        means[i] +/- fwhms[i].

    """
    lower_interval, upper_interval = interval.copy()
    lower_interval_new, upper_interval_new = interval.copy()
    indices = []

    for i, (mean, fwhm) in enumerate(zip(means, fwhms)):
        lower = max(0, mean - fwhm)
        upper = mean + fwhm
        if (lower_interval <= lower <= upper_interval) or (lower_interval <= upper <= upper_interval):
            lower_interval_new = min(lower_interval_new, lower)
            upper_interval_new = max(upper_interval_new, upper)
            indices.append(i)
    return indices, [lower_interval_new, upper_interval_new]


if __name__ == "__main__":

    from astropy.io import fits

    data = fits.getdata(ROOT / "gausspyplus" / "data" / "grs-test_field.fits")
    spectrum = data[:, 31, 40]

    # time for original function: 31.4 µs ± 1.66 µs per loop (mean ± std. dev. of 7 runs, 10000 loops each)
    print(intervals_where_mask_is_true(spectrum > 0.5))
