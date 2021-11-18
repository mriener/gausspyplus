"""Functions for interval determination."""
from typing import List
from typing import List, Optional

import numpy as np

from .noise_estimation import determine_peaks, mask_channels

from gausspyplus.utils.noise_estimation import determine_peaks, mask_channels, intervals_where_mask_is_true, pad_intervals


def _merge_overlapping_intervals(intervals: List[List]) -> List[List]:
    """Merge overlapping intervals (Credit: https://stackoverflow.com/a/43600953)."""
    intervals.sort(key=lambda interval: interval[0])
    merged_intervals = [intervals[0]]
    for current in intervals:
        previous = merged_intervals[-1]
        if current[0] <= previous[1]:
            previous[1] = max(previous[1], current[1])
        else:
            merged_intervals.append(current)
    return merged_intervals


def _add_buffer_to_intervals(ranges: List[Optional[List]],
                             n_channels: int,
                             pad_channels: int = 5) -> List[Optional[List]]:
    """Pad intervals on lower and upper sides and merge overlapping intervals."""
    # TODO: this needs to be tested and compared time-wise with the previous
    if not ranges:  # in case ranges is an empty list
        return ranges
    intervals = pad_intervals(intervals=ranges, pad_channels=pad_channels, upper_limit=n_channels)
    return _merge_overlapping_intervals(intervals)


def check_if_intervals_contain_signal(spectrum: np.ndarray,
                                      rms: float,
                                      ranges: List[Optional[List]],
                                      snr: float = 3.,
                                      significance: float = 5.) -> List[Optional[List]]:
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
    intervals = [[lower, upper] for lower, upper in ranges if np.max(spectrum[lower:upper]) > snr*rms]
    return [[lower, upper] for lower, upper in intervals
            if np.sum(spectrum[lower:upper]) / (np.sqrt(upper - lower)*rms) > significance]


def get_signal_ranges(spectrum: np.ndarray,
                      rms: float,
                      pad_channels: int = 5,
                      snr: float = 3.,
                      significance: float = 5.,
                      min_channels: int = 100,
                      remove_intervals: Optional[List[Optional[List]]] = None) -> List[Optional[List]]:
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
    _, ranges = determine_peaks(spectrum, peak='positive', amp_threshold=snr*rms)

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


def get_noise_spike_ranges(spectrum: np.ndarray, rms: float, snr_noise_spike: int = 5) -> List[Optional[List]]:
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
    _, ranges = determine_peaks(spectrum, peak='negative', amp_threshold=snr_noise_spike*rms)
    return ranges.tolist()

