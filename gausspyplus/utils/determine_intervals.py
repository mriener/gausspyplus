"""Functions for interval determination."""
from typing import List

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


def mask_covering_gaussians(means, fwhms, n_channels, remove_intervals=None,
                            range_slices=False, pad_channels=10, min_channels=100):
    """Define mask around fitted Gaussians for goodness of fit calculations.

    This is currently not in use in GaussPy+.

    Parameters
    ----------
    means : list
        List containing mean position values of all N fitted Gaussian components in the form [mean1, ..., meanN].
    fwhms : list
        List containing FWHM values of all N fitted Gaussian components in the form [fwhm1, ..., fwhmN].
    n_channels : int
        Number of spectral channels.
    remove_intervals : list
        Nested list containing info about ranges of the spectrum that should be masked out.
    range_slices : bool
        Default is 'False'. If set to 'True', the determined ranges are returned in additon to the mask.
    pad_channels : int
        Number of additional channels that get masked out on both sides of an identified (signal?) feature.
    min_channels : int
        Required minimum number of spectral channels that the signal ranges should contain.

    Returns
    -------
    mask : numpy.ndarray
        Boolean array that masks out all spectral channels not covered by fitted Gaussian components.

    """
    ranges = []
    for mean, fwhm in zip(means, fwhms):
        if 2*fwhm < fwhm + pad_channels:
            pad = fwhm + pad_channels
        else:
            pad = 2*fwhm
        ranges.append((int(mean - pad), int(mean + pad) + 2))

    mask = mask_channels(n_channels, ranges, remove_intervals=remove_intervals)

    ranges = intervals_where_mask_is_true(mask)

    if pad_channels is not None:
        i = 0
        while np.count_nonzero(mask) < min_channels:
            i += 1
            ranges = add_buffer_to_intervals(ranges, n_channels, pad_channels=i*pad_channels)
            mask = mask_channels(n_channels, ranges, remove_intervals=remove_intervals)
            if 2*i*pad_channels >= min_channels:
                break

    if range_slices:
        return mask, ranges

    return mask


def check_if_intervals_contain_signal(spectrum, rms, ranges, snr=3.,
                                      significance=5.):
    """Check if selected intervals contain positive signal.

    If the maximum intensity value of an interval (low, upp) is smaller than
    snr * rms, the interval gets removed from 'ranges'.

    The required minimum significance threshold helps to remove narrow noise spikes or insignificant positive intensity peaks.

    Parameters
    ----------
    spectrum : numpy.ndarray
        Array of the data values of the spectrum.
    rms : float
        Root-mean-square noise of the spectrum.
    ranges : list
        List of intervals [(low, upp), ...] that were identified as containing
        positive signal.
    snr : float
        Required minimum signal-to-noise ratio for data peak.
    significance : float
        Required minimum value for significance criterion.

    Returns
    -------
    ranges_new : list
        New list of intervals [(low, upp), ...] that contain positive signal.

    """
    ranges_new = []
    for low, upp in ranges:
        if np.max(spectrum[low:upp]) > snr*rms:
            if np.sum(spectrum[low:upp]) / (np.sqrt(upp - low)*rms) > significance:
                ranges_new.append([low, upp])
    return ranges_new


def get_signal_ranges(spectrum, rms, pad_channels=5, snr=3., significance=5.,
                      min_channels=100, remove_intervals=None):
    """Determine ranges in the spectrum that could contain signal.

    Parameters
    ----------
    spectrum : numpy.ndarray
        Array of the data values of the spectrum.
    rms : float
        Root-mean-square noise of the spectrum.
    pad_channels : int
        Number of additional channels that get masked out on both sides of an identified (signal?) feature.
    snr : float
        Required minimum signal-to-noise ratio for data peak.
    significance : float
        Required minimum value for significance criterion.
    min_channels : int
        Required minimum number of spectral channels that the signal ranges should contain.
    remove_intervals : list
        Nested list containing info about ranges of the spectrum that should be masked out.

    Returns
    -------
    signal_ranges : list
        Nested list containing info about ranges of the spectrum that were estimated to contain signal. The goodness-of-fit calculations are only performed for the spectral channels within these ranges.

    """
    n_channels = len(spectrum)

    max_amp_vals, ranges = determine_peaks(
        spectrum, peak='positive', amp_threshold=snr*rms)

    ranges = check_if_intervals_contain_signal(
        spectrum, rms, ranges, snr=snr, significance=significance)

    if not ranges:
        if remove_intervals is None:
            return []
        else:
            mask = mask_channels(n_channels, [[0, n_channels]],
                                 remove_intervals=remove_intervals)
            return intervals_where_mask_is_true(mask)

    #  safeguard to prevent eternal loop
    if pad_channels <= 0:
        pad_channels = None

    i = 0
    if pad_channels is not None:
        while True:
            i += 1
            # TODO: something is not right here: if pad_channels is 5, in first round ranges gets padded with 5 channels,
            #  and in second round the already padded ranges gets padded by another 10 channels?
            ranges = _add_buffer_to_intervals(ranges, n_channels,
                                             pad_channels=i*pad_channels)
            mask_signal_new = mask_channels(n_channels, ranges,
                                            remove_intervals=remove_intervals)
            ranges = intervals_where_mask_is_true(mask_signal_new)
            if (np.count_nonzero(mask_signal_new) >= min_channels) or \
                    (2*i*pad_channels >= min_channels):
                return ranges
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

