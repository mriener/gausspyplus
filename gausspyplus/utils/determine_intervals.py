"""Functions for interval determination."""

import numpy as np

from .noise_estimation import determine_peaks, mask_channels


def add_subtracted_nan_ranges(nan_ranges, ranges):
    """Add masked out regions to signal or noise spike ranges."""
    for nan_lower, nan_upper in nan_ranges:
        for i, (lower, upper) in enumerate(ranges):
            if lower > nan_lower:
                add_value = (nan_upper - nan_lower)
                ranges[i] = [lower + add_value, upper + add_value]
    return ranges


def intervals_where_mask_is_true(mask):
    """Determine intervals where a 1D boolean mask is True.

    Parameters
    ----------
    mask : numpy.ndarray
        Boolean mask.

    Returns
    -------
    ranges : list
        List of slice intervals [(low, upp), ...] indicating where the mask
        has `True` values.

    """
    indices = np.where(mask == True)[0]
    if indices.size == 0:
        return []

    nonzero = np.append(np.zeros(1), (indices[1:] - indices[:-1]) - 1)
    nonzero = nonzero.astype('int')
    indices_nonzero = np.argwhere(nonzero != 0)

    breakpoints = [indices[0]]
    if indices_nonzero.size != 0:
        for i in indices_nonzero:
            breakpoints.append(indices[i[0] - 1] + 1)
            breakpoints.append(indices[i[0]])
    breakpoints.append(indices[-1] + 1)

    ranges = []
    for i in range(int(len(breakpoints) / 2)):
        low, upp = breakpoints[i*2], breakpoints[i*2 + 1]
        if low != upp:
            ranges.append([low, upp])
        # if there is one single positive channel at the end
        # TODO: check if this may cause problems
        else:
            ranges.append([low, upp + 1])

    return ranges


def add_buffer_to_intervals(ranges, n_channels, pad_channels=5):
    """Extend interval range on both sides by a number of channels.

    Parameters
    ----------
    ranges : list
        List of intervals [(low, upp), ...].
    n_channels : int
        Number of spectral channels.
    pad_channels : int
        Number of channels by which an interval (low, upp) gets extended on both sides, resulting in (low - pad_channels, upp + pad_channels).

    Returns
    -------
    ranges_new : list
        New list of intervals [(low - pad_channels, upp + pad_channels), ...].

    """
    ranges_new, intervals = ([] for i in range(2))
    for i, (low, upp) in enumerate(ranges):
        low, upp = low - pad_channels, upp + pad_channels
        if low < 0:
            low = 0
        if upp > n_channels:
            upp = n_channels

        intervals.append((low, upp))

    # merge intervals if they are overlapping
    sorted_by_lower_bound = sorted(intervals, key=lambda tup: tup[0])

    for higher in sorted_by_lower_bound:
        if not ranges_new:
            ranges_new.append(higher)
        else:
            lower = ranges_new[-1]
            # test for intersection between lower and higher:
            # we know via sorting that lower[0] <= higher[0]
            if higher[0] <= lower[1]:
                upper_bound = max(lower[1], higher[1])
                ranges_new[-1] = (lower[0], upper_bound)  # replace by merged interval
            else:
                ranges_new.append(higher)

    return ranges_new


def gauss_mask(means, fwhms, n_channels, chi2_mask=None,
               range_slices=False, pad_channels=10):
    mask = np.zeros(n_channels)
    for mean, fwhm in zip(means, fwhms):
        if 2*fwhm < fwhm + pad_channels:
            pad = fwhm + pad_channels
        else:
            pad = 2*fwhm
        low, upp = int(mean - pad), int(mean + pad) + 2
        mask[low:upp] = 1

    if chi2_mask is not None:
        for (low, upp) in chi2_mask:
            mask[low:upp] = 0
    mask = mask.astype('bool')

    if range_slices:
        indices = np.where(mask == True)[0]
        nonzero = np.append(np.zeros(1), (indices[1:] - indices[:-1]) - 1)
        nonzero = nonzero.astype('int')
        indices2 = np.argwhere(nonzero != 0)
        breakpoints = [indices[0]]
        if indices2.size != 0:
            for i in indices2:
                breakpoints.append(indices[i[0] - 1])
                breakpoints.append(indices[i[0]])
        breakpoints.append(indices[-1])

        ranges = []
        for i in range(int(len(breakpoints) / 2)):
            low, upp = breakpoints[i*2], breakpoints[i*2 + 1]
            ranges.append((low, upp))
        return mask, ranges
    else:
        return mask


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

    i = 0
    if pad_channels is not None:
        while True:
            i += 1
            ranges = add_buffer_to_intervals(ranges, n_channels,
                                             pad_channels=i*pad_channels)
            mask_signal_new = mask_channels(n_channels, ranges,
                                            remove_intervals=remove_intervals)
            ranges = intervals_where_mask_is_true(mask_signal_new)
            if (np.count_nonzero(mask_signal_new) >= min_channels) or \
                    (2*i*pad_channels >= min_channels):
                return ranges
    return ranges


def get_noise_spike_ranges(spectrum, rms, snr_noise_spike=5):
    """Determine ranges in the spectrum that could contain noise spikes.

    Parameters
    ----------
    spectrum : numpy.ndarray
        Array of the data values of the spectrum.
    rms : float
        Root-mean-square noise of the spectrum.
    snr_noise_spike : float
        Required signal-to-noise ratio for negative data values to be counted as noise spikes.

    Returns
    -------
    noise_spike_ranges : list
        Nested list containing info about ranges of the spectrum that were estimated to contain noise spike features. These will get masked out from goodness-of-fit calculations.

    """
    maxampvals, ranges = determine_peaks(
        spectrum, peak='negative', amp_threshold=snr_noise_spike*rms)
    return ranges.tolist()
