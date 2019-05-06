"""Functions for noise estimation."""

import itertools
import random
import warnings

import numpy as np

from astropy.stats import median_absolute_deviation
from tqdm import tqdm

from .output import format_warning
warnings.showwarning = format_warning


def get_max_consecutive_channels(n_channels, p_limit):
    """Determine the maximum number of random consecutive positive/negative channels.

    Calculate the number of consecutive positive or negative channels,
    whose probability of occurring due to random chance in a spectrum
    is less than p_limit.

    Parameters
    ----------
    n_channels : int
        Number of spectral channels.
    p_limit : float
        Maximum probability for consecutive positive/negative channels being
        due to chance.

    Returns
    -------
    consec_channels : int
        Number of consecutive positive/negative channels that have a probability
        less than p_limit to be due to chance.

    """
    for consec_channels in range(2, 30):
        a = np.zeros((consec_channels, consec_channels))
        for i in range(consec_channels - 1):
            a[i, 0] = a[i, i + 1] = 0.5
        a[consec_channels - 1, consec_channels - 1] = 1.0
        if np.linalg.matrix_power(
                a, n_channels - 1)[0, consec_channels - 1] < p_limit:
            return consec_channels


def determine_peaks(spectrum, peak='both', amp_threshold=None):
    """Find peaks in a spectrum.

    Parameters
    ----------
    spectrum : numpy.ndarray
        Array of the data values of the spectrum.
    peak : 'both' (default), 'positive', 'negative'
        Description of parameter `peak`.
    amp_threshold : float
        Required minimum threshold that at least one data point in a peak feature has to exceed.

    Returns
    -------
    consecutive_channels or amp_vals : numpy.ndarray
        If the 'amp_threshold' value is supplied an array with the maximum data values of the ranges is returned. Otherwise, the number of spectral channels of the ranges is returned.
    ranges : list
        List of intervals [(low, upp), ...] determined to contain peaks.

    """
    if (peak == 'both') or (peak == 'positive'):
        clipped_spectrum = spectrum.clip(max=0)
        # Create an array that is 1 where a is 0, and pad each end with an extra 0.
        iszero = np.concatenate(
            ([0], np.equal(clipped_spectrum, 0).view(np.int8), [0]))
        absdiff = np.abs(np.diff(iszero))
        # Runs start and end where absdiff is 1.
        ranges = np.where(absdiff == 1)[0].reshape(-1, 2)

    if (peak == 'both') or (peak == 'negative'):
        clipped_spectrum = spectrum.clip(min=0)
        # Create an array that is 1 where a is 0, and pad each end with an extra 0.
        iszero = np.concatenate(
            ([0], np.equal(clipped_spectrum, 0).view(np.int8), [0]))
        absdiff = np.abs(np.diff(iszero))
        if peak == 'both':
            # Runs start and end where absdiff is 1.
            ranges = np.append(
                ranges, np.where(absdiff == 1)[0].reshape(-1, 2), axis=0)
        else:
            ranges = np.where(absdiff == 1)[0].reshape(-1, 2)

    if amp_threshold is not None:
        if peak == 'positive':
            mask = spectrum > abs(amp_threshold)
        elif peak == 'negative':
            mask = spectrum < -abs(amp_threshold)
        else:
            mask = np.abs(spectrum) > abs(amp_threshold)

        if np.count_nonzero(mask) == 0:
            return np.array([]), np.array([])

        peak_mask = np.split(mask, ranges[:, 1])
        mask_true = np.array([any(array) for array in peak_mask[:-1]])

        ranges = ranges[mask_true]
        if peak == 'positive':
            amp_vals = np.array([max(spectrum[low:upp]) for low, upp in ranges])
        elif peak == 'negative':
            amp_vals = np.array([min(spectrum[low:upp]) for low, upp in ranges])
        else:
            amp_vals = np.array(
                np.sign(spectrum[low])*max(np.abs(spectrum[low:upp]))
                for low, upp in ranges)
        #  TODO: check if sorting really necessary??
        sort_indices = np.argsort(amp_vals)[::-1]
        return amp_vals[sort_indices], ranges[sort_indices]
    else:
        sort_indices = np.argsort(ranges[:, 0])
        ranges = ranges[sort_indices]

        consecutive_channels = ranges[:, 1] - ranges[:, 0]
        return consecutive_channels, ranges


def mask_channels(n_channels, ranges, pad_channels=None, remove_intervals=None):
    """Determine the 1D boolean mask for a given list of spectral ranges.

    Parameters
    ----------
    n_channels : int
        Number of spectral channels.
    ranges : list
        List of intervals [(low, upp), ...].
    pad_channels : int
        Number of channels by which an interval (low, upp) gets extended on both sides, resulting in (low - pad_channels, upp + pad_channels).
    remove_intervals : type
        Nested list containing info about ranges of the spectrum that should be masked out.

    Returns
    -------
    mask : numpy.ndarray
        1D boolean mask that has 'True' values at the position of the channels contained in ranges.

    """
    mask = np.zeros(n_channels)

    for (lower, upper) in ranges:
        if pad_channels is not None:
            lower = max(0, lower - pad_channels)
            upper = min(n_channels, upper + pad_channels)
        mask[lower:upper] = 1

    if remove_intervals is not None:
        for (low, upp) in remove_intervals:
            mask[low:upp] = 0

    return mask.astype('bool')


def correct_rms(average_rms=None, idx=None):
    """Replace rms noise value with average rms value or mask out spectrum.

    Workaround for issues with bad baselines and/or insufficient continuum subtraction that render the noise computation meaningless.

    Parameters
    ----------
    average_rms : float
        Average root-mean-square noise value that is used in case the noise cannot be determined from the spectrum itself.
    idx : int
        Index of the spectrum.

    Returns
    -------
    rms : float or numpy.nan

    """
    idxInfo = ''
    if idx is not None:
        idxInfo = 'with index {} '.format(idx)
    if average_rms is not None:
        warnings.warn('Could not determine noise for spectrum {} (baseline issue?). Assuminge average rms value of {}'.format(idxInfo, average_rms))
        return average_rms
    else:
        warnings.warn('Could not determine noise for spectrum {} (baseline issue?). Masking out spectrum.'.format(idxInfo))
        return np.nan


def get_rms_noise(spectrum, max_consecutive_channels=14, pad_channels=5,
                  average_rms=None, idx=None, min_fraction_noise_channels=0.1,
                  min_fraction_average_rms=0.1):
    """Determine the root-mean-square noise of a spectrum.

    Parameters
    ----------
    spectrum : numpy.ndarray
        Original data of the spectrum.
    max_consecutive_channels : int
        Determined minimum number of consecutive positive or negative channels a (signal?) feature has to have to get masked out.
    pad_channels : int
        Number of additional channels that get masked out on both sides of an identified (signal?) feature.
    average_rms : float
        Average root-mean-square noise value that is used in case the noise cannot be determined from the spectrum itself.
    idx : int
        Index of the spectrum.
    min_fraction_noise_channels : float
        Required minimum fraction of spectral channels for reliable noise calculation. If this fraction is not reached, the 'average_rms' value (if supplied) is used or the spectrum is masked out.
    min_fraction_average_rms : float
        The estimated rms noise value has to exceed the average rms noise value by this fraction. Otherwise the 'average_rms' value (if supplied) is used or the spectrum is masked out.

    Returns
    -------
    rms : float
        Determined root-mean-square noise value for the spectrum.

    """
    #  Step 1: remove broad features based on number of consecutive channels
    n_channels = len(spectrum)
    consecutive_channels, ranges = determine_peaks(spectrum)
    mask = consecutive_channels >= max_consecutive_channels
    mask_1 = mask_channels(n_channels, ranges[mask], pad_channels=pad_channels)

    #  use average rms value or mask out spectrum in case all spectral channels were masked out in step 1
    if np.count_nonzero(~mask_1) == 0:
        return correct_rms(average_rms=average_rms, idx=idx)

    spectrum_consecs_removed = spectrum[~mask_1]

    #  Step 2: remove features with high positive or negative data values
    negative_indices = (spectrum_consecs_removed < 0.0)
    spectrum_negative_values = spectrum_consecs_removed[negative_indices]

    reflected_noise = np.concatenate((spectrum_negative_values,
                                      np.abs(spectrum_negative_values)))
    MAD = median_absolute_deviation(reflected_noise)

    spectrum_mask1_vals_zero = np.copy(spectrum)
    spectrum_mask1_vals_zero[mask_1] = 0
    inds_high_amps = np.where(np.abs(spectrum_mask1_vals_zero) > 5*MAD)[0]
    if inds_high_amps.size > 0:
        inds_ranges = np.digitize(inds_high_amps, ranges[:, 0]) - 1
        ranges = ranges[inds_ranges]

        mask_2 = mask_channels(n_channels, ranges, pad_channels=pad_channels)

        mask_total = mask_1 + mask_2
    else:
        mask_total = mask_1

    # TODO: change this from 0 to a minimum of required channels?
    if np.count_nonzero(~mask_total) == 0:
        return correct_rms(average_rms=average_rms, idx=idx)

    #  Step 3: determine the noise from the remaining channels
    rms = np.sqrt(np.sum(spectrum[~mask_total]**2) / np.size(spectrum[~mask_total]))

    if np.count_nonzero(
            ~mask_total) < min_fraction_noise_channels*len(spectrum):
        if average_rms is not None:
            if rms < min_fraction_average_rms*average_rms:
                return correct_rms(average_rms=average_rms, idx=idx)
        else:
            return correct_rms(average_rms=average_rms, idx=idx)
    return rms


def determine_noise(spectrum, max_consecutive_channels=14, pad_channels=5,
                    idx=None, average_rms=None, random_seed=111):
    np.random.seed(random_seed)
    if not np.isnan(spectrum).all():
        if np.isnan(spectrum).any():
            # TODO: Case where spectrum contains nans and only positive values
            nans = np.isnan(spectrum)
            error = get_rms_noise(spectrum[~nans], max_consecutive_channels=max_consecutive_channels, pad_channels=pad_channels, idx=idx, average_rms=average_rms)
            spectrum[nans] = np.random.randn(len(spectrum[nans])) * error

        elif (spectrum >= 0).all():
            warnings.warn('Masking spectra that contain only values >= 0')
            error = np.NAN
        else:
            error = get_rms_noise(spectrum, max_consecutive_channels=max_consecutive_channels, pad_channels=pad_channels, idx=idx, average_rms=average_rms)
    else:
        error = np.NAN
    return error


def calculate_average_rms_noise(data, number_rms_spectra, random_seed=111,
                                max_consecutive_channels=14, pad_channels=5):
    random.seed(random_seed)
    yValues = np.arange(data.shape[1])
    xValues = np.arange(data.shape[2])
    locations = list(itertools.product(yValues, xValues))
    if len(locations) > number_rms_spectra:
        locations = random.sample(locations, len(locations))
    rmsList = []
    counter = 0
    pbar = tqdm(total=number_rms_spectra)
    for y, x in locations:
        spectrum = data[:, y, x]
        error = determine_noise(
            spectrum, max_consecutive_channels=max_consecutive_channels,
            pad_channels=pad_channels)

        if not np.isnan(error):
            rmsList.append(error)
            counter += 1
            pbar.update(1)

        if counter >= number_rms_spectra:
            break

    pbar.close()
    return np.nanmean(rmsList)
    # return np.nanmedian(rmsList), median_absolute_deviation(rmsList, ignore_nan=True)
