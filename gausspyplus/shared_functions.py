# @Author: riener
# @Date:   2018-12-19T17:26:54+01:00
# @Filename: shared_functions.py
# @Last modified by:   riener
# @Last modified time: 2019-04-08T10:18:23+02:00

import warnings

import numpy as np

from astropy.stats import median_absolute_deviation

from .output import format_warning
warnings.showwarning = format_warning


def determine_significance(amp, fwhm, rms):
    """Calculate the significance value of a fitted Gaussian component or a feature in the spectrum.

    The area of the Gaussian is:
    area_gauss = amp * fwhm / ((1. / np.sqrt(2*np.pi)) * 2*np.sqrt(2*np.log(2)))

    This is then compared to the integrated rms, with 2*fwhm being a good
    approximation for the width of the emission line

    significance = area_gauss / (np.sqrt(2*fwhm) * rms)

    combining all constants yields a factor of 0.75269184778925247

    Parameters
    ----------
    amp : float
        Amplitude value of the Gaussian component.
    fwhm : float
        FWHM value of the Gaussian component.
    rms : float
        Root-mean-square noise of the spectrum.

    """
    return amp * np.sqrt(fwhm) * 0.75269184778925247 / rms


def area_of_gaussian(amp, fwhm):
    """Calculate the integrated area of the Gaussian function.

    area_gauss = amp * fwhm / ((1. / np.sqrt(2*np.pi)) * 2*np.sqrt(2*np.log(2)))

    combining all constants in the denominator yields a factor of 0.93943727869965132

    Parameters
    ----------
    amp : float
        Amplitude value of the Gaussian component.
    fwhm : float
        FWHM value of the Gaussian component.

    """
    return amp * fwhm / 0.93943727869965132


def gaussian(amp, fwhm, mean, x):
    """Return results of a Gaussian function.

    Parameters
    ----------
    amp : float
        Amplitude of the Gaussian function.
    fwhm : float
        FWHM of the Gaussian function.
    mean : float
        Mean position of the Gaussian function.
    x : numpy.ndarray
        Array of spectral channels.

    Returns
    -------
    gauss : numpy.ndarray
        Gaussian function.

    """
    return amp * np.exp(-4. * np.log(2) * (x-mean)**2 / fwhm**2)


def combined_gaussian(amps, fwhms, means, x):
    """Return results of the combination of N Gaussian functions.

    Parameters
    ----------
    amps : list, numpy.ndarray
        List of the amplitude values of the Gaussian functions [amp1, ..., ampN].
    fwhms : list, numpy.ndarray
        List of the FWHM values of the Gaussian functions [fwhm1, ..., fwhmN].
    means : list, numpy.ndarray
        List of the mean positions of the Gaussian functions [mean1, ..., meanN].
    x : numpy.ndarray
        Array of spectral channels.

    Returns
    -------
    combined_gauss : numpy.ndarray
        Combination of N Gaussian functions.

    """
    if len(amps) > 0.:
        for i in range(len(amps)):
            gauss = gaussian(amps[i], fwhms[i], means[i], x)
            if i == 0:
                combined_gauss = gauss
            else:
                combined_gauss += gauss
    else:
        combined_gauss = np.zeros(len(x))
    return combined_gauss


def goodness_of_fit(data, best_fit_final, errors, ncomps_fit, mask=None,
                    get_aicc=False):
    """Determine the goodness of fit (reduced chi-square, AICc).

    Parameters
    ----------
    data : numpy.ndarray
        Original data.
    best_fit_final : numpy.ndarray
        Fit to the original data.
    errors : numpy.ndarray or float
        Root-mean-square noise for each channel.
    ncomps_fit : int
        Number of Gaussian components used for the fit.
    mask : numpy.ndarray
        Boolean array specifying which regions of the spectrum should be used.
    get_aicc : bool
        If set to `True`, the AICc value will be returned in addition to the
        reduced chi2 value.

    Returns
    -------
    rchi2 : float
        Reduced chi2 value.
    aicc : float
        (optional): The AICc value is returned if get_aicc is set to `True`.

    """
    if type(errors) is not np.ndarray:
        errors = np.ones(len(data)) * errors
    # TODO: check if mask is set to None everywehere there is no mask
    if mask is None:
        mask = np.ones(len(data))
        mask = mask.astype('bool')
    elif len(mask) == 0:
        mask = np.ones(len(data))
        mask = mask.astype('bool')
    elif np.count_nonzero(mask) == 0:
        mask = np.ones(len(data))
        mask = mask.astype('bool')

    squared_residuals = (data[mask] - best_fit_final[mask])**2
    chi2 = np.sum(squared_residuals / errors[mask]**2)
    n_params = 3*ncomps_fit  # degrees of freedom
    n_samples = len(data[mask])
    rchi2 = chi2 / (n_samples - n_params)
    if get_aicc:
        #  sum of squared residuals
        ssr = np.sum(squared_residuals)
        log_likelihood = -0.5 * n_samples * np.log(ssr / n_samples)
        aicc = (2.0 * (n_params - log_likelihood) +
                2.0 * n_params * (n_params + 1.0) /
                (n_samples - n_params - 1.0))
        return rchi2, aicc
    return rchi2


def add_subtracted_nan_ranges(nan_ranges, ranges):
    """Add masked out regions to signal or noise spike ranges."""
    for nan_lower, nan_upper in nan_ranges:
        for i, (lower, upper) in enumerate(ranges):
            if lower > nan_lower:
                add_value = (nan_upper - nan_lower)
                ranges[i] = [lower + add_value, upper + add_value]
    return ranges


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
    for consec_channels in range(1, 30):
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


def get_signal_ranges(spectrum, rms, max_consecutive_channels=14,
                      pad_channels=5, snr=3., significance=5.,
                      min_channels=100, remove_intervals=None):
    """Determine ranges in the spectrum that could contain signal.

    Parameters
    ----------
    spectrum : numpy.ndarray
        Array of the data values of the spectrum.
    rms : float
        Root-mean-square noise of the spectrum.
    max_consecutive_channels : int
        Determined maximum number of consecutive positive or negative channels of a (signal?) feature before it gets masked out.
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
        return []

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


def get_noise_spike_ranges(spectrum, rms, snr_noise_spike=3.5):
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
        Determined maximum number of consecutive positive or negative channels of a (signal?) feature before it gets masked out.
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
    mask = consecutive_channels > max_consecutive_channels
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
