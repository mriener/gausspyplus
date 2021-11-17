"""Functions for noise estimation."""

import itertools
import os
import random
from pathlib import Path
from typing import List, Tuple, Optional, Literal
import warnings

import numpy as np

from astropy.stats import median_absolute_deviation
from tqdm import tqdm

# from .output import format_warning
# warnings.showwarning = format_warning


def determine_maximum_consecutive_channels(n_channels: int, p_limit: float) -> int:
    """Determine the maximum number of random consecutive positive/negative channels.

    Calculate the number of consecutive positive or negative channels,
    whose probability of occurring due to random chance in a spectrum
    is less than p_limit.

    Parameters
    ----------
    n_channels : Number of spectral channels.
    p_limit : Maximum probability for consecutive positive/negative channels being due to chance.

    Returns
    -------
    consec_channels : Number of consecutive positive/negative channels that have a probability less than p_limit
        to be due to chance.

    """
    for n_consecutive_channels in itertools.count(2):
        matrix = np.zeros((n_consecutive_channels, n_consecutive_channels))
        for i in range(n_consecutive_channels - 1):
            matrix[i, 0] = 0.5
            matrix[i, i + 1] = 0.5
        matrix[-1, -1] = 1.0
        if np.linalg.matrix_power(matrix, n_channels - 1)[0, n_consecutive_channels - 1] < p_limit:
            return n_consecutive_channels


def intervals_where_mask_is_true(mask: np.ndarray) -> np.ndarray:
    """Determine intervals where a 1D boolean mask is True.

    Parameters
    ----------
    Boolean mask for 1D array.

    Returns
    -------
    Array of slice intervals [(idx_lower_1, idx_upper_1), ..., (idx_lower_N, idx_upper_N)] indicating where the mask
        has `True` values.

    """
    # TODO: return ranges as np.ndarray instead of list (.tolist() currently is still necessary for pytest to work)
    return np.flatnonzero(
        np.diff(
            np.concatenate((np.array([False]), mask, np.array([False])))
        )
    ).reshape(-1, 2).tolist()


# @jit(nopython=True)
def _determine_peak_intervals(spectrum: np.ndarray,
                              peak: Literal['positive', 'negative'] = 'positive') -> np.ndarray:
    return intervals_where_mask_is_true(mask=spectrum > 0 if peak == 'positive' else spectrum < 0)


def _get_number_of_consecutive_channels(peak_intervals: np.ndarray) -> np.ndarray:
    """Returns a list of the number of spectral channels of peak intervals."""
    return np.array([group[-1] - group[0] for group in peak_intervals])


def _get_peak_intervals(indices_of_peak_intervals: List[np.ndarray]) -> np.ndarray:
    """Returns a list of tuples containing the slice information for peak intervals."""
    return np.array([(array[0], array[-1] + 1) for array in indices_of_peak_intervals])


def determine_peaks(spectrum: np.ndarray,
                    peak: Literal['positive', 'negative'] = 'positive',
                    amp_threshold: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Find peaks (positive or negative) in a spectrum and return the intensity peak value and the peak interval.

    The returned array of the peak intervals is given as [[idx_lower_1, idx_upper_1], ..., [idx_lower_N, idx_upper_N]].
    Peak interval N can be extracted from the spectrum with `spectrum[idx_lower_N:idx_upper_N]`.

    Parameters
    ----------
    spectrum : Intensity values of the spectrum.
    peak : The type of peak to identify
    amp_threshold : Required minimum threshold that at least one data point in a peak feature has to exceed.

    Returns
    -------
    maximum_intensity_in_group : Maximum intensity values within the determined peak intervals
    peak_intervals : Slicing information for the determined peak intervals in the form of
        [(idx_lower_1, idx_upper_1), ..., (idx_lower_N, idx_upper_N)]. Peak interval N can be extracted from the
        spectrum with `spectrum[idx_lower_N:idx_upper_N]`.

    """
    # TODO: check if amp_threshold can ever be None??
    # TODO: rename amp_threshold
    # TODO: rename peak to peak_type
    amp_threshold = 0 if amp_threshold is None else amp_threshold
    indices_peaks = _determine_indices_of_peak_intervals(spectrum, peak=peak)
    sign_peaks = 1 if peak == 'positive' else -1
    maximum_intensity_in_group = np.array([np.abs(spectrum[group]).max() for group in indices_peaks])
    intensity_exceeds_threshold = maximum_intensity_in_group > abs(amp_threshold)
    maximum_intensity_in_group = maximum_intensity_in_group[intensity_exceeds_threshold]
    indices_peaks = [group for group, is_valid in zip(indices_peaks, intensity_exceeds_threshold) if is_valid]
    peak_intervals = _get_peak_intervals(indices_peaks)
    return maximum_intensity_in_group * sign_peaks, peak_intervals


def mask_channels(n_channels: int,
                  ranges: List[Tuple],
                  pad_channels: Optional[int] = None,
                  remove_intervals: Optional[List[Tuple]] = None) -> np.ndarray:
    """Determine the 1D boolean mask for a given list of spectral ranges.

    Parameters
    ----------
    n_channels : Number of spectral channels.
    ranges : List of intervals [(low, upp), ...].
    pad_channels : Number of channels by which an interval (low, upp) gets extended on both sides,
        resulting in (low - pad_channels, upp + pad_channels).
    remove_intervals : Nested list containing info about ranges of the spectrum that should be masked out.

    Returns
    -------
    mask : 1D boolean mask that has 'True' values at the position of the channels contained in ranges.

    """
    # TODO: check if ranges can be None -> Test
    pad_channels = 0 if pad_channels is None else pad_channels

    mask = np.zeros(n_channels, dtype=bool)
    if ranges is not None and len(ranges) > 1:
        ranges_padded = np.clip(a=np.array(ranges) + np.array([[-pad_channels, pad_channels]]),
                                a_min=0,
                                a_max=n_channels)
        indices_true = np.concatenate([np.r_[slice(*interval)] for interval in ranges_padded.tolist()])

        if remove_intervals is not None:
            indices_false = np.concatenate([np.r_[slice(*interval)] for interval in remove_intervals])
            indices_true = np.setdiff1d(indices_true, indices_false)
        mask[indices_true] = True
    return mask


def _correct_rms(average_rms: Optional[float] = None, idx: Optional[int] = None) -> float:
    """Replace rms noise value with average rms value or mask out spectrum.

    Workaround for issues with bad baselines and/or insufficient continuum subtraction that render the noise
    computation meaningless.

    Parameters
    ----------
    average_rms : Average root-mean-square noise value that is used in case the noise cannot be determined from
        the spectrum itself.
    idx : Index of the spectrum.

    Returns
    -------
    rms : float or numpy.nan

    """
    info_index = f'with index {idx} ' if idx is not None else ''
    info_action = f'Assuming average rms value of {average_rms}' if average_rms is not None else 'Masking out spectrum.'
    warnings.warn(f'Could not determine noise for spectrum {info_index} (baseline issue?). {info_action}')
    return average_rms if average_rms is not None else np.nan


# def _mask_features_with_high_peak_values():
#     spectrum_consecs_removed = spectrum[~mask_for_broad_features]
#     spectrum_negative_values = spectrum_consecs_removed[spectrum_consecs_removed < 0.0]
#
#     reflected_noise = np.concatenate(spectrum_negative_values, np.abs(spectrum_negative_values))
#
#     spectrum_mask1_vals_zero = np.copy(spectrum)
#     spectrum_mask1_vals_zero[mask_for_broad_features] = 0
#     inds_high_amps = np.where(np.abs(spectrum_mask1_vals_zero) > 5 * median_absolute_deviation(reflected_noise))[0]
#     if inds_high_amps.size > 0:
#         inds_ranges = np.digitize(inds_high_amps, peak_intervals[:, 0]) - 1
#         peak_intervals = peak_intervals[inds_ranges]
#
#         mask_2 = mask_channels(n_channels, peak_intervals, pad_channels=pad_channels)
#
#         mask_total = mask_for_broad_features + mask_2
#     else:
#         mask_total = mask_for_broad_features


def _get_rms_noise(spectrum, max_consecutive_channels=14, pad_channels=5,
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
    indices_positive_peaks = _determine_indices_of_peak_intervals(spectrum, peak='positive')
    indices_negative_peaks = _determine_indices_of_peak_intervals(spectrum, peak='negative')
    peak_intervals: np.ndarray = _get_peak_intervals(indices_positive_peaks + indices_negative_peaks)
    peak_intervals = np.sort(peak_intervals, axis=0)
    consecutive_channels = _get_number_of_consecutive_channels(peak_intervals)
    mask_for_broad_features = mask_channels(n_channels=n_channels,
                                            ranges=peak_intervals[consecutive_channels >= max_consecutive_channels],
                                            pad_channels=pad_channels)

    #  use average rms value or mask out spectrum in case all spectral channels were masked out in step 1
    if np.count_nonzero(~mask_for_broad_features) == 0:
        return correct_rms(average_rms=average_rms, idx=idx)

    # spectrum_consecs_removed = spectrum[~mask_for_broad_features]
    # spectrum_negative_values = spectrum_consecs_removed[spectrum_consecs_removed < 0]

    #  Step 2: remove features with high positive or negative data values
    spectrum_negative_values = spectrum[~np.logical_or(mask_for_broad_features, spectrum > 0)]

    reflected_noise = np.concatenate((spectrum_negative_values, np.abs(spectrum_negative_values)))

    spectrum_mask1_vals_zero = np.copy(spectrum)
    spectrum_mask1_vals_zero[mask_for_broad_features] = 0
    inds_high_amps = np.where(np.abs(spectrum_mask1_vals_zero) > 5*median_absolute_deviation(reflected_noise))[0]
    if inds_high_amps.size > 0:
        inds_ranges = np.digitize(inds_high_amps, peak_intervals[:, 0]) - 1
        peak_intervals = peak_intervals[inds_ranges]

        mask_2 = mask_channels(n_channels, peak_intervals, pad_channels=pad_channels)

        mask_total = mask_for_broad_features + mask_2
    else:
        mask_total = mask_for_broad_features

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


def get_rms_noise(spectrum: np.ndarray,
                   max_consecutive_channels: int = 14,
                   pad_channels: int = 5,
                   average_rms: Optional[float] = None,
                   idx: Optional[int] = None,
                   min_fraction_noise_channels: float = 0.1,
                   min_fraction_average_rms: float = 0.1) -> float:
    """Determine the root-mean-square noise of a spectrum.

    Parameters
    ----------
    spectrum : Intensity values of the spectrum.
    max_consecutive_channels : Determined minimum number of consecutive positive or negative channels a (signal?)
        feature has to have to get masked out.
    pad_channels : Number of additional channels that get masked out on both sides of an identified (signal?) feature.
    average_rms : Average root-mean-square noise value that is used in case the noise cannot be determined from the
        spectrum itself.
    idx : Index of the spectrum.
    min_fraction_noise_channels : Required minimum fraction of spectral channels for reliable noise calculation. If
        this fraction is not reached, the 'average_rms' value (if supplied) is used or the spectrum is masked out.
    min_fraction_average_rms : The estimated rms noise value has to exceed the average rms noise value by this
        fraction. Otherwise the 'average_rms' value (if supplied) is used or the spectrum is masked out.

    Returns
    -------
    rms : Determined root-mean-square noise value for the spectrum.

    """
    #  Step 1: remove broad features based on number of consecutive channels
    n_channels = len(spectrum)
    n_required_noise_channels = min_fraction_noise_channels * n_channels

    max_intensities_of_positive_intervals, positive_intensity_intervals = determine_peaks(spectrum, peak='positive')
    is_too_broad_positive = _get_number_of_consecutive_channels(positive_intensity_intervals) > max_consecutive_channels

    min_intensities_of_negative_intervals, negative_intensity_intervals = determine_peaks(spectrum, peak='negative')
    is_too_broad_negative = _get_number_of_consecutive_channels(negative_intensity_intervals) > max_consecutive_channels

    intervals_that_are_too_broad = np.concatenate((positive_intensity_intervals[is_too_broad_positive],
                                                   negative_intensity_intervals[is_too_broad_negative]))
    mask_for_broad_features = mask_channels(n_channels=n_channels,
                                            ranges=intervals_that_are_too_broad,
                                            pad_channels=pad_channels)

    valid_negative_channels_for_mad = spectrum[~np.logical_or(mask_for_broad_features, spectrum > 0)]
    if len(valid_negative_channels_for_mad) > n_required_noise_channels:
        mad_value = median_absolute_deviation(np.concatenate((valid_negative_channels_for_mad,
                                                              valid_negative_channels_for_mad * -1)))
        has_too_high_peak_value = max_intensities_of_positive_intervals > 5 * mad_value
        has_too_low_peak_value = min_intensities_of_negative_intervals < -5 * mad_value
        intervals_that_are_too_high = np.concatenate((positive_intensity_intervals[has_too_high_peak_value],
                                                      negative_intensity_intervals[has_too_low_peak_value]))
        mask_for_high_features = mask_channels(n_channels=n_channels,
                                               ranges=intervals_that_are_too_high,
                                               pad_channels=pad_channels)

        has_valid_noise_channels = ~np.logical_or(mask_for_broad_features, mask_for_high_features)
    else:
        has_valid_noise_channels = ~mask_for_broad_features

    #  Step 3: determine the noise from the remaining channels
    noise_values = spectrum[has_valid_noise_channels]
    n_valid_noise_channels = len(noise_values)

    try:
        rms_noise = np.sqrt(np.sum(noise_values ** 2) / n_valid_noise_channels)
        rms_noise_min_value = min_fraction_average_rms * average_rms if average_rms is not None else rms_noise
    except ZeroDivisionError:
        pass

    if (n_valid_noise_channels < n_required_noise_channels) or (rms_noise < rms_noise_min_value):
        rms_noise = correct_rms(average_rms=average_rms, idx=idx)

    return rms_noise


def determine_noise(spectrum, max_consecutive_channels=14, pad_channels=5,
                    idx=None, average_rms=None, random_seed=111):
    np.random.seed(random_seed)
    if not np.isnan(spectrum).all():
        if np.isnan(spectrum).any():
            # TODO: Case where spectrum contains nans and only positive values
            nans = np.isnan(spectrum)
            error = get_rms_noise(
                spectrum[~nans],
                max_consecutive_channels=max_consecutive_channels,
                pad_channels=pad_channels,
                idx=idx,
                average_rms=average_rms)
            spectrum[nans] = np.random.randn(len(spectrum[nans])) * error

        elif (spectrum >= 0).all():
            warnings.warn('Masking spectra that contain only values >= 0')
            error = np.NAN
        else:
            error = get_rms_noise(spectrum, max_consecutive_channels=max_consecutive_channels,
                                  pad_channels=pad_channels, idx=idx, average_rms=average_rms)
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


if __name__ == '__main__':
    # for testing
    from timeit import timeit
    from astropy.io import fits
    ROOT = Path(os.path.realpath(__file__)).parents[1]
    data = fits.getdata(ROOT / 'data' / 'grs-test_field.fits')
    # spectrum = data[:, 26, 8]
    # results = determine_peaks(spectrum, amp_threshold=0.4)
    spectrum = data[:, 31, 40]

    time_new = timeit(lambda: get_rms_noise(spectrum), number=10000)
    time_old = timeit(lambda: _get_rms_noise(spectrum), number=10000)
    print(f'{time_old=}, {time_new=}')
    # results = get_rms_noise(spectrum)
    # print(results)
