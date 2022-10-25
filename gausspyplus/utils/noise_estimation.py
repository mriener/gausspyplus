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


import sys

ROOT = Path(os.path.realpath(__file__)).parents[2]
sys.path.append(str(ROOT))

from gausspyplus.utils.output import format_warning

warnings.showwarning = format_warning


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
        if (
            np.linalg.matrix_power(matrix, n_channels - 1)[
                0, n_consecutive_channels - 1
            ]
            < p_limit
        ):
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
    return (
        np.flatnonzero(
            np.diff(np.concatenate((np.array([False]), mask, np.array([False]))))
        )
        .reshape(-1, 2)
        .tolist()
    )


# @jit(nopython=True)
def _determine_peak_intervals(
    spectrum: np.ndarray, peak: Literal["positive", "negative"] = "positive"
) -> np.ndarray:
    return intervals_where_mask_is_true(
        mask=spectrum > 0 if peak == "positive" else spectrum < 0
    )


def _get_number_of_consecutive_channels(peak_intervals: np.ndarray) -> np.ndarray:
    """Returns a list of the number of spectral channels of peak intervals."""
    return peak_intervals[:, 1] - peak_intervals[:, 0]


def determine_peaks(
    spectrum: np.ndarray,
    peak: Literal["positive", "negative"] = "positive",
    amp_threshold: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
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
    peak_intervals = _determine_peak_intervals(spectrum, peak=peak)
    maximum_value_in_peak_interval = np.array(
        [np.abs(spectrum[low:upp]).max() for low, upp in peak_intervals]
    )
    exceeds_threshold = maximum_value_in_peak_interval > abs(amp_threshold)
    maximum_intensity_in_group = maximum_value_in_peak_interval[exceeds_threshold]
    peak_intervals = np.array(
        [
            interval
            for interval, is_valid in zip(peak_intervals, exceeds_threshold)
            if is_valid
        ]
    )
    return (
        maximum_intensity_in_group * (1 if peak == "positive" else -1),
        peak_intervals,
    )


def pad_intervals(
    intervals: List[List],
    pad_channels: Optional[int],
    lower_limit: int = 0,
    upper_limit: Optional[int] = None,
) -> List[List]:
    """Pad intervals with channels on the lower and upper end."""
    if pad_channels is None:
        return intervals
    else:
        return [
            [
                max(lower_limit, lower - pad_channels),
                min(upper_limit, upper + pad_channels),
            ]
            for lower, upper in intervals
        ]


def mask_channels(
    n_channels: int,
    ranges: List[Tuple],
    pad_channels: Optional[int] = None,
    remove_intervals: Optional[List[Tuple]] = None,
) -> np.ndarray:
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
    mask = np.zeros(n_channels, dtype="bool")
    ranges = pad_intervals(
        intervals=ranges, pad_channels=pad_channels, upper_limit=n_channels
    )
    for (lower, upper) in ranges:
        mask[lower:upper] = True

    if remove_intervals is not None:
        for (lower, upper) in remove_intervals:
            mask[lower:upper] = False

    return mask


def _correct_rms(
    average_rms: Optional[float] = None, idx: Optional[int] = None
) -> float:
    """Replace spurious rms noise value with average rms noise value (if available) or set noise value to NaN.

    This is a safeguard for spectra with bad baselines and/or insufficient continuum subtraction that render the noise
    computation meaningless. If the rms noise value is set to NaN, the associated spectrum is masked out.
    """
    info_index = f"with index {idx} " if idx is not None else ""
    info_action = (
        f"Assuming average rms value of {average_rms}"
        if average_rms is not None
        else "Masking out spectrum."
    )
    warnings.warn(
        f"Could not determine noise for spectrum {info_index} (baseline issue?). {info_action}"
    )
    return average_rms or np.nan


def _identify_valid_noise_values(
    spectrum: np.ndarray,
    peak_intervals: np.ndarray,
    max_consecutive_channels: Optional[int],
    pad_channels: Optional[int],
) -> np.ndarray:
    """Exclude spectral features that are broad or have high values and return remaining valid noise channels"""
    #  Step 1: remove broad features based on number of consecutive channels
    consecutive_channels = _get_number_of_consecutive_channels(peak_intervals)
    mask_for_broad_features = mask_channels(
        n_channels=spectrum.size,
        ranges=peak_intervals[consecutive_channels >= max_consecutive_channels],
        pad_channels=pad_channels,
    )
    if np.count_nonzero(~mask_for_broad_features) == 0:
        return np.empty(0)

    #  Step 2: remove features with high positive or negative data values
    spectrum_negative_values = spectrum[
        ~np.logical_or(mask_for_broad_features, spectrum > 0)
    ]
    reflected_noise = np.concatenate(
        (spectrum_negative_values, np.abs(spectrum_negative_values))
    )

    spectrum_with_broad_features_set_to_zero = spectrum.copy()
    spectrum_with_broad_features_set_to_zero[mask_for_broad_features] = 0

    channels_exceeding_threshold = np.flatnonzero(
        np.abs(spectrum_with_broad_features_set_to_zero)
        > 5 * median_absolute_deviation(reflected_noise)
    )
    if channels_exceeding_threshold.size > 0:
        peak_interval_has_high_value = (
            np.digitize(channels_exceeding_threshold, peak_intervals[:, 0]) - 1
        )
        mask_for_high_features = mask_channels(
            n_channels=spectrum.size,
            ranges=peak_intervals[peak_interval_has_high_value],
            pad_channels=pad_channels,
        )
        has_valid_noise_channels = ~np.logical_or(
            mask_for_broad_features, mask_for_high_features
        )
    else:
        has_valid_noise_channels = ~mask_for_broad_features

    return spectrum[has_valid_noise_channels]


def _determine_valid_noise_channels_and_calculate_rms_noise(
    spectrum: np.ndarray,
    max_consecutive_channels: Optional[int] = None,
    pad_channels: int = 5,
    average_rms: Optional[float] = None,
    idx: Optional[int] = None,
    min_fraction_noise_channels: float = 0.1,
    min_fraction_average_rms: float = 0.1,
) -> float:
    """Identify all suitable noise channels in a spectrum and use them to calculate the root-mean-square noise value."""
    positive_peaks = _determine_peak_intervals(spectrum, peak="positive")
    negative_peaks = _determine_peak_intervals(spectrum, peak="negative")
    peak_intervals = np.concatenate((positive_peaks, negative_peaks))
    peak_intervals = peak_intervals[np.argsort(peak_intervals[:, 0])]
    noise_values = _identify_valid_noise_values(
        spectrum, peak_intervals, max_consecutive_channels, pad_channels
    )

    if noise_values.size <= min_fraction_noise_channels * spectrum.size:
        rms_noise = _correct_rms(average_rms=average_rms, idx=idx)
    else:
        rms_noise = np.sqrt(np.sum(noise_values**2) / noise_values.size)
        if rms_noise < min_fraction_average_rms * (average_rms or 0):
            rms_noise = _correct_rms(average_rms=average_rms, idx=idx)

    return rms_noise


def determine_noise(
    spectrum,
    max_consecutive_channels: Optional[int] = None,
    pad_channels: int = 5,
    idx: Optional[int] = None,
    average_rms: Optional[int] = None,
    random_seed: int = 111,
    min_fraction_noise_channels: float = 0.1,
    min_fraction_average_rms: float = 0.1,
):
    """Determine the root-mean-square noise value of a spectrum.

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
    # TODO: test functionality of refactored function and compare the performance to the previous implementation
    if max_consecutive_channels is None:
        max_consecutive_channels = determine_maximum_consecutive_channels(
            n_channels=spectrum.size, p_limit=0.02
        )
    np.random.seed(random_seed)  # TODO: check if this is really needed here?

    # TODO: The following is only a problem if channels WITHIN the spectrum are NaNs, so that the calculation of
    #  consecutive intervals can be faulty
    spectrum_without_nans = spectrum[~np.isnan(spectrum)]
    if spectrum_without_nans.size == 0:
        return np.nan
    elif (spectrum_without_nans >= 0).all():
        warnings.warn("Masking spectra that contain only values >= 0")
        return np.nan
    else:
        return _determine_valid_noise_channels_and_calculate_rms_noise(
            spectrum=spectrum_without_nans,
            max_consecutive_channels=max_consecutive_channels,
            pad_channels=pad_channels,
            idx=idx,
            average_rms=average_rms,
            min_fraction_noise_channels=min_fraction_noise_channels,
            min_fraction_average_rms=min_fraction_average_rms,
        )


def calculate_average_rms_noise(
    data: np.ndarray,
    number_rms_spectra: int,
    random_seed: int = 111,
    max_consecutive_channels: int = 14,
    pad_channels: int = 5,
) -> float:
    # TODO: test functionality of refactored function and compare the performance to the previous implementation
    random.seed(random_seed)
    y_positions = np.arange(data.shape[1])
    x_positions = np.arange(data.shape[2])
    locations = list(itertools.product(y_positions, x_positions))
    if len(locations) > number_rms_spectra:
        locations = random.sample(locations, len(locations))
    sampled_rms_noise_values = []
    pbar = tqdm(total=number_rms_spectra)
    for y, x in locations:
        rms_noise = determine_noise(
            spectrum=data[:, y, x],
            max_consecutive_channels=max_consecutive_channels,
            pad_channels=pad_channels,
        )
        if not np.isnan(rms_noise):
            sampled_rms_noise_values.append(rms_noise)
            pbar.update(1)
        if len(sampled_rms_noise_values) >= number_rms_spectra:
            break
    pbar.close()
    return np.nanmean(sampled_rms_noise_values)


if __name__ == "__main__":
    # TODO: check if spectrum contains neighboring values with the exact same value (problem for gausspy)
    # for testing
    from timeit import timeit
    from astropy.io import fits

    ROOT = Path(os.path.realpath(__file__)).parents[1]
    data = fits.getdata(ROOT / "data" / "grs-test_field.fits")
    # spectrum = data[:, 26, 8]
    # results = determine_peaks(spectrum, amp_threshold=0.4)
    spectrum = data[:, 31, 40]

    # spectrum = np.empty(0)
    # result = get_rms_noise(spectrum)
    result = determine_noise(spectrum)
    print(result)

    # time_new = timeit(lambda: get_rms_noise(spectrum), number=10000)
    # print(f'{time_new=}')
    # time_old = timeit(lambda: _determine_valid_noise_channels_and_calculate_rms_noise(spectrum), number=10000)
    # print(f'{time_old=}, {time_new=}')
    # results = get_rms_noise(spectrum)
    # print(results)

    # %timeit -r 10 determine_peaks(spectrum)
    # 363 µs ± 11.8 µs per loop (mean ± std. dev. of 10 runs, 1000 loops each)
    # %timeit -r 10 _determine_peaks(spectrum)
    # 419 µs ± 12.3 µs per loop (mean ± std. dev. of 10 runs, 1000 loops each)

    # %timeit -r 10 _determine_valid_noise_channels_and_calculate_rms_noise(spectrum, max_consecutive_channels=14)
    # time speedup with numba
    # 49.1 µs ± 35.5 µs per loop (mean ± std. dev. of 10 runs, 1 loop each)
