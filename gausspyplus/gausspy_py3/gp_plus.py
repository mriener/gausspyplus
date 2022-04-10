import functools
import itertools
from collections import namedtuple
from dataclasses import dataclass
from typing import Optional, List, Union, Any, Tuple, Literal, Dict

import numpy as np

from lmfit import minimize as lmfit_minimize

from gausspyplus.model import Model
from gausspyplus.spectrum import Spectrum
from gausspyplus.utils.determine_intervals import check_if_intervals_contain_signal
from gausspyplus.utils.fit_quality_checks import determine_significance, goodness_of_fit, check_residual_for_normality
from gausspyplus.utils.gaussian_functions import (
    combined_gaussian, 
    area_of_gaussian, 
    split_params,
    number_of_gaussian_components, 
    multi_component_gaussian_model,
    vals_vec_from_lmfit,
    errs_vec_from_lmfit,
    paramvec_to_lmfit
)
from gausspyplus.utils.noise_estimation import determine_peaks, mask_channels


def _perform_least_squares_fit(spectrum: namedtuple,
                               params_fit: List,
                               dct: Dict,
                               params_min: Optional[List] = None,
                               params_max: Optional[List] = None) -> Tuple[List, List, int]:
    # Objective functions for final fit
    def objective_leastsq(paramslm):
        params = vals_vec_from_lmfit(paramslm)
        resids = (multi_component_gaussian_model(
            spectrum.channels, *params).ravel() - spectrum.intensity_values.ravel()) / spectrum.noise_values
        return resids

    #  get new best fit
    lmfit_params = paramvec_to_lmfit(
        paramvec=params_fit,
        max_amp=dct['max_amp'],
        max_fwhm=None,
        params_min=params_min,
        params_max=params_max
    )
    try:
        result = lmfit_minimize(
            objective_leastsq, lmfit_params, method='leastsq')
        params_fit = vals_vec_from_lmfit(lmfit_params=result.params)
        params_errs = errs_vec_from_lmfit(lmfit_params=result.params)
        # TODO: implement bootstrapping method to estimate error in case
        #  error is None (when parameters are close to given bounds)
        # if (len(params_errs) != 0) and (sum(params_errs) == 0):
        #     print('okay')
        #     mini = lmfit.Minimizer(objective_leastsq, lmfit_params)
        #     for p in result.params:
        #         result.params[p].stderr = abs(result.params[p].value * 0.1)
        #     print('params_errs_old', params_errs)
        #     ci = lmfit.conf_interval(mini, result)
        #     print('Step 2')
        #     params_errs = []
        #     for p in ci:
        #         min_val, val, max_val = ci[p][2][1], ci[p][3][1], ci[p][4][1]
        #         std = max(abs(min_val - val), (abs(max_val - val)))
        #         params_errs.append(std)
        #     print('params_errs', params_errs)

        ncomps_fit = number_of_gaussian_components(params=params_fit)

        return params_fit, params_errs, ncomps_fit
    except (ValueError, TypeError):
        return [], [], 0


def remove_components_from_sublists(lst: List, remove_indices: List[int]) -> List:
    """Remove items with indices idx1, ..., idxN from all sublists of a nested list.

    Parameters
    ----------
    lst : Nested list [sublist1, ..., sublistN].
    remove_indices : List of indices [idx1, ..., idxN] indicating the items that should be removed from the sublists.

    Returns
    -------
    lst : list

    """
    for idx, sublst in enumerate(lst):
        lst[idx] = [val for i, val in enumerate(sublst) if i not in remove_indices]
    return lst


def _remove_components_above_max_ncomps(amps_fit: List,
                                        fwhms_fit: List,
                                        ncomps_max: int,
                                        remove_indices: List[int],
                                        quality_control: List[int]) -> Tuple[List[int], List[int]]:
    """Remove all fit components above specified limit.

    Parameters
    ----------
    amps_fit : List containing amplitude values of all N fitted Gaussian components in the form [amp1, ..., ampN].
    fwhms_fit : List containing FWHM values of all N fitted Gaussian components in the form [fwhm1, ..., fwhmN].
    ncomps_max : Specified maximum number of fit components.
    remove_indices : Indices of Gaussian components that should be removed from the fit solution.
    quality_control : Log containing information about which in-built quality control parameters were not fulfilled
        (0: 'max_fwhm', 1: 'min_fwhm', 2: 'snr', 3: 'significance', 4: 'channel_range', 5: 'signal_range')

    Returns
    -------
    remove_indices : Updated list with indices of Gaussian components that should be removed from the fit solution.
    quality_control : Updated log containing information about which in-built quality control parameters were not
        fulfilled.

    """
    ncomps_fit = len(amps_fit)
    if ncomps_fit <= ncomps_max:
        return remove_indices, quality_control
    integrated_intensities = area_of_gaussian(amp=np.array(amps_fit), fwhm=np.array(fwhms_fit))
    indices = np.array(range(len(integrated_intensities)))
    sort_indices = np.argsort(integrated_intensities)

    for index in indices[sort_indices]:
        if index in remove_indices:
            continue
        remove_indices.append(index)
        quality_control.append(6)
        remaining_ncomps = ncomps_fit - len(remove_indices)
        if remaining_ncomps <= ncomps_max:
            break

    return remove_indices, quality_control


def _check_params_fit(spectrum: namedtuple,
                      params_fit: List,
                      params_errs: List,
                      dct: Dict,
                      quality_control: List[int],
                      params_min: Optional[List] = None,
                      params_max: Optional[List] = None) -> Tuple[List, List, int, List, List, List, bool]:
    """Perform quality checks for the fitted Gaussians components.

    All Gaussian components that are not satisfying the criteria are discarded from the fit.

    Parameters
    ----------
    params_fit : Parameter vector in the form of [amp1, ..., ampN, fwhm1, ..., fwhmN, mean1, ..., meanN].
    params_errs : Parameter error vector in the form of [e_amp1, ..., e_ampN, e_fwhm1, ..., e_fwhmN,
        e_mean1, ..., e_meanN].
    dct : Dictionary containing parameter settings for the improved fitting.
    params_min : List of minimum limits for parameters: [min_amp1, ..., min_ampN, min_fwhm1, ..., min_fwhmN,
        min_mean1, ..., min_meanN]
    params_max : List of maximum limits for parameters: [max_amp1, ..., max_ampN, max_fwhm1, ..., max_fwhmN,
        max_mean1, ..., max_meanN]
    quality_control : Log containing information about which in-built quality control parameters were not fulfilled
        (0: 'max_fwhm', 1: 'min_fwhm', 2: 'snr', 3: 'significance', 4: 'channel_range', 5: 'signal_range')

    Returns
    -------
    params_fit : Corrected version from which all Gaussian components that did not satisfy the quality criteria are
        removed.
    params_err : Corrected version from which all Gaussian components that did not satisfy the quality criteria are
        removed.
    ncomps_fit : Number of remaining fitted Gaussian components.
    params_min : Corrected version from which all Gaussian components that did not satisfy the quality criteria are
        removed.
    params_max : Corrected version from which all Gaussian components that did not satisfy the quality criteria are
        removed.
    quality_control : Updated log containing information about which in-built quality control parameters were not
        fulfilled.
    refit : If 'True', the spectrum was refit because one or more of the Gaussian fit parameters did not satisfy the
        quality control parameters.

    """
    ncomps_fit = number_of_gaussian_components(params=params_fit)

    amps_fit, fwhms_fit, offsets_fit = split_params(params=params_fit, ncomps=ncomps_fit)
    amps_errs, fwhms_errs, offsets_errs = split_params(params=params_errs, ncomps=ncomps_fit)
    if params_min is not None:
        amps_min, fwhms_min, offsets_min = split_params(params=params_min, ncomps=ncomps_fit)
    if params_max is not None:
        amps_max, fwhms_max, offsets_max = split_params(params=params_max, ncomps=ncomps_fit)

    #  check if Gaussian components satisfy quality criteria

    remove_indices = []
    for i, (amp, fwhm, offset) in enumerate(
            zip(amps_fit, fwhms_fit, offsets_fit)):
        if dct['max_fwhm'] is not None:
            if fwhm > dct['max_fwhm']:
                remove_indices.append(i)
                quality_control.append(0)
                continue

        if dct['min_fwhm'] is not None:
            if fwhm < dct['min_fwhm']:
                remove_indices.append(i)
                quality_control.append(1)
                continue

        #  discard the Gaussian component if its amplitude value does not satisfy the required minimum S/N value or is larger than the limit
        if amp < dct['snr_fit'] * spectrum.rms_noise:
            remove_indices.append(i)
            quality_control.append(2)
            continue

        # if amp > dct['max_amp']:
        #     remove_indices.append(i)
        #     quality_control.append(3)
        #     continue

        #  discard the Gaussian component if it does not satisfy the significance criterion
        if determine_significance(amp, fwhm, spectrum.rms_noise) < dct['significance']:
            remove_indices.append(i)
            quality_control.append(3)
            continue

        #  If the Gaussian component was fit outside the determined signal ranges, we check the significance of signal feature fitted by the Gaussian component. We remove the Gaussian component if the signal feature does not satisfy the significance criterion.
        if (offset < np.min(spectrum.channels)) or (offset > np.max(spectrum.channels)):
            remove_indices.append(i)
            quality_control.append(4)
            continue

        if spectrum.signal_intervals:
            if not any(low <= offset <= upp for low, upp in spectrum.signal_intervals):
                low = max(0, int(offset - fwhm))
                upp = int(offset + fwhm) + 2

                if not check_if_intervals_contain_signal(
                        spectrum=spectrum.intensity_values,
                        rms=spectrum.rms_noise,
                        ranges=[(low, upp)],
                        snr=dct['snr'],
                        significance=dct['significance']):
                    remove_indices.append(i)
                    quality_control.append(5)
                    continue

    if dct['max_ncomps'] is not None:
        remove_indices, quality_control = _remove_components_above_max_ncomps(
            amps_fit=amps_fit,
            fwhms_fit=fwhms_fit,
            ncomps_max=dct['max_ncomps'],
            remove_indices=remove_indices,
            quality_control=quality_control
        )

    remove_indices = list(set(remove_indices))

    refit = False

    if len(remove_indices) > 0:
        amps_fit, fwhms_fit, offsets_fit = remove_components_from_sublists(
            lst=[amps_fit, fwhms_fit, offsets_fit],
            remove_indices=remove_indices)
        params_fit = amps_fit + fwhms_fit + offsets_fit

        amps_errs, fwhms_errs, offsets_errs = remove_components_from_sublists(
            lst=[amps_errs, fwhms_errs, offsets_errs],
            remove_indices=remove_indices)
        params_errs = amps_errs + fwhms_errs + offsets_errs

        if params_min is not None:
            amps_min, fwhms_min, offsets_min = remove_components_from_sublists(
                lst=[amps_min, fwhms_min, offsets_min],
                remove_indices=remove_indices)
            params_min = amps_min + fwhms_min + offsets_min

        if params_max is not None:
            amps_max, fwhms_max, offsets_max = remove_components_from_sublists(
                lst=[amps_max, fwhms_max, offsets_max],
                remove_indices=remove_indices
            )
            params_max = amps_max + fwhms_max + offsets_max

        params_fit, params_errs, ncomps_fit = _perform_least_squares_fit(
            spectrum=spectrum,
            params_fit=params_fit,
            dct=dct,
            params_min=params_min,
            params_max=params_max
        )

        refit = True

    return params_fit, params_errs, ncomps_fit, params_min, params_max, quality_control, refit


def _check_which_gaussian_contains_feature(idx_low: int,
                                           idx_upp: int,
                                           fwhms_fit: List,
                                           offsets_fit: List) -> Optional[int]:
    """Return index of Gaussian component contained within a range in the spectrum.

    The FWHM interval (mean - FWHM, mean + FWHM ) of the Gaussian component has to be fully contained in the range of
    the spectrum. If no Gaussians satisfy this criterion 'None' is returned. In case multiple Gaussians satisfy this
    criterion, the Gaussian with the highest FWHM parameter is selected.

    Parameters
    ----------
    idx_low : Index of the first channel of the range in the spectrum.
    idx_upp : Index of the last channel of the range in the spectrum.
    fwhms_fit : List containing FWHM values of all N fitted Gaussian components in the form [fwhm1, ..., fwhmN].
    offsets_fit : List containing mean position values of all N fitted Gaussian components in the form
        [mean1, ..., meanN].

    Returns
    -------
    index : Index of Gaussian component contained within the spectral range.

    """
    lower = [int(offset - fwhm) for fwhm, offset in zip(fwhms_fit, offsets_fit)]
    lower = np.array([max(x, 0) for x in lower])
    upper = np.array([int(offset + fwhm) + 2 for fwhm, offset in zip(fwhms_fit, offsets_fit)])

    indices = np.arange(len(fwhms_fit))
    conditions = np.logical_and(lower <= idx_low, upper >= idx_upp)

    if np.count_nonzero(conditions) == 0:
        return None
    elif np.count_nonzero(conditions) == 1:
        return int(indices[conditions])
    else:
        remaining_indices = indices[conditions]
        select = np.argmax(np.array(fwhms_fit)[remaining_indices])
        return int(remaining_indices[select])


def _replace_gaussian_with_two_new_ones(spectrum: namedtuple,
                                        snr: float,
                                        significance: float,
                                        params_fit: List,
                                        exclude_idx: int,
                                        offset: float) -> List:
    """Replace broad Gaussian fit component with initial guesses for two narrower components.

    Parameters
    ----------
    snr : Required minimum signal-to-noise ratio for data peak.
    significance : Required minimum value for significance criterion.
    params_fit : Parameter vector in the form of [amp1, ..., ampN, fwhm1, ..., fwhmN, mean1, ..., meanN].
    exclude_idx : Index of the broad Gaussian fit component that will be removed from params_fit.
    offset : Mean position of the broad Gaussian fit component that will be removed.

    Returns
    -------
    params_fit : Updated list from which the parameters of the broad Gaussian fit components were removed and the
        parameters of the two narrower components were added.

    """
    ncomps_fit = number_of_gaussian_components(params=params_fit)
    amps_fit, fwhms_fit, offsets_fit = split_params(params=params_fit, ncomps=ncomps_fit)

    # TODO: check if this is still necessary?
    if exclude_idx is None:
        return params_fit

    #  remove the broad Gaussian component from the fit parameter list and determine new residual

    idx_low_residual = max(0, int(
        offsets_fit[exclude_idx] - fwhms_fit[exclude_idx]))
    idx_upp_residual = int(
        offsets_fit[exclude_idx] + fwhms_fit[exclude_idx]) + 2

    mask = np.arange(len(amps_fit)) == exclude_idx
    amps_fit = np.array(amps_fit)[~mask]
    fwhms_fit = np.array(fwhms_fit)[~mask]
    offsets_fit = np.array(offsets_fit)[~mask]

    residual = spectrum.intensity_values - combined_gaussian(amps=amps_fit,
                                                             fwhms=fwhms_fit,
                                                             means=offsets_fit,
                                                             x=spectrum.channels)

    #  search for residual peaks in new residual

    for low, upp in zip([idx_low_residual, int(offset)],
                        [int(offset), idx_upp_residual]):
        amp_guess, fwhm_guess, offset_guess = _get_initial_guesses(
            residual=residual[low:upp],
            rms=spectrum.rms_noise,
            snr=snr,
            significance=significance,
            peak='positive',
            maximum=True
        )

        if amp_guess.size == 0:
            continue

        amps_fit, fwhms_fit, offsets_fit = list(amps_fit), list(fwhms_fit), list(offsets_fit)

        amps_fit.append(amp_guess)
        fwhms_fit.append(fwhm_guess)
        offsets_fit.append(offset_guess + low)

    return amps_fit + fwhms_fit + offsets_fit


def _get_initial_guesses(residual: np.ndarray,
                         rms: float,
                         snr: float,
                         significance: float,
                         peak: Literal['positive', 'negative'] = 'positive',
                         maximum: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Get initial guesses of Gaussian fit parameters for residual peaks.

    Parameters
    ----------
    residual : Residual of the spectrum in which we search for unfit peaks.
    rms : Root-mean-square noise of the spectrum.
    snr : Required minimum signal-to-noise ratio for data peak.
    significance : Required minimum value for significance criterion.
    peak : Whether to search for positive (default) or negative peaks in the residual.
    maximum : Default is 'False'. If set to 'True', only the input parameter guesses for a single Gaussian fit
        component -- the one with the highest guessed amplitude value -- are returned.

    Returns
    -------
    amp_guesses : Initial guesses for amplitude values of Gaussian fit parameters for residual peaks.
    fwhm_guesses : Initial guesses for FWHM values of Gaussian fit parameters for residual peaks.
    offset_guesses : Initial guesses for mean positions of Gaussian fit parameters for residual peaks.

    """
    amp_guesses, ranges = determine_peaks(spectrum=residual, peak=peak, amp_threshold=snr*rms)

    if amp_guesses.size == 0:
        return np.array([]), np.array([]), np.array([])

    amp_guesses = amp_guesses

    sort = np.argsort(ranges[:, 0])
    amp_guesses = amp_guesses[sort]
    ranges = ranges[sort]

    #  determine whether identified residual peaks satisfy the significance criterion for signal peaks
    keep_indices = np.array([])
    significance_vals = np.array([])
    for i, (lower, upper) in enumerate(ranges):
        significance_val = np.sum(
            np.abs(residual[lower:upper])) / (np.sqrt(upper - lower)*rms)
        significance_vals = np.append(significance_vals, significance_val)
        if significance_val > significance:
            keep_indices = np.append(keep_indices, i)

    keep_indices = keep_indices.astype('int')
    amp_guesses = amp_guesses[keep_indices]
    significance_vals = significance_vals[keep_indices]

    if amp_guesses.size == 0:
        return np.array([]), np.array([]), np.array([])

    amp_guesses_position_mask = np.in1d(residual, amp_guesses)
    offset_guesses = np.where(amp_guesses_position_mask == True)[0]

    #  we use the determined significance values to get input guesses for the FWHM values
    # TODO: Where does this magic factor come from?
    fwhm_guesses = (significance_vals*rms / (amp_guesses * 0.75269184778925247))**2

    if maximum:
        idx_max = np.argmax(amp_guesses)
        amp_guesses = amp_guesses[idx_max]
        fwhm_guesses = fwhm_guesses[idx_max]
        offset_guesses = offset_guesses[idx_max]

    return amp_guesses, fwhm_guesses, offset_guesses


def get_fully_blended_gaussians(params_fit: List,
                                get_count: bool = False,
                                # TODO: Where does this magic factor come from?
                                separation_factor: float = 0.8493218002991817) -> Union[int, np.ndarray]:
    """Return information about blended Gaussian fit components.

    A Gaussian fit component i is blended with another component if the separation of their mean positions is less than the FWHM value of the narrower component multiplied by the 'separation_factor'.

    The default value for the separation_factor (0.8493218002991817) is based on the minimum required separation distance to distinguish two identical Gaussian peaks (= 2*std).

    Parameters
    ----------
    params_fit : list
        Parameter vector in the form of [amp1, ..., ampN, fwhm1, ..., fwhmN, mean1, ..., meanN].
    get_count : bool
        Default is 'False'. If set to 'True' only the number of blended components is returned.
    separation_factor : float
        The required minimum separation between two Gaussian components (mean1, fwhm1) and (mean2, fwhm2) is determined as separation_factor * min(fwhm1, fwhm2).

    Returns
    -------
    indices_blended : numpy.ndarray
        Indices of fitted Gaussian components that satisfy the criterion for blendedness, sorted from lowest to highest amplitude values.

    """
    ncomps_fit = number_of_gaussian_components(params_fit)
    amps_fit, fwhms_fit, offsets_fit = split_params(params_fit, ncomps_fit)
    # stddevs_fit = list(np.array(fwhms_fit) / CONVERSION_STD_TO_FWHM)
    indices_blended = np.array([])
    blended_pairs = []
    items = list(range(ncomps_fit))

    N_blended = 0

    for idx1, idx2 in itertools.combinations(items, 2):
        min_separation = min(
            fwhms_fit[idx1], fwhms_fit[idx2]) * separation_factor
        separation = abs(offsets_fit[idx1] - offsets_fit[idx2])

        if separation < min_separation:
            indices_blended = np.append(indices_blended, np.array([idx1, idx2]))
            blended_pairs.append([idx1, idx2])
            N_blended += 1

    if get_count:
        return N_blended

    indices_blended = np.unique(indices_blended).astype('int')
    #  sort the identified blended components from lowest to highest amplitude value
    sort = np.argsort(np.array(amps_fit)[indices_blended])

    return indices_blended[sort]


def _return_as_list(var: Any) -> List:
    if isinstance(var, np.ndarray):
        var = list(var)
    elif not isinstance(var, list):
        var = [var]
    return var


# TODO: Identical function in utils.grouping_functions -> remove redundancy; this function is imported also by
#  spatial_fitting.py
def remove_components(params_fit: List,
                      remove_indices: Union[int, List, np.ndarray]) -> List:
    """Remove parameters of Gaussian fit components.

    Parameters
    ----------
    params_fit : list
        Parameter vector in the form of [amp1, ..., ampN, fwhm1, ..., fwhmN, mean1, ..., meanN].
    remove_indices : int, list, np.ndarray
        Indices of Gaussian fit components, whose parameters should be removed from params_fit.

    Returns
    -------
    params_fit : list
        Updated list from which the parameters of the selected Gaussian fit components were removed.

    """
    ncomps_fit = number_of_gaussian_components(params_fit)
    amps_fit, fwhms_fit, offsets_fit = split_params(params=params_fit, ncomps=ncomps_fit)

    remove_indices = _return_as_list(remove_indices)

    amps_fit = [amp for idx, amp in enumerate(amps_fit) if idx not in remove_indices]
    fwhms_fit = [fwhm for idx, fwhm in enumerate(fwhms_fit) if idx not in remove_indices]
    offsets_fit = [offset for idx, offset in enumerate(offsets_fit) if idx not in remove_indices]
    return amps_fit + fwhms_fit + offsets_fit


def get_best_fit(spectrum: namedtuple,
                 params_fit: List,
                 dct: Dict,
                 first: bool = False,
                 best_fit_info: Optional[Dict] = None,
                 force_accept: bool = False,
                 params_min: Optional[List] = None,
                 params_max: Optional[List] = None,
                 ) -> Dict:
    """Determine new best fit for spectrum.

    If this is the first fit iteration for the spectrum a new best fit is assigned and its parameters are returned in best_fit_info.

    If it is not the first fit iteration, the new fit is compared to the current best fit supplied in best_fit_info. If the new fit is preferred (decided via the AICc criterion), the parameters of the new fit are returned in best_fit_info. Otherwise, the old best_fit_info is returned.

    Parameters
    ----------
    params_fit : list
        Parameter vector in the form of [amp1, ..., ampN, fwhm1, ..., fwhmN, mean1, ..., meanN].
    dct : dict
        Dictionary containing parameter settings for the improved fitting.
    first : bool
        Default is 'False'. If set to 'True', the new fit will be assigned as best fit and returned in best_fit_info.
    best_fit_info : Dictionary containing parameters of the chosen best fit for the spectrum.
    force_accept : bool
        Experimental feature. Default is 'False'. If set to 'True', the new fit will be forced to become the best fit.
    params_min : list
        List of minimum limits for parameters: [min_amp1, ..., min_ampN, min_fwhm1, ..., min_fwhmN, min_mean1, ..., min_meanN]
    params_max : list
        List of maximum limits for parameters: [max_amp1, ..., max_ampN, max_fwhm1, ..., max_fwhmN, max_mean1, ..., max_meanN]

    Returns
    -------
    best_fit_info : Dictionary containing parameters of the chosen best fit for the spectrum.

    """
    if not first:
        best_fit_info["new_fit"] = False
        quality_control = best_fit_info["quality_control"]
    else:
        quality_control = []

    params_fit, params_errs, ncomps_fit = _perform_least_squares_fit(
        spectrum=spectrum,
        params_fit=params_fit,
        dct=dct,
        params_min=params_min,
        params_max=params_max
    )

    #  check if fit components satisfy mandatory criteria
    if ncomps_fit > 0:
        refit = True
        while refit:
            params_fit, params_errs, ncomps_fit, params_min, params_max, quality_control, refit = _check_params_fit(
                spectrum=spectrum,
                params_fit=params_fit,
                params_errs=params_errs,
                dct=dct,
                quality_control=quality_control,
            )

        best_fit = multi_component_gaussian_model(spectrum.channels, *params_fit).ravel()
    else:
        best_fit = spectrum.intensity_values * 0

    rchi2, aicc = goodness_of_fit(
        data=spectrum.intensity_values,
        best_fit_final=best_fit,
        errors=spectrum.noise_values,
        ncomps_fit=ncomps_fit,
        mask=spectrum.signal_mask,
        get_aicc=True
    )

    residual = spectrum.intensity_values - best_fit

    pvalue = check_residual_for_normality(
        data=residual,
        errors=spectrum.noise_values,
        mask=spectrum.signal_mask,
        noise_spike_mask=spectrum.noise_spike_mask
    )

    #  return the list of best fit results if there was no old list of best fit results for comparison
    if first:
        return {
            "params_fit": params_fit,
            "params_errs": params_errs,
            "ncomps_fit": ncomps_fit,
            "best_fit_final": best_fit,
            "residual": residual,
            "rchi2": rchi2,
            "aicc": aicc,
            "new_fit": True,
            "params_min": params_min,
            "params_max": params_max,
            "pvalue": pvalue,
            "quality_control": quality_control
        }

    #  return new best_fit_info if the AICc value is smaller
    aicc_old = best_fit_info["aicc"]
    if ((aicc < aicc_old) and not np.isclose(aicc, aicc_old, atol=1e-1)) or force_accept:
        return {
            "params_fit": params_fit,
            "params_errs": params_errs,
            "ncomps_fit": ncomps_fit,
            "best_fit_final": best_fit,
            "residual": residual,
            "rchi2": rchi2,
            "aicc": aicc,
            "new_fit": True,
            "params_min": params_min,
            "params_max": params_max,
            "pvalue": pvalue,
            "quality_control": quality_control
        }

    #  return old best_fit_info if the aicc value is higher
    best_fit_info["new_fit"] = False
    return best_fit_info


def check_for_negative_residual(spectrum: namedtuple,
                                best_fit_info: Dict,
                                dct: Dict,
                                force_accept: bool = False,
                                get_count: bool = False,
                                get_idx: bool = False,
                                ) -> Union[int, Dict]:
    """Check for negative residual features and try to refit them.

    We define negative residual features as negative peaks in the residual that were introduced by the fit. These
    negative peaks have to have a minimum negative signal-to-noise ratio of dct['snr_negative'].

    In case of a negative residual feature, we try to replace the Gaussian fit component that is causing the feature
    with two narrower components. We only accept this solution if it yields a better fit as determined by the AICc
    value.

    Parameters
    ----------
    best_fit_info : Dictionary containing parameters of the chosen best fit for the spectrum.
    dct : Dictionary containing parameter settings for the improved fitting.
    force_accept : Experimental feature. Default is 'False'. If set to 'True', the new fit will be forced to become the
        best fit.
    get_count : Default is 'False'. If set to 'True', only the number of occurring negative residual features will be
        returned.
    get_idx : Default is 'False'. If set to 'True', the index of the Gaussian fit component causing the negative
        residual feature is returned. In case of multiple negative residual features, only the index of one of them is
        returned.

    Returns
    -------
    best_fit_info : Dictionary containing parameters of the chosen best fit for the spectrum.

    """
    if best_fit_info["ncomps_fit"] == 0:
        return 0 if get_count else best_fit_info

    residual = best_fit_info["residual"]

    amps_fit, fwhms_fit, offsets_fit = split_params(
        params=best_fit_info["params_fit"],
        ncomps=best_fit_info["ncomps_fit"]
    )

    amp_guesses, fwhm_guesses, offset_guesses = _get_initial_guesses(
        residual=residual,
        rms=spectrum.rms_noise,
        snr=dct['snr_negative'],
        significance=dct['significance'],
        peak='negative',
        maximum=False
    )

    #  check if negative residual feature was already present in the data
    remove_indices = [i for i, offset in enumerate(offset_guesses)
                      if residual[offset] > (spectrum.intensity_values[offset] - dct['snr'] * spectrum.rms_noise)]

    if remove_indices:
        amp_guesses, fwhm_guesses, offset_guesses = remove_components_from_sublists(
            [amp_guesses, fwhm_guesses, offset_guesses], remove_indices)

    if get_count:
        return len(amp_guesses)

    if len(amp_guesses) == 0:
        return best_fit_info

    #  in case of multiple negative residual features, sort them in order of increasing amplitude values
    sort = np.argsort(amp_guesses)
    amp_guesses = np.array(amp_guesses)[sort]
    fwhm_guesses = np.array(fwhm_guesses)[sort]
    offset_guesses = np.array(offset_guesses)[sort]

    for amp, fwhm, offset in zip(amp_guesses, fwhm_guesses, offset_guesses):
        exclude_idx = _check_which_gaussian_contains_feature(
            idx_low=max(0, int(offset - fwhm)),
            idx_upp=int(offset + fwhm) + 2,
            fwhms_fit=fwhms_fit,
            offsets_fit=offsets_fit)
        if get_idx:
            return exclude_idx
        if exclude_idx is None:
            continue

        params_fit = _replace_gaussian_with_two_new_ones(
            spectrum=spectrum,
            snr=dct['snr'],
            significance=dct['significance'],
            params_fit=best_fit_info["params_fit"],
            exclude_idx=exclude_idx,
            offset=offset
        )

        best_fit_info = get_best_fit(
            spectrum=spectrum,
            params_fit=params_fit,
            dct=dct,
            first=False,
            best_fit_info=best_fit_info,
            force_accept=force_accept,
        )

        # TODO: What's the purpose of the following three lines?
        params_fit = best_fit_info["params_fit"]
        ncomps_fit = best_fit_info["ncomps_fit"]
        amps_fit, fwhms_fit, offsets_fit = split_params(params_fit, ncomps_fit)
    return best_fit_info


def _try_fit_with_new_components(spectrum: namedtuple,
                                 best_fit_info: Dict,
                                 dct: Dict,
                                 exclude_idx: int,
                                 force_accept: bool = False,
                                 ) -> Dict:
    """Exclude Gaussian fit component and try fit with new initial guesses.

    First we try a new refit by just removing the component (i) and adding no new components. If this does not work we
    determine guesses for additional fit components from the residual that is produced if the component (i) is
    discarded and try a new fit. We only accept the new fit solution if it yields a better fit as determined by the
    AICc value.

    Parameters
    ----------
    best_fit_info : Dictionary containing parameters of the chosen best fit for the spectrum.
    dct : Dictionary containing parameter settings for the improved fitting.
    exclude_idx : Index of Gaussian fit component that will be removed.
    force_accept : Experimental feature. Default is 'False'. If set to 'True', the new fit will be forced to become the b
        est fit.

    Returns
    -------
    best_fit_info : Dictionary containing parameters of the chosen best fit for the spectrum.

    """
    #  produce new best fit with excluded components
    best_fit_info_new = get_best_fit(
        spectrum=spectrum,
        # exclude component from parameter list of components
        params_fit=remove_components(params_fit=best_fit_info["params_fit"], remove_indices=exclude_idx),
        dct=dct,
        first=True,
        best_fit_info=None,
        force_accept=force_accept,
    )

    #  return new best fit with excluded component if its AICc value is lower
    if ((best_fit_info_new["aicc"] < best_fit_info["aicc"]) and
            not np.isclose(best_fit_info_new["aicc"], best_fit_info["aicc"], atol=1e-1)):
        return best_fit_info_new

    #  search for new positive residual peaks

    amps_fit, fwhms_fit, offsets_fit = split_params(
        params=best_fit_info_new["params_fit"],
        ncomps=best_fit_info_new["ncomps_fit"]
    )

    amp_guesses, fwhm_guesses, offset_guesses = _get_initial_guesses(
        residual=spectrum.intensity_values - combined_gaussian(amps_fit, fwhms_fit, offsets_fit, spectrum.channels),
        rms=spectrum.rms_noise,
        snr=dct['snr'],
        significance=dct['significance'],
        peak='positive'
    )

    #  return original best fit list if there are no guesses for new components to fit in the residual
    if amp_guesses.size == 0:
        return best_fit_info

    #  get new best fit with additional components guessed from the residual
    amps_fit = list(amps_fit) + list(amp_guesses)
    fwhms_fit = list(fwhms_fit) + list(fwhm_guesses)
    offsets_fit = list(offsets_fit) + list(offset_guesses)

    params_fit_new = amps_fit + fwhms_fit + offsets_fit

    best_fit_info_new = get_best_fit(
        spectrum=spectrum,
        params_fit=params_fit_new,
        dct=dct,
        first=False,
        best_fit_info=best_fit_info_new,
        force_accept=force_accept,
    )

    #  return new best fit if its AICc value is lower
    if ((best_fit_info_new["aicc"] < best_fit_info["aicc"]) and
            not np.isclose(best_fit_info_new["aicc"], best_fit_info["aicc"], atol=1e-1)):
        return best_fit_info_new

    return best_fit_info


def _check_for_broad_feature(spectrum: namedtuple,
                             best_fit_info: Dict,
                             dct: Dict,
                             force_accept: bool = False,
                             ) -> Dict:
    """Check for broad features and try to refit them.

    We define broad fit components as having a FWHM value that is bigger by a factor of dct['fwhm_factor'] than the
    second broadest component in the spectrum.

    In case of a broad fit component, we first try to replace it with two narrower components. If this does not work,
    we determine guesses for additional fit components from the residual that is produced if the component (i) is
    discarded and try a new fit.

    We only accept a new fit solution if it yields a better fit as determined by the AICc value.

    If there is only one fit component in the spectrum, this check is not performed.

    Parameters
    ----------
    best_fit_info : Dictionary containing parameters of the chosen best fit for the spectrum.
    dct : Dictionary containing parameter settings for the improved fitting.
    force_accept : Experimental feature. Default is 'False'. If set to 'True', the new fit will be forced to become the
        best fit.

    Returns
    -------
    best_fit_info : Dictionary containing parameters of the chosen best fit for the spectrum.

    """
    best_fit_info["new_fit"] = False

    if best_fit_info["ncomps_fit"] < 2 and dct['fwhm_factor'] > 0:
        return best_fit_info

    fwhms_fit, offsets_fit = split_params(
        params=best_fit_info["params_fit"], ncomps=best_fit_info["ncomps_fit"])[1:]

    fwhms_sorted = sorted(fwhms_fit)
    if (fwhms_sorted[-1] < dct['fwhm_factor'] * fwhms_sorted[-2]):
        return best_fit_info

    exclude_idx = np.argmax(np.array(fwhms_fit))

    params_fit = _replace_gaussian_with_two_new_ones(
        spectrum=spectrum,
        snr=dct['snr'],
        significance=dct['significance'],
        params_fit=best_fit_info["params_fit"],
        exclude_idx=exclude_idx,
        offset=offsets_fit[exclude_idx]
    )

    if len(params_fit) > 0:
        best_fit_info = get_best_fit(
            spectrum=spectrum,
            params_fit=params_fit,
            dct=dct,
            first=False,
            best_fit_info=best_fit_info,
            force_accept=force_accept,
        )

    if best_fit_info["new_fit"]:
        return best_fit_info

    if best_fit_info["ncomps_fit"] == 0:
        return best_fit_info

    fwhms_fit = split_params(params=best_fit_info["params_fit"], ncomps=best_fit_info["ncomps_fit"])[1]

    return _try_fit_with_new_components(
        spectrum=spectrum,
        best_fit_info=best_fit_info,
        dct=dct,
        exclude_idx=np.argmax(np.array(fwhms_fit)),
        force_accept=force_accept,
    )


def _check_for_blended_feature(spectrum: namedtuple,
                               best_fit_info: Dict,
                               dct: Dict,
                               force_accept: bool = False,
                               ) -> Dict:
    """Check for blended features and try to refit them.

    We define two fit components as blended if the mean position of one fit component is contained within the standard
    deviation interval (mean - std, mean + std) of another fit component.

    In case of blended fit components, we try to determine guesses for new fit components from the residual that is
    produced if one of the components is discarded and try a new fit. We start by excluding the fit component with the
    lowest amplitude value.

    We only accept a new fit solution if it yields a better fit as determined by the AICc value.

    Parameters
    ----------
    best_fit_info : Dictionary containing parameters of the chosen best fit for the spectrum.
    dct : Dictionary containing parameter settings for the improved fitting.
    force_accept : Experimental feature. Default is 'False'. If set to 'True', the new fit will be forced to become the
        best fit.

    Returns
    -------
    best_fit_info : Dictionary containing parameters of the chosen best fit for the spectrum.

    """
    if best_fit_info["ncomps_fit"] < 2:
        return best_fit_info

    exclude_indices = get_fully_blended_gaussians(
        params_fit=best_fit_info["params_fit"],
        get_count=False,
        separation_factor=dct['separation_factor']
    )

    #  skip if there are no blended features
    if exclude_indices.size == 0:
        return best_fit_info

    for exclude_idx in exclude_indices:
        best_fit_info = _try_fit_with_new_components(
            spectrum=spectrum,
            best_fit_info=best_fit_info,
            dct=dct,
            exclude_idx=exclude_idx,
            force_accept=force_accept,
        )
        if best_fit_info["new_fit"]:
            break

    return best_fit_info


def _get_dictionary_from_model(model):
    """Return best_fit_info dictionary from Model dataclass"""
    return {
        "params_fit": model.parameters,
        "params_errs": model.parameter_uncertainties,
        "ncomps_fit": model.n_components,
        "best_fit_final": model.modelled_intensity_values,
        "residual": model.residual,
        "rchi2": model.rchi2,
        "aicc": model.aicc,
        "new_fit": model.new_best_fit,
        "params_min": model.parameters_min_values,
        "params_max": model.parameters_max_values,
        "pvalue": model.pvalue,
        "quality_control": model.quality_control
    }


def _quality_check(spectrum: namedtuple, model, dct: Dict) -> Dict:
    """Quality check for GaussPy best fit results.

    All Gaussian fit components that are not satisfying the mandatory quality criteria get discarded from the fit.
    A dictionary containing parameters of the chosen best fit for the spectrum is returned.
    """
    return _get_dictionary_from_model(model) if model.n_components == 0 else get_best_fit(
        spectrum=spectrum,
        params_fit=model.parameters,
        dct=dct,
        first=True,
        best_fit_info=None,
    )


def check_for_peaks_in_residual(spectrum: namedtuple,
                                best_fit_info: Dict,
                                dct: Dict,
                                fitted_residual_peaks: List,
                                force_accept: bool = False,
                                params_min: Optional[List] = None,
                                params_max: Optional[List] = None,
                                ) -> Tuple[Dict, List]:
    """Try fit by adding new components, whose initial parameters were determined from residual peaks.

    Parameters
    ----------
    best_fit_info : Dictionary containing parameters of the chosen best fit for the spectrum.
    dct : Dictionary containing parameter settings for the improved fitting.
    fitted_residual_peaks : List of initial mean position guesses for new fit components determined from residual peaks
        that were already tried in previous iterations.
    force_accept : Experimental feature. Default is 'False'. If set to 'True', the new fit will be forced to become the
        best fit.
    params_min : List of minimum limits for parameters: [min_amp1, ..., min_ampN, min_fwhm1, ..., min_fwhmN,
        min_mean1, ..., min_meanN]
    params_max : List of maximum limits for parameters: [max_amp1, ..., max_ampN, max_fwhm1, ..., max_fwhmN,
        max_mean1, ..., max_meanN]

    Returns
    -------
    best_fit_info : Dictionary containing parameters of the chosen best fit for the spectrum.
    fitted_residual_peaks : Updated list of initial mean position guesses for new fit components determined from
        residual peaks.

    """
    #  TODO: remove params_min and params_max keywords
    amps_fit, fwhms_fit, offsets_fit = split_params(
        params=best_fit_info["params_fit"],
        ncomps=best_fit_info["ncomps_fit"]
    )

    amp_guesses, fwhm_guesses, offset_guesses = _get_initial_guesses(
        residual=best_fit_info["residual"],
        rms=spectrum.rms_noise,
        snr=dct['snr'],
        significance=dct['significance'],
        peak='positive'
    )

    if (amp_guesses.size == 0) or (list(offset_guesses) in fitted_residual_peaks):
        best_fit_info["new_fit"] = False
        return best_fit_info, fitted_residual_peaks

    fitted_residual_peaks.append(list(offset_guesses))

    amps_fit = list(amps_fit) + list(amp_guesses)
    fwhms_fit = list(fwhms_fit) + list(fwhm_guesses)
    offsets_fit = list(offsets_fit) + list(offset_guesses)

    best_fit_info = get_best_fit(
        spectrum=spectrum,
        params_fit=amps_fit + fwhms_fit + offsets_fit,
        dct=dct,
        first=False,
        best_fit_info=best_fit_info,
        force_accept=force_accept,
        params_min=params_min,
        params_max=params_max,
    )

    return best_fit_info, fitted_residual_peaks


def _log_new_fit(new_fit: bool,
                 log_gplus: List,
                 mode: Literal['positive_residual_peak', 'negative_residual_peak', 'broad', 'blended'] = 'residual'
                 ) -> List:
    """Log the successful refits of a spectrum.

    Parameters
    ----------
    new_fit : If 'True', the spectrum was successfully refit.
    log_gplus : Log of all previous successful refits of the spectrum.
    mode : Specifies the feature that was refit or used for a new successful refit.

    Returns
    -------
    log_gplus : Updated log of successful refits of the spectrum.

    """
    if not new_fit:
        return log_gplus

    modes = {'positive_residual_peak': 1, 'negative_residual_peak': 2, 'broad': 3, 'blended': 4}
    log_gplus.append(modes[mode])
    return log_gplus


def try_to_improve_fitting(vel: np.ndarray,
                           data: np.ndarray,
                           errors: np.ndarray,
                           params_fit: List,
                           ncomps_fit: int,
                           dct: Dict,
                           signal_ranges: Optional[List] = None,
                           noise_spike_ranges: Optional[List] = None) -> Tuple[Dict, int, int, List]:
    """Short summary.

    Parameters
    ----------
    vel : Velocity channels (unitless).
    data : Original data of spectrum.
    errors : Root-mean-square noise values.
    params_fit : Parameter vector in the form of [amp1, ..., ampN, fwhm1, ..., fwhmN, mean1, ..., meanN]. Corresponds
        to the final best fit results of the GaussPy decomposition.
    ncomps_fit : Number of fitted Gaussian components.
    dct : Dictionary containing parameter settings for the improved fitting.
    signal_ranges : Nested list containing info about ranges of the spectrum that were estimated to contain signal. The
        goodness-of-fit calculations are only performed for the spectral channels within these ranges.
    noise_spike_ranges : Nested list containing info about ranges of the spectrum that were estimated to contain noise
        spike features. These will get masked out from goodness-of-fit calculations.

    Returns
    -------
    best_fit_info : Dictionary containing parameters of the chosen best fit for the spectrum.
    N_neg_res_peak : Number of negative residual features that occur in the best fit of the spectrum.
    N_blended : Number of blended Gaussian components that occur in the best fit of the spectrum.
    log_gplus : Log of all successful refits of the spectrum.

    """
    spectrum = Spectrum(
        intensity_values=data,
        channels=vel,
        rms_noise=errors[0],
        signal_intervals=signal_ranges,
        noise_spike_intervals=noise_spike_ranges
    )
    model = Model(spectrum=spectrum)
    model.parameters = params_fit

    #  Check the quality of the final fit from GaussPy
    best_fit_info: Dict = _quality_check(spectrum=spectrum, model=model, dct=dct)

    #  Try to improve fit by searching for peaks in the residual
    first_run = True
    fitted_residual_peaks = []
    log_gplus = []

    # while (rchi2 > dct['rchi2_limit']) or first_run:
    while (best_fit_info["pvalue"] < dct['min_pvalue']) or first_run:
        new_fit = True
        new_peaks = False

        count_old = len(fitted_residual_peaks)
        while new_fit:
            best_fit_info["new_fit"] = False
            best_fit_info, fitted_residual_peaks = check_for_peaks_in_residual(
                spectrum=spectrum,
                best_fit_info=best_fit_info,
                dct=dct,
                fitted_residual_peaks=fitted_residual_peaks,
            )
            new_fit = best_fit_info["new_fit"]
            log_gplus = _log_new_fit(new_fit=new_fit, log_gplus=log_gplus, mode='positive_residual_peak')
        count_new = len(fitted_residual_peaks)

        if count_old != count_new:
            new_peaks = True

        #  stop refitting loop if no new peaks were fit from the residual
        if (not first_run and not new_peaks) or (best_fit_info["ncomps_fit"] == 0):
            break

        #  try to refit negative residual feature
        if dct['neg_res_peak']:
            best_fit_info = check_for_negative_residual(
                spectrum=spectrum,
                best_fit_info=best_fit_info,
                dct=dct,
            )
            new_fit = best_fit_info["new_fit"]
            log_gplus = _log_new_fit(new_fit=new_fit, log_gplus=log_gplus, mode='negative_residual_peak')

        #  try to refit broad Gaussian components
        if dct['broad']:
            new_fit = True
            while new_fit:
                best_fit_info["new_fit"] = False
                best_fit_info = _check_for_broad_feature(
                    spectrum=spectrum,
                    best_fit_info=best_fit_info,
                    dct=dct,
                )
                new_fit = best_fit_info["new_fit"]
                log_gplus = _log_new_fit(new_fit=new_fit, log_gplus=log_gplus, mode='broad')

        #  try to refit blended Gaussian components
        if dct['blended']:
            new_fit = True
            while new_fit:
                best_fit_info["new_fit"] = False
                best_fit_info = _check_for_blended_feature(
                    spectrum=spectrum,
                    best_fit_info=best_fit_info,
                    dct=dct,
                )
                new_fit = best_fit_info["new_fit"]
                log_gplus = _log_new_fit(new_fit=new_fit, log_gplus=log_gplus, mode='blended')

        if not first_run:
            break
        first_run = False

    N_neg_res_peak = check_for_negative_residual(
        spectrum=spectrum,
        best_fit_info=best_fit_info,
        dct=dct,
        get_count=True
    )

    params_fit = best_fit_info["params_fit"]
    N_blended = get_fully_blended_gaussians(
        params_fit=params_fit,
        get_count=True,
        separation_factor=dct['separation_factor']
    )

    return best_fit_info, N_neg_res_peak, N_blended, log_gplus


if __name__ == "__main__":
    model = Model()
    model.parameters = [1, 10, 20]
    pass
