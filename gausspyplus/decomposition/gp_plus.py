from collections import namedtuple
from typing import Optional, List, Union, Tuple, Literal, Dict

import numpy as np

from lmfit import minimize as lmfit_minimize

from gausspyplus.definitions.definitions import SettingsImproveFit
from gausspyplus.definitions.model import Model
from gausspyplus.preparation.determine_intervals import (
    check_if_intervals_contain_signal,
    get_slice_indices_for_interval,
)
from gausspyplus.decomposition.fit_quality_checks import (
    determine_significance,
    get_indices_of_fully_blended_gaussians,
    get_number_of_fully_blended_gaussians,
)
from gausspyplus.decomposition.gaussian_functions import (
    multi_component_gaussian_model,
    area_of_gaussian,
    split_params,
    number_of_gaussian_components,
    vals_vec_from_lmfit,
    errs_vec_from_lmfit,
    paramvec_to_lmfit,
    sort_parameters,
)
from gausspyplus.utils.misc import remove_elements_at_indices
from gausspyplus.preparation.noise_estimation import determine_peaks


def _perform_least_squares_fit(
    spectrum: namedtuple,
    params_fit: List,
    settings_improve_fit: SettingsImproveFit,
    params_min: Optional[List] = None,
    params_max: Optional[List] = None,
) -> Tuple[List, List, int]:
    # Objective functions for final fit
    def objective_leastsq(paramslm):
        params = vals_vec_from_lmfit(paramslm)
        resids = (
            multi_component_gaussian_model(*np.split(np.array(params), 3), spectrum.channels).ravel()
            - spectrum.intensity_values.ravel()
        ) / spectrum.noise_values
        return resids

    # Get new best fit
    lmfit_params = paramvec_to_lmfit(
        paramvec=params_fit,
        max_amp=settings_improve_fit.max_amp,
        max_fwhm=None,
        params_min=params_min,
        params_max=params_max,
    )
    try:
        result = lmfit_minimize(objective_leastsq, lmfit_params, method="leastsq")
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


def _remove_components_above_max_ncomps(
    params_fit: List,
    ncomps_max: int,
    remove_indices: List[int],
    quality_control: List[int],
) -> Tuple[List[int], List[int]]:
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
    amps_fit, fwhms_fit, _ = np.split(np.array(params_fit), 3)
    ncomps_fit = len(amps_fit)
    if ncomps_fit <= ncomps_max:
        return remove_indices, quality_control
    integrated_intensities = area_of_gaussian(amp=amps_fit, fwhm=fwhms_fit)
    sort_indices = np.argsort(integrated_intensities)

    for index in np.arange(ncomps_fit)[sort_indices]:
        if index in remove_indices:
            continue
        remove_indices.append(index)
        quality_control.append(6)
        remaining_ncomps = ncomps_fit - len(remove_indices)
        if remaining_ncomps <= ncomps_max:
            break

    return remove_indices, quality_control


def _check_params_fit(
    spectrum: namedtuple,
    params_fit: List,
    params_errs: List,
    settings: SettingsImproveFit,
    quality_control: List[int],
    params_min: Optional[List] = None,
    params_max: Optional[List] = None,
) -> Tuple[List, List, int, List, List, List, bool]:
    """Perform quality checks for the fitted Gaussians components.

    All Gaussian components that are not satisfying the criteria are discarded from the fit.

    Parameters
    ----------
    params_fit : Parameter vector in the form of [amp1, ..., ampN, fwhm1, ..., fwhmN, mean1, ..., meanN].
    params_errs : Parameter error vector in the form of [e_amp1, ..., e_ampN, e_fwhm1, ..., e_fwhmN,
        e_mean1, ..., e_meanN].
    settings_improve_fit : Dataclass containing parameter settings for the improved fitting.
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
    # Check if Gaussian components satisfy quality criteria
    remove_indices = []
    for i, (amp, fwhm, offset) in enumerate(zip(*np.split(np.array(params_fit), 3))):
        if settings.max_fwhm is not None and fwhm > settings.max_fwhm:
            remove_indices.append(i)
            quality_control.append(0)
            continue

        if settings.min_fwhm is not None and fwhm < settings.min_fwhm:
            remove_indices.append(i)
            quality_control.append(1)
            continue

        # Discard the Gaussian component if its amplitude value does not satisfy the required minimum S/N value or is
        # larger than the limit
        if amp < settings.snr_fit * spectrum.rms_noise:
            remove_indices.append(i)
            quality_control.append(2)
            continue

        # Discard the Gaussian component if it does not satisfy the significance criterion
        if determine_significance(amp, fwhm, spectrum.rms_noise) < settings.significance:
            remove_indices.append(i)
            quality_control.append(3)
            continue

        # If the Gaussian component was fit outside the determined signal ranges, we check the significance of signal
        # feature fitted by the Gaussian component. We remove the Gaussian component if the signal feature does not
        # satisfy the significance criterion.
        if (offset < np.min(spectrum.channels)) or (offset > np.max(spectrum.channels)):
            remove_indices.append(i)
            quality_control.append(4)
            continue

        if spectrum.signal_intervals and not any(low <= offset <= upp for low, upp in spectrum.signal_intervals):
            low, upp = get_slice_indices_for_interval(interval_center=offset, interval_half_width=fwhm)

            if not check_if_intervals_contain_signal(
                spectrum=spectrum.intensity_values,
                rms=spectrum.rms_noise,
                ranges=[(low, upp)],
                snr=settings.snr,
                significance=settings.significance,
            ):
                remove_indices.append(i)
                quality_control.append(5)

    if settings.max_ncomps is not None:
        remove_indices, quality_control = _remove_components_above_max_ncomps(
            params_fit=params_fit,
            ncomps_max=settings.max_ncomps,
            remove_indices=remove_indices,
            quality_control=quality_control,
        )

    refit = False

    if indices := list(set(remove_indices)):
        params_fit = remove_elements_at_indices(params_fit, indices, n_subarrays=3)

        if params_min is not None:
            params_min = remove_elements_at_indices(params_min, indices, n_subarrays=3)

        if params_max is not None:
            params_max = remove_elements_at_indices(params_max, indices, n_subarrays=3)

        params_fit, params_errs, ncomps_fit = _perform_least_squares_fit(
            spectrum=spectrum,
            params_fit=params_fit,
            settings_improve_fit=settings,
            params_min=params_min,
            params_max=params_max,
        )

        refit = True

    return (
        params_fit,
        params_errs,
        len(params_fit) // 3,
        params_min,
        params_max,
        quality_control,
        refit,
    )


def _check_which_gaussian_contains_feature(
    idx_low: int, idx_upp: int, fwhms_fit: List, offsets_fit: List
) -> Optional[int]:
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
    interval_bounds = [
        get_slice_indices_for_interval(interval_center=offset, interval_half_width=fwhm)
        for offset, fwhm in zip(offsets_fit, fwhms_fit)
    ]
    indices_of_contained_intervals = [
        idx for idx, (lower, upper) in enumerate(interval_bounds) if (lower <= idx_low and upper >= idx_upp)
    ]

    if not indices_of_contained_intervals:
        return
    elif len(indices_of_contained_intervals) == 1:
        return indices_of_contained_intervals[0]
    else:
        idx_largest_interval = np.argmax(np.array(fwhms_fit)[indices_of_contained_intervals])
        return indices_of_contained_intervals[idx_largest_interval]


def _replace_gaussian_with_two_new_ones(
    spectrum: namedtuple,
    snr: float,
    significance: float,
    params_fit: List,
    exclude_idx: int,
    offset: float,
) -> List:
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

    # Remove the broad Gaussian component from the fit parameter list and determine new residual

    idx_low_residual, idx_upp_residual = get_slice_indices_for_interval(
        interval_center=offsets_fit[exclude_idx],
        interval_half_width=fwhms_fit[exclude_idx],
    )

    amps_fit.pop(exclude_idx)
    fwhms_fit.pop(exclude_idx)
    offsets_fit.pop(exclude_idx)

    residual = spectrum.intensity_values - multi_component_gaussian_model(
        amps_fit, fwhms_fit, offsets_fit, spectrum.channels
    )

    # Search for residual peaks in new residual

    for low, upp in zip([idx_low_residual, int(offset)], [int(offset), idx_upp_residual]):
        amp_guesses, fwhm_guesses, offset_guesses = _get_initial_guesses(
            residual=residual[low:upp],
            rms=spectrum.rms_noise,
            snr=snr,
            significance=significance,
            peak="positive",
        )

        if amp_guesses.size == 0:
            continue

        # Use only the guess with the highest amplitude value.
        idx_max = np.argmax(amp_guesses)
        amps_fit.append(amp_guesses[idx_max])
        fwhms_fit.append(fwhm_guesses[idx_max])
        offsets_fit.append(offset_guesses[idx_max] + low)

    return amps_fit + fwhms_fit + offsets_fit


def _get_initial_guesses(
    residual: np.ndarray,
    rms: float,
    snr: float,
    significance: float,
    peak: Literal["positive", "negative"] = "positive",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Get initial guesses of Gaussian fit parameters for residual peaks.

    Parameters
    ----------
    residual : Residual of the spectrum in which we search for unfit peaks.
    rms : Root-mean-square noise of the spectrum.
    snr : Required minimum signal-to-noise ratio for data peak.
    significance : Required minimum value for significance criterion.
    peak : Whether to search for positive (default) or negative peaks in the residual.

    Returns
    -------
    amp_guesses : Initial guesses for amplitude values of Gaussian fit parameters for residual peaks.
    fwhm_guesses : Initial guesses for FWHM values of Gaussian fit parameters for residual peaks.
    offset_guesses : Initial guesses for mean positions of Gaussian fit parameters for residual peaks.

    """
    amp_vals_of_peaks, peak_intervals = determine_peaks(spectrum=residual, peak=peak, amp_threshold=snr * rms)

    if amp_vals_of_peaks.size == 0:
        return np.array([]), np.array([]), np.array([])

    sort = np.argsort(peak_intervals[:, 0])
    amp_vals_of_peaks = amp_vals_of_peaks[sort]
    peak_intervals = peak_intervals[sort]

    significance_values = np.array(
        [np.sum(np.abs(residual[lower:upper])) / (np.sqrt(upper - lower) * rms) for lower, upper in peak_intervals]
    )
    is_valid_peak = significance_values > significance

    if not any(is_valid_peak):
        return np.array([]), np.array([]), np.array([])

    peak_positions = np.flatnonzero(np.in1d(residual, amp_vals_of_peaks[is_valid_peak]))
    # We use the determined significance values to get input guesses for the FWHM values
    fwhm_guesses_for_peaks = (8 * np.log(2) / np.pi) * (
        significance_values[is_valid_peak] * rms / amp_vals_of_peaks[is_valid_peak]
    ) ** 2

    return amp_vals_of_peaks[is_valid_peak], fwhm_guesses_for_peaks, peak_positions


def get_best_fit_model(
    model: Model,  # model is a new instance of Model
    params_fit: List,
    settings_improve_fit: SettingsImproveFit,
    params_min: Optional[List] = None,
    params_max: Optional[List] = None,
) -> Model:
    """Determine new best fit for spectrum.

    Parameters
    ----------
    params_fit : list
        Parameter vector in the form of [amp1, ..., ampN, fwhm1, ..., fwhmN, mean1, ..., meanN].
    settings_improve_fit : dict
        Dataclass containing parameter settings for the improved fitting.
    params_min : list
        List of minimum limits for parameters: [min_amp1, ..., min_ampN, min_fwhm1, ..., min_fwhmN, min_mean1, ...,
        min_meanN]
    params_max : list
        List of maximum limits for parameters: [max_amp1, ..., max_ampN, max_fwhm1, ..., max_fwhmN, max_mean1, ...,
        max_meanN]

    Returns
    -------
    model : Best fit model

    """

    quality_control = model.quality_control

    params_fit, params_errs, ncomps_fit = _perform_least_squares_fit(
        spectrum=model.spectrum,
        params_fit=params_fit,
        settings_improve_fit=settings_improve_fit,
        params_min=params_min,
        params_max=params_max,
    )

    # Check if fit components satisfy mandatory criteria
    if ncomps_fit > 0:
        refit = True
        while refit:
            (params_fit, params_errs, ncomps_fit, params_min, params_max, quality_control, refit,) = _check_params_fit(
                spectrum=model.spectrum,
                params_fit=params_fit,
                params_errs=params_errs,
                settings=settings_improve_fit,
                quality_control=quality_control,
            )

    model.parameters = params_fit
    model.parameter_uncertainties = params_errs
    model.parameters_min_values = params_min
    model.parameters_max_values = params_max
    model.quality_control = quality_control
    model.new_best_fit = True
    return model


def choose_better_model_based_on_aicc(old_model: Model, new_model: Model) -> Model:
    if (new_model.aicc < old_model.aicc) and not np.isclose(new_model.aicc, old_model.aicc, atol=1e-1):
        new_model.new_best_fit = True
        new_model.quality_control = old_model.quality_control + new_model.quality_control
        return new_model
    else:
        old_model.new_best_fit = False
        old_model.quality_control = old_model.quality_control + new_model.quality_control
        return old_model


def check_for_negative_residual(
    model: Model,
    settings_improve_fit: SettingsImproveFit,
    get_count: bool = False,
    get_idx: bool = False,
) -> Union[Optional[int], Model]:
    """Check for negative residual features and try to refit them.

    We define negative residual features as negative peaks in the residual that were introduced by the fit. These
    negative peaks have to have a minimum negative signal-to-noise ratio of settings_improve_fit['snr_negative'].

    In case of a negative residual feature, we try to replace the Gaussian fit component that is causing the feature
    with two narrower components. We only accept this solution if it yields a better fit as determined by the AICc
    value.

    Parameters
    ----------
    model : Current best fit model
    settings_improve_fit : Dataclass containing parameter settings for the improved fitting.
    get_count : Default is 'False'. If set to 'True', only the number of occurring negative residual features will be
        returned.
    get_idx : Default is 'False'. If set to 'True', the index of the Gaussian fit component causing the negative
        residual feature is returned. In case of multiple negative residual features, only the index of one of them is
        returned.

    Returns
    -------
    model : Best fit model

    """
    # TODO: Rework the whole get_idx functionality
    if model.n_components == 0:
        if get_count:
            return 0
        elif get_idx:
            return None
        else:
            return model

    amp_guesses, fwhm_guesses, offset_guesses = _get_initial_guesses(
        residual=model.residual,
        rms=model.spectrum.rms_noise,
        snr=settings_improve_fit.snr_negative,
        significance=settings_improve_fit.significance,
        peak="negative",
    )

    # Check if negative residual feature was already present in the data
    remove_indices = [
        i
        for i, offset in enumerate(offset_guesses)
        if model.residual[offset]
        > (model.spectrum.intensity_values[offset] - settings_improve_fit.snr * model.spectrum.rms_noise)
    ]

    if remove_indices:
        amp_guesses = remove_elements_at_indices(amp_guesses, remove_indices)
        fwhm_guesses = remove_elements_at_indices(fwhm_guesses, remove_indices)
        offset_guesses = remove_elements_at_indices(offset_guesses, remove_indices)

    if get_count:
        return len(amp_guesses)

    if len(amp_guesses) == 0:
        return model

    # In case of multiple negative residual features, sort them in order of increasing amplitude values
    sorted_parameters = sort_parameters(amps=amp_guesses, fwhms=fwhm_guesses, means=offset_guesses, descending=False)
    if get_idx and sorted_parameters.size == 0:
        return None

    for amp, fwhm, offset in zip(*np.split(sorted_parameters, 3)):
        idx_low, idx_up = get_slice_indices_for_interval(interval_center=offset, interval_half_width=fwhm)
        exclude_idx = _check_which_gaussian_contains_feature(
            idx_low=idx_low,
            idx_upp=idx_up,
            fwhms_fit=model.fwhms,
            offsets_fit=model.means,
        )
        if get_idx:
            return exclude_idx
        if exclude_idx is None:
            continue

        params_fit = _replace_gaussian_with_two_new_ones(
            spectrum=model.spectrum,
            snr=settings_improve_fit.snr,
            significance=settings_improve_fit.significance,
            params_fit=model.parameters,
            exclude_idx=exclude_idx,
            offset=offset,
        )

        new_model = get_best_fit_model(
            model=Model(spectrum=model.spectrum),
            params_fit=params_fit,
            settings_improve_fit=settings_improve_fit,
        )
    return choose_better_model_based_on_aicc(old_model=model, new_model=new_model)


def _try_fit_with_new_components(model: Model, settings_improve_fit: SettingsImproveFit, exclude_idx: int) -> Model:
    """Exclude Gaussian fit component and try fit with new initial guesses.

    First we try a new refit by just removing the component (i) and adding no new components. If this does not work we
    determine guesses for additional fit components from the residual that is produced if the component (i) is
    discarded and try a new fit. We only accept the new fit solution if it yields a better fit as determined by the
    AICc value.

    Parameters
    ----------
    model
    settings_improve_fit : Dataclass containing parameter settings for the improved fitting.
    exclude_idx : Index of Gaussian fit component that will be removed.

    Returns
    -------
    model : Best fit model

    """
    # Produce new best fit with excluded components
    new_model = get_best_fit_model(
        model=Model(spectrum=model.spectrum),
        params_fit=remove_elements_at_indices(
            array=model.parameters,
            indices=exclude_idx,
            n_subarrays=3,
        ),
        settings_improve_fit=settings_improve_fit,
    )

    model = choose_better_model_based_on_aicc(old_model=model, new_model=new_model)
    if model.new_best_fit:
        return model

    # Search for new positive residual peaks

    amp_guesses, fwhm_guesses, offset_guesses = _get_initial_guesses(
        residual=new_model.residual,
        rms=new_model.spectrum.rms_noise,
        snr=settings_improve_fit.snr,
        significance=settings_improve_fit.significance,
        peak="positive",
    )

    # Return original best fit list if there are no guesses for new components to fit in the residual
    if amp_guesses.size == 0:
        return model

    # Get new best fit with additional components guessed from the residual

    new_model = get_best_fit_model(
        model=Model(spectrum=model.spectrum),
        params_fit=(
            new_model.amps
            + list(amp_guesses)
            + new_model.fwhms
            + list(fwhm_guesses)
            + new_model.means
            + list(offset_guesses)
        ),
        settings_improve_fit=settings_improve_fit,
    )

    return choose_better_model_based_on_aicc(old_model=model, new_model=new_model)


def _check_for_broad_feature(model: Model, settings_improve_fit: SettingsImproveFit) -> Model:
    """Check for broad features and try to refit them.

    We define broad fit components as having a FWHM value that is bigger by a factor of
    settings_improve_fit['fwhm_factor'] than the second-broadest component in the spectrum.

    In case of a broad fit component, we first try to replace it with two narrower components. If this does not work,
    we determine guesses for additional fit components from the residual that is produced if the component (i) is
    discarded and try a new fit.

    We only accept a new fit solution if it yields a better fit as determined by the AICc value.

    If there is only one fit component in the spectrum, this check is not performed.

    Parameters
    ----------
    model
    settings_improve_fit : Dataclass containing parameter settings for the improved fitting.

    Returns
    -------
    model : Best fit model

    """
    model.new_best_fit = False

    if model.n_components < 2 and settings_improve_fit.fwhm_factor > 0:
        return model

    fwhms_sorted = sorted(model.fwhms)
    if fwhms_sorted[-1] < settings_improve_fit.fwhm_factor * fwhms_sorted[-2]:
        return model

    exclude_idx = np.argmax(model.fwhms)

    params_fit = _replace_gaussian_with_two_new_ones(
        spectrum=model.spectrum,
        snr=settings_improve_fit.snr,
        significance=settings_improve_fit.significance,
        params_fit=model.parameters,
        exclude_idx=exclude_idx,
        offset=model.means[exclude_idx],
    )

    if len(params_fit) > 0:
        new_model = get_best_fit_model(
            model=Model(spectrum=model.spectrum),
            params_fit=params_fit,
            settings_improve_fit=settings_improve_fit,
        )

        model = choose_better_model_based_on_aicc(old_model=model, new_model=new_model)

    if model.new_best_fit or model.n_components == 0:
        return model

    return _try_fit_with_new_components(
        model=model,
        settings_improve_fit=settings_improve_fit,
        exclude_idx=np.argmax(model.fwhms),
    )


def _check_for_blended_feature(model: Model, settings_improve_fit: SettingsImproveFit) -> Model:
    """Check for blended features and try to refit them.

    We define two fit components as blended if the mean position of one fit component is contained within the standard
    deviation interval (mean - std, mean + std) of another fit component.

    In case of blended fit components, we try to determine guesses for new fit components from the residual that is
    produced if one of the components is discarded and try a new fit. We start by excluding the fit component with the
    lowest amplitude value.

    We only accept a new fit solution if it yields a better fit as determined by the AICc value.

    Parameters
    ----------
    model
    settings_improve_fit : Dataclass containing parameter settings for the improved fitting.

    Returns
    -------
    model : Best fit model

    """
    if model.n_components < 2:
        return model

    exclude_indices = get_indices_of_fully_blended_gaussians(
        params_fit=model.parameters,
        separation_factor=settings_improve_fit.separation_factor,
    )

    for exclude_idx in exclude_indices:
        model = _try_fit_with_new_components(
            model=model,
            settings_improve_fit=settings_improve_fit,
            exclude_idx=exclude_idx,
        )
        if model.new_best_fit:
            break

    return model


def check_for_peaks_in_residual(
    model: Model,
    settings_improve_fit: SettingsImproveFit,
    fitted_residual_peaks: List,
    params_min: Optional[List] = None,
    params_max: Optional[List] = None,
) -> Tuple[Model, List]:
    """Try fit by adding new components, whose initial parameters were determined from residual peaks.

    Parameters
    ----------
    model
    settings_improve_fit : Dataclass containing parameter settings for the improved fitting.
    fitted_residual_peaks : List of initial mean position guesses for new fit components determined from residual peaks
        that were already tried in previous iterations.
    params_min : List of minimum limits for parameters: [min_amp1, ..., min_ampN, min_fwhm1, ..., min_fwhmN,
        min_mean1, ..., min_meanN]
    params_max : List of maximum limits for parameters: [max_amp1, ..., max_ampN, max_fwhm1, ..., max_fwhmN,
        max_mean1, ..., max_meanN]

    Returns
    -------
    model : Best fit model
    fitted_residual_peaks : Updated list of initial mean position guesses for new fit components determined from
        residual peaks.

    """
    amp_guesses, fwhm_guesses, offset_guesses = _get_initial_guesses(
        residual=model.residual,
        rms=model.spectrum.rms_noise,
        snr=settings_improve_fit.snr,
        significance=settings_improve_fit.significance,
        peak="positive",
    )

    if (amp_guesses.size == 0) or (list(offset_guesses) in fitted_residual_peaks):
        model.new_best_fit = False
        return model, fitted_residual_peaks

    amps_fit = model.amps + list(amp_guesses)
    fwhms_fit = model.fwhms + list(fwhm_guesses)
    offsets_fit = model.means + list(offset_guesses)

    fitted_residual_peaks.append(list(offset_guesses))

    new_model = get_best_fit_model(
        model=Model(spectrum=model.spectrum),
        params_fit=amps_fit + fwhms_fit + offsets_fit,
        settings_improve_fit=settings_improve_fit,
        params_min=params_min,
        params_max=params_max,
    )

    chosen_model = choose_better_model_based_on_aicc(old_model=model, new_model=new_model)

    return chosen_model, fitted_residual_peaks


# TODO: Add function 'log_in_case_of_successful_refit' instead to Model
def _log_new_fit(
    new_fit: bool,
    log_gplus: List,
    mode: Literal["positive_residual_peak", "negative_residual_peak", "broad", "blended"] = "residual",
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
    if new_fit:
        log_gplus.append(
            {
                "positive_residual_peak": 1,
                "negative_residual_peak": 2,
                "broad": 3,
                "blended": 4,
            }[mode]
        )
    return log_gplus


def try_to_improve_fitting(model: Model, settings_improve_fit: SettingsImproveFit) -> Tuple[Dict, int, int, List]:
    """Short summary.

    Parameters
    ----------
    model
    settings_improve_fit : Dataclass containing parameter settings for the improved fitting.

    Returns
    -------
    model : Best fit model
    N_neg_res_peak : Number of negative residual features that occur in the best fit of the spectrum.
    N_blended : Number of blended Gaussian components that occur in the best fit of the spectrum.
    log_gplus : Log of all successful refits of the spectrum.

    """

    # Check the quality of the final fit from GaussPy
    if model.n_components > 0:
        model = get_best_fit_model(
            model=Model(spectrum=model.spectrum),
            params_fit=model.parameters,
            settings_improve_fit=settings_improve_fit,
        )

    # Try to improve fit by searching for peaks in the residual
    first_run = True
    fitted_residual_peaks = []
    log_gplus = []
    selected_for_refit = {
        "negative_residual_peak": settings_improve_fit.refit_neg_res_peak,
        "broad": settings_improve_fit.refit_broad,
        "blended": settings_improve_fit.refit_blended,
    }
    refit_function = {
        "negative_residual_peak": check_for_negative_residual,
        "broad": _check_for_broad_feature,
        "blended": _check_for_blended_feature,
    }

    # while (rchi2 > settings_improve_fit.rchi2_limit) or first_run:
    while (model.pvalue < settings_improve_fit.min_pvalue) or first_run:
        n_fitted_residual_peaks_before_check = len(fitted_residual_peaks)
        while True:
            model.new_best_fit = False
            model, fitted_residual_peaks = check_for_peaks_in_residual(
                model, settings_improve_fit, fitted_residual_peaks
            )
            log_gplus = _log_new_fit(new_fit=model.new_best_fit, log_gplus=log_gplus, mode="positive_residual_peak")
            if not model.new_best_fit:
                break
        n_fitted_residual_peaks_after_check = len(fitted_residual_peaks)

        # Stop refitting loop if no new peaks were fit from the residual
        has_no_new_fitted_residual_peaks = n_fitted_residual_peaks_before_check == n_fitted_residual_peaks_after_check
        if (not first_run and has_no_new_fitted_residual_peaks) or (model.n_components == 0):
            break

        for mode in ["negative_residual_peak", "broad", "blended"]:
            while selected_for_refit[mode]:
                model.new_best_fit = False
                model = refit_function[mode](model=model, settings_improve_fit=settings_improve_fit)
                log_gplus = _log_new_fit(new_fit=model.new_best_fit, log_gplus=log_gplus, mode=mode)
                if not model.new_best_fit:
                    break

        if not first_run:
            break
        first_run = False

    N_neg_res_peak = check_for_negative_residual(
        model=model, settings_improve_fit=settings_improve_fit, get_count=True
    )

    N_blended = get_number_of_fully_blended_gaussians(
        params_fit=model.parameters,
        separation_factor=settings_improve_fit.separation_factor,
    )

    return model.best_fit_info, N_neg_res_peak, N_blended, log_gplus


if __name__ == "__main__":
    model = Model()
    model.parameters = [1, 10, 20]
