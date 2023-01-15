"""Gaussian functions."""
from typing import List, Tuple, Optional, Union

import numpy as np
from lmfit import Parameters

from gausspyplus.preparation.determine_intervals import get_slice_indices_for_interval

CONVERSION_STD_TO_FWHM = 2 * np.sqrt(2 * np.log(2))


# TODO: rename to integrated_area_under_gaussian_curve
def area_of_gaussian(amp: float, fwhm: float) -> float:
    """Calculate the integrated area of the Gaussian function.

    Parameters
    ----------
    amp : Amplitude value of the Gaussian component.
    fwhm : FWHM value of the Gaussian component.

    """
    return amp * fwhm / ((1.0 / np.sqrt(2 * np.pi)) * CONVERSION_STD_TO_FWHM)


def single_component_gaussian_model(amp: float, fwhm: float, mean: float, x: np.ndarray) -> np.ndarray:
    """Return results of a Gaussian function.

    Parameters
    ----------
    amp : Amplitude of the Gaussian function.
    fwhm : FWHM of the Gaussian function.
    mean : Mean position of the Gaussian function.
    x : Array of spectral channels.

    Returns
    -------
    Gaussian function.

    """
    return amp * np.exp(-4.0 * np.log(2) * (x - mean) ** 2 / fwhm**2)


def multi_component_gaussian_model(
    amps: Union[List, np.ndarray],
    fwhms: Union[List, np.ndarray],
    means: Union[List, np.ndarray],
    x: np.ndarray,
) -> np.ndarray:
    """Return results of the combination of N Gaussian functions.

    Parameters
    ----------
    amps : List of the amplitude values of the Gaussian functions [amp1, ..., ampN].
    fwhms : List of the FWHM values of the Gaussian functions [fwhm1, ..., fwhmN].
    means : List of the mean positions of the Gaussian functions [mean1, ..., meanN].
    x : Array of spectral channels.

    Returns
    -------
    modelled_spectrum : Combination of N Gaussian functions.

    """
    modelled_spectrum = np.zeros(x.size)
    for amp, fwhm, mean in zip(amps, fwhms, means):
        modelled_spectrum += single_component_gaussian_model(amp, fwhm, mean, x)
    return modelled_spectrum


def split_params(params: List, ncomps: int) -> Tuple[List, List, List]:
    """Split params into amps, fwhms, offsets."""
    return params[:ncomps], params[ncomps : 2 * ncomps], params[2 * ncomps : 3 * ncomps]


def number_of_gaussian_components(params: List) -> int:
    """Compute number of Gaussian components."""
    return len(params) // 3


def vals_vec_from_lmfit(lmfit_params):
    """Return Python list of parameter values from LMFIT Parameters object."""
    # if (sys.version_info >= (3, 0)):
    #     vals = [value.value for value in list(lmfit_params.values())]
    # else:
    #     vals = [value.value for value in lmfit_params.values()]
    # return vals
    return [value.value for value in lmfit_params.values()]


def errs_vec_from_lmfit(lmfit_params):
    """Return Python list of parameter uncertainties from LMFIT Parameters object."""
    # if (sys.version_info >= (3, 0)):
    #     errs = [value.stderr for value in list(lmfit_params.values())]
    # else:
    #     errs = [value.stderr for value in lmfit_params.values()]
    # # TODO: estimate errors via bootstrapping instead of setting them to zero
    # errs = [0 if err is None else err for err in errs]
    # return errs
    return [0 if value.stderr is None else value.stderr for value in lmfit_params.values()]


# TODO: Identical function in AGD_decomposer -> remove redundancy
def paramvec_to_lmfit(
    paramvec: List,
    max_amp: Optional[float] = None,
    max_fwhm: Optional[float] = None,
    params_min: Optional[List] = None,
    params_max: Optional[List] = None,
):
    """Transform a Python iterable of parameters into a LMFIT Parameters object.

    Parameters
    ----------
    paramvec : Parameter vector = [amp1, ..., ampN, fwhm1, ..., fwhmN, mean1, ..., meanN].
    max_amp : Enforced maximum value for amplitude parameter.
    max_fwhm : Enforced maximum value for FWHM parameter. Use with caution! Can lead to artifacts in the fitting.
    params_min : List of minimum limits for parameters: [min_amp1, ..., min_ampN, min_fwhm1, ..., min_fwhmN,
        min_mean1, ..., min_meanN]
    params_max : List of maximum limits for parameters: [max_amp1, ..., max_ampN, max_fwhm1, ..., max_fwhmN,
        max_mean1, ..., max_meanN]

    Returns
    -------
    params: lmfit.parameter.Parameters

    """
    ncomps = number_of_gaussian_components(params=paramvec)
    params = Parameters()

    if params_min is None:
        params_min = len(paramvec) * [0.0]

    if params_max is None:
        params_max = len(paramvec) * [None]

        if max_amp is not None:
            params_max[:ncomps] = ncomps * [max_amp]
        if max_fwhm is not None:
            params_max[ncomps : 2 * ncomps] = ncomps * [max_fwhm]

    for i in range(len(paramvec)):
        params.add(name=f"p{i + 1}", value=paramvec[i], min=params_min[i], max=params_max[i])

    return params


def sort_parameters(amps, fwhms, means, descending: bool = True):
    """Parameters are sorted according to the amplitude values."""
    sort_oder = np.argsort(amps)[::-1] if descending else np.argsort(amps)
    return np.concatenate(
        [
            np.array(amps)[sort_oder],
            np.array(fwhms)[sort_oder],
            np.array(means)[sort_oder],
        ]
    )


def upper_limit_for_amplitude(
    intensity_values: np.ndarray,
    mean: float,
    fwhm: float,
    buffer_factor: float = 1.0,
) -> float:
    idx_low, idx_upp = get_slice_indices_for_interval(
        interval_center=mean,
        # TODO: is this correct or should interval_half_width be fwhm / 2?
        interval_half_width=fwhm / CONVERSION_STD_TO_FWHM,
    )
    return buffer_factor * np.max(intensity_values[idx_low:idx_upp])
