"""Gaussian functions."""
import sys
from typing import List, Tuple, Callable, Optional, Union

import numpy as np
from lmfit import Parameters


# TODO: rename to integrated_area_under_gaussian_curve
def area_of_gaussian(amp: float, fwhm: float) -> float:
    """Calculate the integrated area of the Gaussian function.

    Parameters
    ----------
    amp : Amplitude value of the Gaussian component.
    fwhm : FWHM value of the Gaussian component.

    """
    return amp * fwhm / ((1. / np.sqrt(2*np.pi)) * 2*np.sqrt(2*np.log(2)))


def gaussian(amp:float , fwhm: float, mean: float, x: np.ndarray) -> np.ndarray:
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
    return amp * np.exp(-4. * np.log(2) * (x-mean)**2 / fwhm**2)


def combined_gaussian(amps: Union[List, np.ndarray],
                      fwhms: Union[List, np.ndarray],
                      means: Union[List, np.ndarray],
                      x: np.ndarray) -> np.ndarray:
    """Return results of the combination of N Gaussian functions.

    Parameters
    ----------
    amps : List of the amplitude values of the Gaussian functions [amp1, ..., ampN].
    fwhms : List of the FWHM values of the Gaussian functions [fwhm1, ..., fwhmN].
    means : List of the mean positions of the Gaussian functions [mean1, ..., meanN].
    x : Array of spectral channels.

    Returns
    -------
    combined_gauss : Combination of N Gaussian functions.

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


def split_params(params: List, ncomps: int) -> Tuple[List, List, List]:
    """Split params into amps, fwhms, offsets."""
    return params[:ncomps], params[ncomps:2*ncomps], params[2*ncomps:3*ncomps]


def number_of_gaussian_components(params: List) -> int:
    """Compute number of Gaussian components."""
    return len(params) // 3


# TODO: check if this function is used anywhere (function was called 'gaussian_function' originally)
def single_component_gaussian_model(peak: float, fwhm: float, mean: float) -> Callable:
    """Return a Gaussian function."""
    sigma = fwhm / 2.354820045  # (2 * sqrt( 2 * ln(2)))
    return lambda x: peak * np.exp(-(x - mean)**2 / 2. / sigma**2)


def multi_component_gaussian_model(x, *args):
    """Return multi-component Gaussian model F(x).

    Parameter vector kargs = [amp1, ..., ampN, width1, ..., widthN, mean1, ..., meanN],
    and therefore has len(args) = 3 x N_components.
    """
    ncomps = number_of_gaussian_components(params=args)
    yout = x * 0.
    for i in range(ncomps):
        yout = yout + single_component_gaussian_model(peak=args[i], fwhm=args[i+ncomps], mean=args[i+2*ncomps])(x)
    return yout


# TODO: Identical function in AGD_decomposer -> remove redundancy
def vals_vec_from_lmfit(lmfit_params):
    """Return Python list of parameter values from LMFIT Parameters object."""
    # if (sys.version_info >= (3, 0)):
    #     vals = [value.value for value in list(lmfit_params.values())]
    # else:
    #     vals = [value.value for value in lmfit_params.values()]
    # return vals
    return [value.value for value in lmfit_params.values()]


# TODO: Identical function in AGD_decomposer -> remove redundancy
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
def paramvec_to_lmfit(paramvec: List,
                      max_amp: Optional[float] = None,
                      max_fwhm: Optional[float] = None,
                      params_min: Optional[List] = None,
                      params_max: Optional[List] = None):
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
        params_min = len(paramvec)*[0.]

    if params_max is None:
        params_max = len(paramvec)*[None]

        if max_amp is not None:
            params_max[:ncomps] = ncomps*[max_amp]
        if max_fwhm is not None:
            params_max[ncomps:2*ncomps] = ncomps*[max_fwhm]

    for i in range(len(paramvec)):
        params.add(name=f'p{i + 1}', value=paramvec[i], min=params_min[i], max=params_max[i])

    return params
