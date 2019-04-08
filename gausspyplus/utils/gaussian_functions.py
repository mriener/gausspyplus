"""Gaussian functions."""

import numpy as np


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
