"""Functions that check the quality of the fit."""

import numpy as np
import warnings
from scipy.stats import normaltest, kstest

from gausspyplus.preparation.noise_estimation import determine_peaks
from gausspyplus.utils.output import format_warning

warnings.showwarning = format_warning


def determine_significance(amp, fwhm, rms):
    """Calculate the significance value of a fitted Gaussian component or a feature in the spectrum.

    The area of the Gaussian is:
    area_gauss = amp * fwhm / ((1. / np.sqrt(2*np.pi)) * 2*np.sqrt(2*np.log(2)))

    This is then compared to the integrated rms, with 2*fwhm being a good
    approximation for the width of the emission line

    significance = area_gauss / (np.sqrt(2*fwhm) * rms)

    area_gauss = amp * fwhm / ( 2*np.sqrt(np.log(2) / np.pi) )

    significance = amp * np.sqrt(fwhm) / (2 * np.sqrt(2 * np.log(2) / np.pi) * rms )


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
    return amp * np.sqrt(fwhm) / (np.sqrt(8 * np.log(2) / np.pi) * rms)


def goodness_of_fit(data, best_fit_final, errors, ncomps_fit, mask=None, get_aicc=False):
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
    mask = _define_mask_if_not_given(mask, len(data))

    squared_residuals = (data[mask] - best_fit_final[mask]) ** 2
    chi2 = np.sum(squared_residuals / errors[mask] ** 2)
    n_params = 3 * ncomps_fit  # degrees of freedom
    n_samples = len(data[mask])
    rchi2 = chi2 / (n_samples - n_params)
    if get_aicc:
        #  sum of squared residuals
        ssr = np.sum(squared_residuals)
        log_likelihood = -0.5 * n_samples * np.log(ssr / n_samples)
        aicc = 2.0 * (n_params - log_likelihood) + 2.0 * n_params * (n_params + 1.0) / (n_samples - n_params - 1.0)
        return rchi2, aicc
    return rchi2


def _define_mask_if_not_given(mask, n_channels) -> np.ndarray:
    if mask is None or len(mask) == 0 or np.count_nonzero(mask) == 0:
        return np.full(n_channels, True)
    else:
        return mask.astype("bool")


def get_pvalue_from_normaltest(data, mask=None):
    mask = _define_mask_if_not_given(mask, len(data))
    statistic, pvalue = normaltest(data[mask])

    return pvalue


def get_pvalue_from_kstest(data, errors, mask=None):
    if type(errors) is not np.ndarray:
        errors = np.ones(len(data)) * errors
    mask = _define_mask_if_not_given(mask, len(data))
    statistic, pvalue = kstest(data[mask] / errors[mask], "norm")

    return pvalue


def check_residual_for_normality(data, errors, mask=None, noise_spike_mask=None):
    n_channels = len(data)
    if type(errors) is not np.ndarray:
        errors = np.ones(n_channels) * errors
    mask = _define_mask_if_not_given(mask, n_channels)
    noise_spike_mask = _define_mask_if_not_given(noise_spike_mask, n_channels)
    try:
        ks_statistic, ks_pvalue = kstest(data[mask] / errors[mask], "norm")
    except ValueError:
        warnings.warn("Normality test for residual unsuccessful. Setting pvalue to 0.")
        ks_pvalue = 0

    if n_channels > 20:
        try:
            statistic, pvalue = normaltest(data[noise_spike_mask])
        except ValueError:
            warnings.warn("Normality test for residual unsuccessful. Setting pvalue to 0.")
            pvalue = 0

        try:
            statistic, pvalue_mask = normaltest(data[mask])
        except ValueError:
            warnings.warn("Normality test for residual unsuccessful. Setting pvalue to 0.")
            pvalue_mask = 0

        return min(ks_pvalue, pvalue, pvalue_mask)

    return ks_pvalue


def negative_residuals(spectrum, residual, rms, neg_res_snr=3.0, get_flags=False, fwhms=None, means=None):
    N_negative_residuals = 0

    if get_flags:
        flags = np.zeros(len(fwhms)).astype("bool")

    amp_vals, ranges = determine_peaks(residual, peak="negative", amp_threshold=neg_res_snr * rms)

    if len(amp_vals) > 0:
        amp_vals_position_mask = np.in1d(residual, amp_vals)
        offset_vals = np.where(amp_vals_position_mask == True)[0]

        for offset in offset_vals:
            if residual[offset] < (spectrum[offset] - neg_res_snr * rms):
                N_negative_residuals += 1

                if get_flags:
                    lower = np.array(means) - np.array(fwhms) / 2
                    upper = np.array(means) + np.array(fwhms) / 2

                    flags += np.logical_and(lower < offset, upper > offset)

    return flags.astype("int") if get_flags else N_negative_residuals
