"""Functions that check the quality of the fit."""

import numpy as np
from scipy.stats import normaltest, kstest

from .noise_estimation import determine_peaks


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


def check_mask(mask, n_channels):
    if mask is None:
        mask = np.ones(n_channels)
    elif len(mask) == 0:
        mask = np.ones(n_channels)
    elif np.count_nonzero(mask) == 0:
        mask = np.ones(n_channels)
    return mask.astype('bool')


def get_pvalue_from_normaltest(data, mask=None):
    mask = check_mask(mask, len(data))
    statistic, pvalue = normaltest(data[mask])

    return pvalue


def get_pvalue_from_kstest(data, errors, mask=None):
    if type(errors) is not np.ndarray:
        errors = np.ones(len(data)) * errors
    mask = check_mask(mask, len(data))
    statistic, pvalue = kstest(data[mask] / errors[mask], 'norm')

    return pvalue


def check_residual_for_normality(data, errors, mask=None,
                                 noise_spike_mask=None):
    n_channels = len(data)
    if type(errors) is not np.ndarray:
        errors = np.ones(n_channels) * errors
    mask = check_mask(mask, n_channels)
    noise_spike_mask = check_mask(noise_spike_mask, n_channels)
    ks_statistic, ks_pvalue = kstest(data[mask] / errors[mask], 'norm')
    if n_channels > 20:
        statistic, pvalue = normaltest(data[noise_spike_mask])
        statistic, pvalue_mask = normaltest(data[mask])
        return min(ks_pvalue, pvalue, pvalue_mask)

    return ks_pvalue


def negative_residuals(spectrum, residual, rms, neg_res_snr=3.):
    N_negative_residuals = 0

    amp_vals, ranges = determine_peaks(
        residual, peak='negative', amp_threshold=neg_res_snr*rms)

    if len(amp_vals) > 0:
        amp_vals_position_mask = np.in1d(residual, amp_vals)
        offset_vals = np.where(amp_vals_position_mask == True)[0]

        for offset in offset_vals:
            if residual[offset] < (spectrum[offset] - neg_res_snr*rms):
                N_negative_residuals += 1

    return N_negative_residuals
