# @Author: Robert Lindner
# @Date:   Nov 10, 2014
# @Filename: AGD_decomposer.py
# @Last modified by:   riener
# @Last modified time: 18-05-2020

# Standard Libs
import time

# Standard Third Party
from typing import Optional, Dict, Literal, Tuple, List, Union

import numpy as np
from scipy.interpolate import interp1d
from lmfit import minimize as lmfit_minimize

import matplotlib.pyplot as plt
from numpy.linalg import lstsq
from scipy.ndimage.filters import median_filter, convolve

from gausspyplus.definitions import SettingsImproveFit
from gausspyplus.gausspy_py3.gp_plus import try_to_improve_fitting
from gausspyplus.model import Model
from gausspyplus.plotting import plot_fit_stages
from gausspyplus.spectrum import Spectrum
from gausspyplus.utils.fit_quality_checks import goodness_of_fit
from gausspyplus.utils.gaussian_functions import (
    CONVERSION_STD_TO_FWHM,
    errs_vec_from_lmfit,
    paramvec_to_lmfit,
    multi_component_gaussian_model,
    single_component_gaussian_model,
    vals_vec_from_lmfit,
)
from gausspyplus.utils.output import say


def _create_fitmask(size: int, offsets_i: np.ndarray, di: np.ndarray) -> np.ndarray:
    """Return valid domain for intermediate fit in d2/dx2 space.

    fitmask = (0,1)
    """
    fitmask = np.zeros(size)
    for i in range(len(offsets_i)):
        fitmask[int(offsets_i[i] - di[i]) : int(offsets_i[i] + di[i])] = 1.0
    return fitmask


def _determine_derivatives(
    data: np.ndarray, dv: Union[int, float], alpha: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    gauss_sigma = alpha
    gauss_sigma_int = np.max([np.fix(gauss_sigma), 5])
    gauss_dn = gauss_sigma_int * 6

    xx = np.arange(2 * gauss_dn + 2) - (gauss_dn) - 0.5
    gauss = np.exp(-(xx**2) / 2.0 / gauss_sigma**2)
    gauss = gauss / np.sum(gauss)
    gauss1 = np.diff(gauss) / dv
    gauss3 = np.diff(np.diff(gauss1)) / dv**2

    xx2 = np.arange(2 * gauss_dn + 1) - (gauss_dn)
    gauss2 = np.exp(-(xx2**2) / 2.0 / gauss_sigma**2)
    gauss2 = gauss2 / np.sum(gauss2)
    gauss2 = np.diff(gauss2) / dv
    gauss2 = np.diff(gauss2) / dv
    gauss4 = np.diff(np.diff(gauss2)) / dv**2

    u = convolve(data, gauss1, mode="wrap")
    u2 = convolve(data, gauss2, mode="wrap")
    u3 = convolve(data, gauss3, mode="wrap")
    u4 = convolve(data, gauss4, mode="wrap")
    return u, u2, u3, u4


def initialGuess(
    vel: np.ndarray,
    data: np.ndarray,
    errors: Optional[float] = None,
    alpha: Optional[float] = None,
    verbose: bool = False,
    SNR_thresh: float = 5.0,
    SNR2_thresh: float = 5.0,
) -> Optional[Dict]:
    """Find initial parameter guesses (AGD algorithm).

    data,             Input data
    alpha = No Default,     regularization parameter
    verbose = True    Diagnostic messages
    SNR_thresh = 5.0  Initial Spectrum S/N threshold
    SNR2_thresh =   S/N threshold for Second derivative
    """
    say("\n\n  --> initialGuess() \n", verbose=verbose)
    say("Algorithm parameters: ", verbose=verbose)
    say(f"alpha = {alpha}", verbose=verbose)
    say(f"SNR_thesh = {SNR_thresh}", verbose=verbose)
    say(f"SNR2_thesh = {SNR2_thresh}", verbose=verbose)

    if np.any(np.isnan(data)):
        print("NaN-values in data, cannot continue.")
        return

    # Data inspection
    vel = np.array(vel)
    data = np.array(data)
    dv = np.abs(vel[1] - vel[0])  # x-spacing in absolute units
    fvel = interp1d(np.arange(len(vel)), vel)  # Converts from index -> x domain

    # Take regularized derivatives
    t0 = time.time()
    say(f"Convolution sigma [pixels]: {alpha}", verbose=verbose)
    u, u2, u3, u4 = _determine_derivatives(data, dv, alpha)
    say(
        "...took {0:4.2f} seconds per derivative.".format((time.time() - t0) / 4.0),
        verbose=verbose,
    )

    # Decide on signal threshold
    if not errors:
        errors = np.std(data[data < abs(np.min(data))])  # added by M.Riener

    thresh = SNR_thresh * errors
    mask1 = np.array(data > thresh, dtype="int")[1:]  # Raw Data S/N
    mask3 = np.array(u4.copy()[1:] > 0.0, dtype="int")  # Positive 4th derivative

    if SNR2_thresh > 0.0:
        wsort = np.argsort(np.abs(u2))
        RMSD2 = (
            np.std(u2[wsort[: int(0.5 * len(u2))]]) / 0.377
        )  # RMS based in +-1 sigma fluctuations
        say(f"Second derivative noise: {RMSD2}", verbose=verbose)
        thresh2 = -RMSD2 * SNR2_thresh
        say(f"Second derivative threshold: {thresh2}", verbose=verbose)
    else:
        thresh2 = 0.0
    mask4 = np.array(u2.copy()[1:] < thresh2, dtype="int")  # Negative second derivative

    # Find optima of second derivative
    # --------------------------------
    zeros = np.abs(np.diff(np.sign(u3)))
    zeros = zeros * mask1 * mask3 * mask4
    indices_of_offsets = np.array(np.where(zeros)).ravel()  # Index offsets
    offsets = fvel(indices_of_offsets + 0.5)  # Velocity offsets
    N_components = len(offsets)
    say(f"Components found for alpha={alpha}: {N_components}", verbose=verbose)

    # Check if nothing was found, if so, return null
    # ----------------------------------------------
    if N_components == 0:
        odict = {
            "means": [],
            "FWHMs": [],
            "amps": [],
            "u2": u2,
            "errors": errors,
            "thresh2": thresh2,
            "thresh": thresh,
            "N_components": N_components,
        }

        return odict

    # Find Relative widths, then measure peak-to-inflection distance for sharpest peak
    FWHMs = np.sqrt(np.abs(data / u2)[indices_of_offsets]) * CONVERSION_STD_TO_FWHM

    amps = np.array(data[indices_of_offsets])

    # Attempt deblending. If Deblending results in all non-negative answers, keep.
    FF_matrix = np.zeros([len(amps), len(amps)])
    for i in range(FF_matrix.shape[0]):
        for j in range(FF_matrix.shape[1]):
            FF_matrix[i, j] = np.exp(
                -((offsets[i] - offsets[j]) ** 2)
                / 2.0
                / (FWHMs[j] / CONVERSION_STD_TO_FWHM) ** 2
            )
    amps_new = lstsq(FF_matrix, amps, rcond=None)[0]
    if np.all(amps_new > 0):
        amps = amps_new

    odict = {
        "means": offsets,
        "FWHMs": FWHMs,
        "amps": amps,
        "u2": u2,
        "errors": errors,
        "thresh2": thresh2,
        "thresh": thresh,
        "N_components": N_components,
    }

    return odict


def AGD(
    vel: np.ndarray,
    data: np.ndarray,
    errors: np.ndarray,
    idx: Optional[int] = None,
    signal_ranges: Optional[List] = None,
    noise_spike_ranges: Optional[List] = None,
    settings_improve_fit: Optional[SettingsImproveFit] = None,
    alpha1: Optional[float] = None,
    alpha2: Optional[float] = None,
    plot: bool = False,
    verbose: bool = False,
    SNR_thresh: float = 5.0,
    SNR2_thresh: float = 5.0,
    perform_final_fit: bool = True,
    phase: Literal["one", "two"] = "one",
) -> Dict:
    """Autonomous Gaussian Decomposition."""
    max_amp = (
        settings_improve_fit.max_amp_factor * np.max(data)
        if settings_improve_fit is not None
        else None
    )
    if settings_improve_fit is not None:
        settings_improve_fit.max_amp = max_amp
    improve_fitting = (
        settings_improve_fit.improve_fitting
        if settings_improve_fit is not None
        else False
    )

    say("\n  --> AGD() \n", verbose=verbose)

    dv = np.abs(vel[1] - vel[0])
    v_to_i = interp1d(vel, np.arange(len(vel)))

    # TODO: put this in private function -> combine it with phase 2 guesses -> maybe rename initialGuess?
    # -------------------------------------- #
    # Find phase-one guesses                 #
    # -------------------------------------- #
    agd_phase1 = initialGuess(
        vel,
        data,
        errors=errors[0],
        alpha=alpha1,
        verbose=verbose,
        SNR_thresh=SNR_thresh,
        SNR2_thresh=SNR2_thresh,
    )

    amps_guess_phase1, widths_guess_phase1, offsets_guess_phase1, u2 = (
        agd_phase1["amps"],
        agd_phase1["FWHMs"],
        agd_phase1["means"],
        agd_phase1["u2"],
    )
    params_guess_phase1 = np.append(
        np.append(amps_guess_phase1, widths_guess_phase1), offsets_guess_phase1
    )
    ncomps_guess_phase1 = int(len(params_guess_phase1) / 3)
    ncomps_guess_phase2 = 0  # Default
    ncomps_fit_phase1 = 0  # Default

    # ----------------------------#
    # Find phase-two guesses #
    # ----------------------------#
    if phase == "two":
        say("Beginning phase-two AGD... ", verbose=verbose)
        ncomps_guess_phase2 = 0

        # ----------------------------------------------------------#
        # Produce the residual signal                               #
        #  -- Either the original data, or intermediate subtraction #
        # ----------------------------------------------------------#
        if ncomps_guess_phase1 == 0:
            say(
                "Phase 2 with no narrow comps -> No intermediate subtration... ",
                verbose=verbose,
            )
            residuals = data
        else:
            # "Else" Narrow components were found, and Phase == 2, so perform intermediate subtraction...

            # "fitmask" is a collection of windows around the a list of phase-one components
            fitmask = _create_fitmask(
                len(vel),
                v_to_i(offsets_guess_phase1),
                widths_guess_phase1 / dv / CONVERSION_STD_TO_FWHM * 0.9,
            )
            notfitmask = 1 - fitmask

            # Error function for intermediate optimization
            def objectiveD2_leastsq(paramslm):
                params = vals_vec_from_lmfit(paramslm)
                model0 = multi_component_gaussian_model(vel, *params)
                model2 = np.diff(np.diff(model0.ravel())) / dv / dv
                resids1 = fitmask[1:-1] * (model2 - u2[1:-1]) / errors[1:-1]
                resids2 = notfitmask * (model0 - data) / errors / 10.0
                return np.append(resids1, resids2)

            # Perform the intermediate fit using LMFIT
            t0 = time.time()
            say("Running LMFIT on initial narrow components...", verbose=verbose)
            lmfit_params = paramvec_to_lmfit(
                paramvec=params_guess_phase1,
                max_amp=max_amp,
                max_fwhm=settings_improve_fit.max_fwhm
                if settings_improve_fit is not None
                else None,
            )
            result = lmfit_minimize(objectiveD2_leastsq, lmfit_params, method="leastsq")
            params_fit_phase1 = vals_vec_from_lmfit(result.params)
            ncomps_fit_phase1 = int(len(params_fit_phase1) / 3)

            del lmfit_params
            say(f"LMFIT fit took {time.time() - t0} seconds.")

            if result.success:
                # Compute intermediate residuals
                # Median filter on 2x effective scale to remove poor subtractions of strong components
                intermediate_model = multi_component_gaussian_model(
                    vel, *params_fit_phase1
                ).ravel()  # Explicit final (narrow) model
                median_window = 2.0 * 10 ** ((np.log10(alpha1) + 2.187) / 3.859)
                residuals = median_filter(
                    data - intermediate_model, np.int64(median_window)
                )
            else:
                residuals = data
            # Finished producing residual signal # ---------------------------

        # Search for phase-two guesses
        agd_phase2 = initialGuess(
            vel,
            residuals,
            errors=errors[0],
            alpha=alpha2,
            verbose=verbose,
            SNR_thresh=SNR_thresh,
            SNR2_thresh=SNR2_thresh,
        )
        ncomps_guess_phase2 = agd_phase2["N_components"]
        if ncomps_guess_phase2 > 0:
            params_guess_phase2 = np.concatenate(
                [agd_phase2["amps"], agd_phase2["FWHMs"], agd_phase2["means"]]
            )
        else:
            params_guess_phase2 = []
        u2_phase2 = agd_phase2["u2"]

        # END PHASE 2 <<<

    # Check for phase two components, make final guess list
    # ------------------------------------------------------
    if phase == "two" and (ncomps_guess_phase2 > 0):
        amps_guess_final = np.append(
            params_guess_phase1[:ncomps_guess_phase1],
            params_guess_phase2[:ncomps_guess_phase2],
        )
        widths_guess_final = np.append(
            params_guess_phase1[ncomps_guess_phase1 : 2 * ncomps_guess_phase1],
            params_guess_phase2[ncomps_guess_phase2 : 2 * ncomps_guess_phase2],
        )
        offsets_guess_final = np.append(
            params_guess_phase1[2 * ncomps_guess_phase1 : 3 * ncomps_guess_phase1],
            params_guess_phase2[2 * ncomps_guess_phase2 : 3 * ncomps_guess_phase2],
        )
        params_guess_final = np.concatenate(
            [amps_guess_final, widths_guess_final, offsets_guess_final]
        )
        ncomps_guess_final = int(len(params_guess_final) / 3)
    else:
        params_guess_final = params_guess_phase1
        ncomps_guess_final = int(len(params_guess_final) / 3)

    # Sort final guess list by amplitude
    # ----------------------------------
    say("N final parameter guesses: " + str(ncomps_guess_final))
    amps_temp = params_guess_final[:ncomps_guess_final]
    widths_temp = params_guess_final[ncomps_guess_final : 2 * ncomps_guess_final]
    offsets_temp = params_guess_final[2 * ncomps_guess_final : 3 * ncomps_guess_final]
    w_sort_amp = np.argsort(amps_temp)[::-1]
    params_guess_final = np.concatenate(
        [amps_temp[w_sort_amp], widths_temp[w_sort_amp], offsets_temp[w_sort_amp]]
    )

    if (perform_final_fit is True) and (ncomps_guess_final > 0):
        say("\n\n  --> Final Fitting... \n", verbose=verbose)

        # Objective functions for final fit
        def objective_leastsq(paramslm):
            params = vals_vec_from_lmfit(paramslm)
            resids = (
                multi_component_gaussian_model(vel, *params).ravel() - data.ravel()
            ) / errors
            return resids

        # Final fit using unconstrained parameters
        t0 = time.time()
        lmfit_params = paramvec_to_lmfit(params_guess_final, max_amp, None)
        result2 = lmfit_minimize(objective_leastsq, lmfit_params, method="leastsq")
        params_fit = vals_vec_from_lmfit(result2.params)
        params_errs = errs_vec_from_lmfit(result2.params)

        del lmfit_params
        say(f"Final fit took {time.time() - t0} seconds.", verbose=verbose)

    # Try to improve the fit
    # ----------------------
    if improve_fitting:
        model = Model(
            spectrum=Spectrum(
                intensity_values=data,
                channels=vel,
                rms_noise=errors[0],
                signal_intervals=signal_ranges,
                noise_spike_intervals=noise_spike_ranges,
            )
        )
        model.parameters = [] if ncomps_guess_final == 0 else params_fit
        best_fit_info, N_neg_res_peak, N_blended, log_gplus = try_to_improve_fitting(
            model=model, settings_improve_fit=settings_improve_fit
        )

        params_fit = best_fit_info["params_fit"]
        params_errs = best_fit_info["params_errs"]
        ncomps_guess_final = best_fit_info["ncomps_fit"]
        rchi2 = best_fit_info["rchi2"]
        aicc = best_fit_info["aicc"]
        pvalue = best_fit_info["pvalue"]
        quality_control = best_fit_info["quality_control"]

    # TODO: Simplify the parameters for this function
    if plot:
        plot_fit_stages(
            data,
            errors,
            vel,
            params_fit,
            ncomps_guess_final,
            improve_fitting,
            best_fit_info,
            perform_final_fit,
            ncomps_guess_phase1,
            ncomps_guess_phase2,
            u2,
            agd_phase1,
            params_guess_phase1,
            params_fit_phase1,
            ncomps_fit_phase1,
            phase,
            u2_phase2,
            residuals,
            agd_phase2,
            params_guess_phase2,
        )

    # Construct output dictionary (odict)
    # -----------------------------------
    odict = {
        "initial_parameters": params_guess_final,
        "N_components": ncomps_guess_final,
        "index": idx,
    }
    if (perform_final_fit is True) and (ncomps_guess_final > 0):
        odict = {
            **odict,
            **{"best_fit_parameters": params_fit, "best_fit_errors": params_errs},
        }

    if improve_fitting:
        odict = {
            **odict,
            **{
                "best_fit_rchi2": rchi2,
                "best_fit_aicc": aicc,
                "pvalue": pvalue,
                "N_neg_res_peak": N_neg_res_peak,
                "N_blended": N_blended,
                "log_gplus": log_gplus,
                "quality_control": quality_control,
            },
        }

    return odict
