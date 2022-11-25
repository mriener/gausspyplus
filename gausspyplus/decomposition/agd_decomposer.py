import time

from typing import Optional, Dict, Literal, Tuple, List, Union

import numpy as np
from scipy.interpolate import interp1d
from lmfit import minimize as lmfit_minimize

from numpy.linalg import lstsq
from scipy.ndimage.filters import median_filter, convolve

from gausspyplus.definitions.definitions import SettingsImproveFit
from gausspyplus.decomposition.gp_plus import try_to_improve_fitting
from gausspyplus.definitions.model import Model
from gausspyplus.plotting.plotting import plot_fit_stages
from gausspyplus.definitions.spectrum import Spectrum
from gausspyplus.decomposition.gaussian_functions import (
    CONVERSION_STD_TO_FWHM,
    errs_vec_from_lmfit,
    paramvec_to_lmfit,
    vals_vec_from_lmfit,
    multi_component_gaussian_model,
    sort_parameters,
)
from gausspyplus.utils.output import say


def _create_fitmask(size: int, offsets_i: np.ndarray, di: np.ndarray) -> np.ndarray:
    """Return valid domain for intermediate fit in d2/dx2 space.

    fitmask = (0,1)
    """
    fitmask = np.zeros(size)
    for offset, d in zip(offsets_i, di):
        fitmask[int(offset - d) : int(offset + d)] = 1
    return fitmask.astype(bool)


def _determine_derivatives(
    data: np.ndarray, dv: Union[int, float], gauss_sigma: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """

    Parameters
    ----------
    data
    dv: x-spacing in absolute units
    gauss_sigma

    Returns
    -------

    """
    gauss_sigma_int = np.max([np.fix(gauss_sigma), 5])
    gauss_dn = gauss_sigma_int * 6

    xx = np.arange(2 * gauss_dn + 2) - gauss_dn - 0.5
    gauss = np.exp(-(xx**2) / 2 / gauss_sigma**2)
    gauss = gauss / np.sum(gauss)
    gauss1 = np.diff(gauss) / dv
    gauss3 = np.diff(gauss1, 2) / dv**2

    xx2 = np.arange(2 * gauss_dn + 1) - gauss_dn
    gauss2 = np.exp(-(xx2**2) / 2 / gauss_sigma**2)
    gauss2 = gauss2 / np.sum(gauss2)
    # TODO: Should the following two lines be `np.diff(gauss2, 2) / dv**2`?
    gauss2 = np.diff(gauss2) / dv
    gauss2 = np.diff(gauss2) / dv
    gauss4 = np.diff(gauss2, 2) / dv**2

    u = convolve(data, gauss1, mode="wrap")
    u2 = convolve(data, gauss2, mode="wrap")
    u3 = convolve(data, gauss3, mode="wrap")
    u4 = convolve(data, gauss4, mode="wrap")
    return u, u2, u3, u4


def _initial_guess(
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
    say("\n\n  --> _initial_guess() \n", verbose=verbose)
    say("Algorithm parameters: ", verbose=verbose)
    say(f"alpha = {alpha}", verbose=verbose)
    say(f"SNR_thesh = {SNR_thresh}", verbose=verbose)
    say(f"SNR2_thesh = {SNR2_thresh}", verbose=verbose)

    if np.any(np.isnan(data)):
        print("NaN-values in data, cannot continue.")
        return

    # Take regularized derivatives
    t0 = time.time()
    say(f"Convolution sigma [pixels]: {alpha}", verbose=verbose)
    u, u2, u3, u4 = _determine_derivatives(
        data, dv=np.abs(vel[1] - vel[0]), gauss_sigma=alpha
    )
    say(
        "...took {0:4.2f} seconds per derivative.".format((time.time() - t0) / 4.0),
        verbose=verbose,
    )

    # Decide on signal threshold
    if not errors:
        errors = np.std(data[data < abs(np.min(data))])  # added by M.Riener

    thresh = SNR_thresh * errors

    if SNR2_thresh > 0:
        wsort = np.argsort(np.abs(u2))
        RMSD2 = (
            np.std(u2[wsort[: int(0.5 * len(u2))]]) / 0.377
        )  # RMS based in +-1 sigma fluctuations
        say(f"Second derivative noise: {RMSD2}", verbose=verbose)
        thresh2 = -RMSD2 * SNR2_thresh
        say(f"Second derivative threshold: {thresh2}", verbose=verbose)
    else:
        thresh2 = 0

    # Find optima of second derivative
    # --------------------------------
    mask = np.all(
        (
            data[1:] > thresh,  # Raw Data S/N
            u4[1:] > 0,  # Positive 4th derivative
            u2[1:] < thresh2,  # # Negative second derivative
        ),
        axis=0,
    )
    indices_of_offsets = np.flatnonzero(
        np.abs(np.diff(np.sign(u3))) * mask
    )  # Index offsets
    fvel = interp1d(np.arange(len(vel)), vel)  # Converts from index -> x domain
    offsets = fvel(indices_of_offsets + 0.5)  # Velocity offsets
    N_components = len(offsets)
    say(f"Components found for alpha={alpha}: {N_components}", verbose=verbose)

    if not N_components:
        amps, offsets, fwhms = [], [], []
    else:
        # Find Relative widths, then measure peak-to-inflection distance for sharpest peak
        fwhms = np.sqrt(np.abs(data / u2)[indices_of_offsets]) * CONVERSION_STD_TO_FWHM

        amps = data[indices_of_offsets]

        # Attempt deblending. If Deblending results in all non-negative answers, keep.
        FF_matrix = np.zeros((N_components, N_components))
        for i in range(FF_matrix.shape[0]):
            for j in range(FF_matrix.shape[1]):
                FF_matrix[i, j] = np.exp(
                    -((offsets[i] - offsets[j]) ** 2)
                    / 2
                    / (fwhms[j] / CONVERSION_STD_TO_FWHM) ** 2
                )
        amps_new = lstsq(FF_matrix, amps, rcond=None)[0]
        if np.all(amps_new > 0):
            amps = amps_new

    return {
        "means": offsets,
        "fwhms": fwhms,
        "amps": amps,
        "u2": u2,
        "thresh2": thresh2,
        "thresh": thresh,
        "N_components": N_components,
    }


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

    # TODO: put this in private function -> combine it with phase 2 guesses -> maybe rename _initial_guess?
    # -------------------------------------- #
    # Find phase-one guesses                 #
    # -------------------------------------- #
    agd_phase1 = _initial_guess(
        vel,
        data,
        errors=errors[0],
        alpha=alpha1,
        verbose=verbose,
        SNR_thresh=SNR_thresh,
        SNR2_thresh=SNR2_thresh,
    )

    params_guess_phase1 = sort_parameters(
        amps=agd_phase1["amps"], fwhms=agd_phase1["fwhms"], means=agd_phase1["means"]
    )
    ncomps_guess_phase2 = 0  # Default
    params_fit_phase1 = []  # Default
    residuals = data  # Default

    # ----------------------------#
    # Find phase-two guesses #
    # ----------------------------#
    if phase == "two":
        say("Beginning phase-two AGD... ", verbose=verbose)

        # ----------------------------------------------------------#
        # Produce the residual signal                               #
        #  -- Either the original data, or intermediate subtraction #
        # ----------------------------------------------------------#
        if agd_phase1["N_components"] == 0:
            say(
                "Phase 2 with no narrow comps -> No intermediate subtraction... ",
                verbose=verbose,
            )
        else:
            # "Else" Narrow components were found, and Phase == 2, so perform intermediate subtraction...

            v_to_i = interp1d(vel, np.arange(len(vel)))
            # `fitmask` is a collection of windows around the list of phase-one components
            fitmask = _create_fitmask(
                size=len(vel),
                offsets_i=v_to_i(agd_phase1["means"]),
                di=agd_phase1["fwhms"] / dv / CONVERSION_STD_TO_FWHM * 0.9,
            )

            # Error function for intermediate optimization
            def objectiveD2_leastsq(paramslm):
                params = vals_vec_from_lmfit(paramslm)
                model0 = multi_component_gaussian_model(
                    *np.split(np.array(params), 3), vel
                )
                model2 = np.diff(np.diff(model0.ravel())) / dv / dv
                resids1 = (
                    fitmask[1:-1] * (model2 - agd_phase1["u2"][1:-1]) / errors[1:-1]
                )
                resids2 = ~fitmask * (model0 - data) / errors / 10.0
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

            del lmfit_params
            say(f"LMFIT fit took {time.time() - t0} seconds.")

            if result.success:
                # Compute intermediate residuals
                # Median filter on 2x effective scale to remove poor subtractions of strong components
                # Explicit final (narrow) model
                intermediate_model = multi_component_gaussian_model(
                    *np.split(np.array(params_fit_phase1), 3), vel
                ).ravel()
                median_window = 2 * 10 ** ((np.log10(alpha1) + 2.187) / 3.859)
                residuals = median_filter(
                    data - intermediate_model, np.int64(median_window)
                )
            # Finished producing residual signal # ---------------------------

        # Search for phase-two guesses
        agd_phase2 = _initial_guess(
            vel,
            residuals,
            errors=errors[0],
            alpha=alpha2,
            verbose=verbose,
            SNR_thresh=SNR_thresh,
            SNR2_thresh=SNR2_thresh,
        )
        ncomps_guess_phase2 = agd_phase2["N_components"]

        # END PHASE 2 <<<

    # Check for phase two components, make final guess list
    # ------------------------------------------------------
    if phase == "two" and (ncomps_guess_phase2 > 0):
        params_guess_final = sort_parameters(
            amps=np.append(agd_phase1["amps"], agd_phase2["amps"]),
            fwhms=np.append(agd_phase1["fwhms"], agd_phase2["fwhms"]),
            means=np.append(agd_phase1["means"], agd_phase2["means"]),
        )
        ncomps_guess_final = agd_phase1["N_components"] + agd_phase2["N_components"]
    else:
        params_guess_final = params_guess_phase1
        ncomps_guess_final = agd_phase1["N_components"]

    say(f"N final parameter guesses: {ncomps_guess_final}", verbose=verbose)

    if perform_final_fit and ncomps_guess_final > 0:
        say("\n\n  --> Final Fitting... \n", verbose=verbose)

        # Objective functions for final fit
        def objective_leastsq(paramslm):
            params = vals_vec_from_lmfit(paramslm)
            resids = (
                multi_component_gaussian_model(
                    *np.split(np.array(params), 3), vel
                ).ravel()
                - data.ravel()
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
    else:
        params_fit = []

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
        model.parameters = params_fit
        best_fit_info, N_neg_res_peak, N_blended, log_gplus = try_to_improve_fitting(
            model=model, settings_improve_fit=settings_improve_fit
        )

        params_fit = best_fit_info["params_fit"]
        params_errs = best_fit_info["params_errs"]
        ncomps_guess_final = best_fit_info["ncomps_fit"]
    else:
        best_fit_info = None

    # TODO: Simplify the parameters for this function
    if plot:
        plot_fit_stages(
            data,
            errors,
            vel,
            params_fit,
            ncomps_guess_final,
            improve_fitting,
            perform_final_fit,
            ncomps_guess_phase2,
            agd_phase1,
            phase,
            residuals,
            agd_phase2,
            best_fit_info=best_fit_info,
            params_fit_phase1=params_fit_phase1,
        )

    # Construct output dictionary (odict)
    # -----------------------------------
    odict = {
        "initial_parameters": params_guess_final,
        "N_components": ncomps_guess_final,
        "index": idx,
    }
    if perform_final_fit and (ncomps_guess_final > 0):
        odict = {
            **odict,
            **{"best_fit_parameters": params_fit, "best_fit_errors": params_errs},
        }

    if improve_fitting:
        odict = {
            **odict,
            **{
                "best_fit_rchi2": best_fit_info["rchi2"],
                "best_fit_aicc": best_fit_info["aicc"],
                "pvalue": best_fit_info["pvalue"],
                "N_neg_res_peak": N_neg_res_peak,
                "N_blended": N_blended,
                "log_gplus": log_gplus,
                "quality_control": best_fit_info["quality_control"],
            },
        }

    return odict
