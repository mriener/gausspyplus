import pickle
from typing import Optional, Dict, List

import numpy as np

from gausspyplus.training import gradient_descent
from gausspyplus.decomposition.gaussian_functions import split_params
from gausspyplus.decomposition import agd_decomposer


class GaussianDecomposer(object):
    def __init__(self, filename=None):
        if filename:
            temp = pickle.load(open(filename, "rb"), encoding="latin1")
            self.p = temp.p
        else:
            self.p = {
                "alpha1": None,
                "alpha2": None,
                "training_results": None,
                "settings_improve_fit": None,
                "use_ncpus": None,
                "phase": "one",
                "SNR2_thresh": 5.0,
                "SNR_thresh": 5.0,
                "verbose": False,
                "plot": False,
                "perform_final_fit": True,
            }

    def load_training_data(self, filename):
        self.p["training_data"] = pickle.load(open(filename, "rb"), encoding="latin1")

    def train(
        self,
        alpha1_initial: Optional[float] = None,
        alpha2_initial: Optional[float] = None,
        plot: bool = False,
        verbose: bool = False,
        learning_rate: float = 0.9,
        eps: float = 0.25,
        MAD: float = 0.1,
        logger: bool = False,
    ) -> None:
        """Solve for optimal values of alpha1 (and alpha2) using training data."""
        print("Training...")

        (
            self.p["alpha1"],
            self.p["alpha2"],
            self.p["training_results"],
        ) = gradient_descent.train(
            alpha1_initial=alpha1_initial,
            alpha2_initial=alpha2_initial,
            training_data=self.p["training_data"],
            phase=self.p["phase"],
            SNR_thresh=self.p["SNR_thresh"],
            SNR2_thresh=self.p["SNR2_thresh"],
            plot=plot,
            eps=eps,
            verbose=verbose,
            learning_rate=learning_rate,
            MAD=MAD,
            logger=logger,
        )

    def decompose(
        self,
        xdata: np.ndarray,
        ydata: np.ndarray,
        edata: np.ndarray,
        idx: Optional[int] = None,
        signal_ranges: Optional[List] = None,
        noise_spike_ranges: Optional[List] = None,
    ) -> Dict:
        """Decompose a single spectrum using current parameters."""
        results = agd_decomposer.AGD(
            xdata,
            ydata,
            edata,
            idx=idx,
            signal_ranges=signal_ranges,
            noise_spike_ranges=noise_spike_ranges,
            alpha1=self.p["alpha1"],
            alpha2=(self.p["alpha2"] if self.p["phase"] == "two" else None),
            settings_improve_fit=self.p["settings_improve_fit"],
            phase=self.p["phase"],
            verbose=self.p["verbose"],
            SNR_thresh=self.p["SNR_thresh"],
            SNR2_thresh=self.p["SNR2_thresh"],
            plot=self.p["plot"],
            perform_final_fit=self.p["perform_final_fit"],
        )
        return results

    def set(self, key, value):
        if key in self.p:
            self.p[key] = value
        else:
            print("Given key does not exist.")

    def batch_decomposition(self, *args, ilist=None, dct=None):
        """Science data should be AGD format ilist is either None or an integer list."""
        from gausspyplus.parallel_processing import parallel_processing

        if args:
            science_data_path = args[0]
            # Dump information to hard drive to allow multiprocessing
            pickle.dump(
                [self, science_data_path, ilist],
                open("batchdecomp_temp.pickle", "wb"),
                protocol=2,
            )
            parallel_processing.init_gausspy()
            result_list = parallel_processing.func(
                use_ncpus=self.p["use_ncpus"], function="gausspy_decompose"
            )
        else:
            #  if only a single spectrum is decomposed
            parallel_processing.init_gausspy([self, dct, ilist])
            result_list = [parallel_processing.decompose_one(0)]

        new_keys = [
            "index_fit",
            "amplitudes_fit",
            "fwhms_fit",
            "means_fit",
            "N_components_initial",
            "amplitudes_initial",
            "fwhms_initial",
            "means_initial",
            "amplitudes_fit_err",
            "fwhms_fit_err",
            "means_fit_err",
            "best_fit_rchi2",
            "best_fit_aicc",
            "N_components",
            "N_neg_res_peak",
            "N_blended",
            "log_gplus",
            "pvalue",
            "quality_control",
        ]

        output_data = {key: [] for key in new_keys}

        failed_decompositions = []

        for i, result in enumerate(result_list):
            try:
                # Save best-fit parameters
                output_data["index_fit"].append(result["index"])
                ncomps = result["N_components"]
                output_data["N_components"].append(ncomps)
                if ncomps > 0:
                    amps, fwhms, offsets = split_params(
                        params=result["best_fit_parameters"], ncomps=ncomps
                    )
                    output_data["amplitudes_fit"].append(amps)
                    output_data["fwhms_fit"].append(fwhms)
                    output_data["means_fit"].append(offsets)

                    # Final fit errors
                    amps_err, fwhms_err, offsets_err = split_params(
                        params=result["best_fit_errors"], ncomps=ncomps
                    )
                    output_data["amplitudes_fit_err"].append(amps_err)
                    output_data["fwhms_fit_err"].append(fwhms_err)
                    output_data["means_fit_err"].append(offsets_err)
                else:
                    for key in [
                        "amplitudes_fit",
                        "fwhms_fit",
                        "means_fit",
                        "amplitudes_fit_err",
                        "fwhms_fit_err",
                        "means_fit_err",
                    ]:
                        output_data[key].append([])

                # Save initial guesses if something was found
                ncomps_ini = len(result["initial_parameters"]) // 3
                output_data["N_components_initial"].append(ncomps_ini)
                if ncomps_ini > 0:
                    amps_ini, fwhms_ini, offsets_ini = split_params(
                        params=result["initial_parameters"], ncomps=ncomps_ini
                    )
                    output_data["means_initial"].append(offsets_ini)
                    output_data["fwhms_initial"].append(fwhms_ini)
                    output_data["amplitudes_initial"].append(amps_ini)
                else:
                    for key in [
                        "amplitudes_initial",
                        "fwhms_initial",
                        "means_initial",
                    ]:
                        output_data[key].append([])

                for key in [
                    "best_fit_rchi2",
                    "best_fit_aicc",
                    "N_neg_res_peak",
                    "N_blended",
                    "log_gplus",
                    "pvalue",
                    "quality_control",
                ]:
                    output_data[key].append(result[key] if key in result else None)
            except TypeError:
                if result is not None:
                    failed_decompositions.append(f"Problem with index {i}: {result}")

                for key in new_keys:
                    output_data[key].append(None)
                output_data["index_fit"][i] = i

        if failed_decompositions:
            print(
                "Could not fit the following spectra and replaced their entries with None:"
            )
            for item in failed_decompositions:
                print(item)

        return output_data
