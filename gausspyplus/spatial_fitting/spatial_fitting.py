import itertools
import os
import pickle
import textwrap
from pathlib import Path
from typing import Dict, Optional, List, Tuple, Union, Callable

import numpy as np
import scipy.ndimage as ndimage

from tqdm import tqdm

from gausspyplus.definitions.config_file import get_values_from_config_file
from gausspyplus.definitions.spectrum import Spectrum
from gausspyplus.decomposition.gp_plus import (
    get_fully_blended_gaussians,
    check_for_peaks_in_residual,
    check_for_negative_residual,
    get_best_fit_model,
)
from gausspyplus.definitions.model import Model
from gausspyplus.definitions.checks import BaseChecks
from gausspyplus.preparation.determine_intervals import (
    merge_overlapping_intervals,
    indices_of_fit_components_in_interval,
)
from gausspyplus.spatial_fitting.flags import (
    get_flags_rchi2,
    get_flags,
    get_flags_pvalue,
    get_flags_broad,
    get_flags_centroids,
    get_flags_ncomps,
)
from gausspyplus.decomposition.gaussian_functions import upper_limit_for_amplitude
from gausspyplus.spatial_fitting.grouping import (
    get_neighbors,
    group_fit_solutions,
    determine_average_values,
)
from gausspyplus.utils.misc import remove_elements_at_indices
from gausspyplus.spatial_fitting.ndimage_functions import (
    weighted_median,
    number_of_component_jumps,
    broad_components,
)
from gausspyplus.utils.output import set_up_logger, say, make_pretty_header
from gausspyplus.definitions.definitions import SettingsDefault, SettingsSpatialFitting


class SpatialFitting(SettingsDefault, SettingsSpatialFitting, BaseChecks):
    def __init__(
        self,
        path_to_pickle_file: Optional[Union[str, Path]] = None,
        path_to_decomp_file: Optional[Union[str, Path]] = None,
        fin_filename: Optional[Union[str, Path]] = None,
        config_file: Union[str, Path] = "",
    ):
        """Class implementing the two phases of spatially coherent refitting discussed in Riener+ 2019."""
        self.path_to_pickle_file = path_to_pickle_file
        self.path_to_decomp_file = path_to_decomp_file
        self.dirpath_gpy = None
        self.fin_filename = fin_filename

        self.constrain_fwhm = False
        self.pixel_range = None
        self._w_start = 1.0
        self._finalize = False

        if config_file:
            get_values_from_config_file(self, config_file, config_key="spatial fitting")

    def _check_settings(self) -> None:
        """Check user settings and raise error messages or apply corrections."""
        self.raise_exception_if_attribute_is_none("path_to_pickle_file")
        self.raise_exception_if_attribute_is_none("path_to_decomp_file")
        self.decomp_dirname = os.path.dirname(self.path_to_decomp_file)
        self.file = os.path.basename(self.path_to_decomp_file)
        self.filename, self.file_extension = os.path.splitext(self.file)

        if self.fin_filename is None:
            self.fin_filename = f"{self.filename}_sf-p1{self.suffix or ''}"

            if self.phase_two:
                self.fin_filename = (
                    f"{self.filename.replace('_sf-p1', '')}_sf-p2{self.suffix or ''}"
                )

        self.set_attribute_if_none("dirpath_gpy", os.path.dirname(self.decomp_dirname))
        self.set_attribute_if_none("rchi2_limit_refit", self.rchi2_limit)
        self.set_attribute_if_none("fwhm_factor_refit", self.fwhm_factor)
        self.set_attribute_if_none("flag_blended", self.refit_blended)
        self.set_attribute_if_none("flag_neg_res_peak", self.refit_neg_res_peak)
        self.set_attribute_if_none("flag_rchi2", self.refit_rchi2)
        self.set_attribute_if_none("flag_residual", self.refit_residual)
        self.set_attribute_if_none("flag_broad", self.refit_broad)
        self.set_attribute_if_none("flag_ncomps", self.refit_ncomps)
        self.raise_exception_if_all_attributes_are_none_or_false(
            [
                "refit_blended",
                "refit_neg_res_peak",
                "refit_rchi2",
                "refit_residual",
                "refit_broad",
                "refit_ncomps",
            ]
        )
        if self.flag_rchi2:
            self.raise_exception_if_attribute_is_none(
                "rchi2_limit",
                error_message="You need to set 'rchi2_limit' if 'flag_rchi2=True' or 'refit_rchi2=True'.",
            )

    def _initialize(self) -> None:
        """Read in data files and initialize parameters."""
        with open(self.path_to_pickle_file, "rb") as pickle_file:
            pickled_data = pickle.load(pickle_file, encoding="latin1")

        self.data = pickled_data["data_list"]
        self.errors = pickled_data["error"]
        # TODO: What to do if there is no header? self.shape and self.location are then not defined
        if "header" in pickled_data.keys():
            header = pickled_data["header"]
            self.shape = (header["NAXIS2"], header["NAXIS1"])
            self.length = header["NAXIS2"] * header["NAXIS1"]
            self.location = pickled_data["location"]
            n_channels = header["NAXIS3"]
        else:
            self.length = len(self.data)
            n_channels = len(self.data[0])
        self.channels = np.arange(n_channels)
        if self.max_fwhm is None:
            # TODO: Why the following definition?
            self.max_fwhm = int(n_channels / 3)

        self.signal_intervals = pickled_data["signal_ranges"]
        self.noise_spike_intervals = pickled_data["noise_spike_ranges"]

        with open(self.path_to_decomp_file, "rb") as pickle_file:
            # TODO: It could make sense to already cast quantities to numpy arrays here
            self.decomposition = pickle.load(pickle_file, encoding="latin1")

        self.n_indices = len(self.decomposition["index_fit"])

        self.decomposition["refit_iteration"] = [0] * self.n_indices

        self.neighbor_indices = np.array([None] * self.n_indices)
        self.neighbor_indices_all = np.array([None] * self.n_indices)

        self.nan_mask = np.isnan(
            [n or np.nan for n in self.decomposition["N_components"]]
        )

        if self.pixel_range is not None:
            self._mask_out_beyond_pixel_range()

        #  starting condition so that refitting iteration can start
        # self.mask_refitted = np.ones(1)
        # TODO: Make mask_refitted a boolean array
        self.mask_refitted = np.ones(self.n_indices)
        self.list_n_refit = []
        self.refitting_iteration = 0

        normalization_factor = 1 / (2 * (self.weight_factor + 1))
        self.w_2 = normalization_factor
        self.w_1 = self.weight_factor * normalization_factor
        self.min_p = self._w_start - self.w_2

    def _mask_out_beyond_pixel_range(self) -> None:
        locations = list(
            itertools.product(
                range(self.pixel_range["y"][0], self.pixel_range["y"][1]),
                range(self.pixel_range["x"][0], self.pixel_range["x"][1]),
            )
        )

        for idx, loc in enumerate(self.location):
            if loc not in locations:
                self.nan_mask[idx] = np.nan

    def _info_text(self, refit=False):
        text_phase_1 = (
            ""
            if not refit
            else textwrap.dedent(
                f"""
            For phase 1:
            Exclude flagged spectra as possible refit solutions in first refit attempts: {self.exclude_flagged}
            Use also flagged spectra as refit solutions in case no new best fit could be obtained from unflagged spectra: {self.use_all_neighors}"""
            )
        )
        return text_phase_1 + textwrap.dedent(
            f"""
            {('Flagging', 'Refitting')[refit]}:
             - Blended components: {(self.flag_blended, self.refit_blended)[refit]}'
             - Negative residual features: {(self.flag_neg_res_peak, self.refit_neg_res_peak)[refit]}
             - Broad components: {(self.flag_broad, self.refit_broad)[refit]}
             \t flagged if FWHM of broadest component in spectrum is:
             \t >= {(self.fwhm_factor, self.fwhm_factor_refit)[refit]} times the FWHM of second broadest component
             \t or
             \t >= {(self.fwhm_factor, self.fwhm_factor_refit)[refit]} times any FWHM in >= {self.broad_neighbor_fraction:.0%} of its neigbors
             - High reduced chi2 values (> {(self.rchi2_limit, self.rchi2_limit_refit)[refit]}): {(self.flag_rchi2, self.refit_rchi2)[refit]}
             - Non-Gaussian distributed residuals: {(self.flag_residual, self.refit_residual)[refit]}
             - Differing number of components: {(self.flag_ncomps, self.refit_ncomps)[refit]}"""
        )

    def _getting_ready(self) -> None:
        """Set up logger and write initial output to terminal."""
        if self.log_output:
            self.logger = set_up_logger(
                self.dirpath_gpy, self.filename, method="g+_spatial_refitting"
            )
        else:
            self.logger = False
        say(
            message=make_pretty_header(
                f"Spatial refitting - Phase {1 + self.phase_two}"
            ),
            logger=self.logger,
        )
        say(self._info_text(refit=False), logger=self.logger)
        if not self.phase_two:
            say(self._info_text(refit=True), logger=self.logger)

    def finalize(self):
        # TODO: What is the return type of this function?
        # TODO: Is this function called by the Finalize class?
        self.logger = False
        self.phase_two = True
        self._finalize = True
        self._check_settings()
        self._initialize()
        self.determine_spectra_for_flagging()

        self.refitting_iteration = 1
        self._check_indices_refit()
        return self._refitting()

    def spatial_fitting(self, continuity: bool = False) -> None:
        """Start the spatially coherent refitting.

        :param continuity: Set to 'True' for phase 2 of the spatially coherent refitting (coherence of centroid
        positions).
        """

        self.phase_two = continuity
        self._check_settings()
        self._initialize()
        self._getting_ready()
        if self.phase_two:
            self.list_n_refit.append([self.length])
            self._check_continuity()
        else:
            self._determine_spectra_for_refitting()

    def _define_mask(
        self,
        key: str,
        limit: Union[int, float],
        flag: bool,
        comparison_func: Callable = np.greater,
    ) -> np.ndarray:
        """Create boolean mask with data values exceeding the defined limits set to 'True'.

        This mask is only created if 'flag=True'.

        :param key: Dictionary key of the parameter: 'N_blended', 'N_neg_res_peak', or 'best_fit_rchi2'.
        :param limit: Upper limit of the corresponding value.
        :param flag: User-defined flag for the corresponding dictionary parameter.
        :param comparison_func: Function that compares the array to limit, e.g. np.less or np.greater
        :return: Boolean mask with values exceeding 'limit' set to 'True'.
        """

        return (
            comparison_func(
                np.where(self.nan_mask, limit, self.decomposition[key]),
                limit,
            )
            if flag
            else np.zeros(self.length, dtype=bool)
        )

    def _define_mask_broad_limit(self) -> np.ndarray:
        # TODO: Can _define_mask_broad_limit be combined with _broad_components?
        """Return boolean mask identifying the location of broad fit components."""
        return np.array(
            [
                False
                if (fwhms is None or len(fwhms) == 0)
                else np.any(np.array(fwhms) > self.max_fwhm)
                for fwhms in self.decomposition["fwhms_fit"]
            ]
        )

    def _check_individual_spectrum_for_broad_fit_component(self, fwhms):
        # In case there is only one fit parameter there are no other components to compare to
        if fwhms is None or len(fwhms) < 2:
            return False
        # In case of multiple fit parameters select the one with the largest FWHM value and check whether it
        #  exceeds the second largest FWHM value in that spectrum by a factor of 'self.fwhm_factor'; also check if
        #  the absolute difference of their values exceeds 'self.fwhm_separation' to avoid narrow components.
        fwhms = sorted(fwhms)
        return (fwhms[-1] > self.fwhm_factor * fwhms[-2]) and (
            fwhms[-1] - fwhms[-2]
        ) > self.fwhm_separation

    def _define_mask_broad(self) -> np.ndarray:
        """Return a boolean mask indicating the location of broad fit components."""
        is_broad_compared_to_other_fit_components_in_spectrum = np.array(
            [
                self._check_individual_spectrum_for_broad_fit_component(fwhms)
                for fwhms in self.decomposition["fwhms_fit"]
            ]
        )

        # Check if the fit component with the largest FWHM value of a spectrum satisfies the criteria to be flagged as
        #  a broad component by comparing it to the largest FWHM values of its 8 immediate neighbors.
        #  The input 2D array consists of the maximum FWHM fit component per spectrum.
        is_broad_compared_to_fit_components_of_neighbors = (
            ndimage.generic_filter(
                input=np.array(
                    [
                        np.nan if (fwhms is None or len(fwhms) == 0) else max(fwhms)
                        for fwhms in self.decomposition["fwhms_fit"]
                    ]
                ).reshape(self.shape),
                function=broad_components,
                footprint=np.ones((3, 3)),
                mode="constant",
                cval=np.nan,
                extra_arguments=(
                    self.fwhm_factor,
                    self.fwhm_separation,
                    self.broad_neighbor_fraction,
                ),
            )
            .flatten()
            .astype("bool")
        )
        # TODO: is nan_mask masking needed if _mask_out_beyond_pixel_range is set?
        # is_broad_compared_to_other_fit_components_in_spectrum[self.nan_mask] = False
        return (
            is_broad_compared_to_other_fit_components_in_spectrum
            | is_broad_compared_to_fit_components_of_neighbors
        )

    def _define_mask_neighbor_ncomps(
        self, flag: bool
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """Create a boolean mask indicating the location of component jumps.

        :param flag: User-defined 'self.flag_ncomps' parameter.
        :return: (mask_neighbor, ncomps_jumps, ncomps_1d)
            - mask_neighbor: Boolean mask indicating the location of component jump.
            - ncomps_jumps: Array containing the information of how many component jumps occur at which location.
            - ncomps_1d: Array containing the information about the number of fitted components per location.
        """

        if not flag:
            return np.zeros(self.length, dtype=bool), None, None

        self.ncomps = np.where(
            self.nan_mask,
            np.nan,
            np.array(self.decomposition["N_components"], dtype=float),
        )

        # TODO: Is it a problem that self.ncomps is a float array?
        ncomps_wmedian = ndimage.generic_filter(
            input=self.ncomps.reshape(self.shape),
            function=weighted_median,
            footprint=np.ones((3, 3)),
            mode="constant",
            cval=np.nan,
        ).flatten()

        ncomps_jumps = ndimage.generic_filter(
            input=self.ncomps.reshape(self.shape),
            function=number_of_component_jumps,
            footprint=np.ones((3, 3)),
            mode="reflect",
            cval=np.nan,
            extra_arguments=(self.max_jump_comps,),
        ).flatten()

        mask_neighbor = np.logical_or(
            np.where(self.nan_mask, False, ncomps_wmedian > self.max_diff_comps),
            np.where(self.nan_mask, False, ncomps_jumps > self.n_max_jump_comps),
        )
        return mask_neighbor, ncomps_wmedian, ncomps_jumps

    def determine_spectra_for_flagging(self) -> None:
        """Flag spectra not satisfying user-defined flagging criteria."""
        self.mask_blended = self._define_mask("N_blended", 0, self.flag_blended)
        self.mask_neg_res_peak = self._define_mask(
            "N_neg_res_peak", 0, self.flag_neg_res_peak
        )
        self.mask_rchi2_flagged = self._define_mask(
            "best_fit_rchi2", self.rchi2_limit, self.flag_rchi2
        )
        self.mask_residual = self._define_mask(
            "pvalue", self.min_pvalue, self.flag_residual, comparison_func=np.less
        )
        self.mask_broad_flagged = (
            self._define_mask_broad()
            if self.flag_broad
            else np.zeros(self.length, dtype=bool)
        )
        self.mask_broad_limit = (
            self._define_mask_broad_limit()
            if self.flag_broad
            else np.zeros(self.length, dtype=bool)
        )
        (
            self.mask_ncomps,
            self.ncomps_wmedian,
            self.ncomps_jumps,
        ) = self._define_mask_neighbor_ncomps(self.flag_ncomps)

        if self._finalize:
            return

        self.count_flags = np.sum(
            (
                self.mask_blended,
                self.mask_neg_res_peak,
                self.mask_broad_flagged,
                self.mask_rchi2_flagged,
                self.mask_residual,
                self.mask_ncomps,
            ),
            axis=0,
        )

        if self.phase_two:
            text = textwrap.dedent(
                f"""
                Flags:
                - {self.mask_blended.sum()} spectra w/ blended components
                - {self.mask_neg_res_peak.sum()} spectra w/ negative residual feature
                - {self.mask_broad_flagged.sum()} spectra w/ broad feature
                \t (info: {self.mask_broad_limit.sum()} spectra w/ a FWHM > {int(self.max_fwhm)} channels)
                - {self.mask_rchi2_flagged.sum()} spectra w/ high rchi2 value
                - {self.mask_residual.sum()} spectra w/ residual not passing normality test
                - {self.mask_ncomps.sum()} spectra w/ differing number of components"""
            )
            say(text, logger=self.logger)

    def _define_mask_refit(self) -> None:
        """Select spectra to refit in phase 1 of the spatially coherent refitting."""
        mask_refit = np.sum(
            (
                self.mask_blended * self.refit_blended,
                self.mask_neg_res_peak * self.refit_neg_res_peak,
                self.mask_broad_refit * self.refit_broad,
                self.mask_rchi2_refit * self.refit_rchi2,
                self.mask_residual * self.refit_residual,
                self.mask_ncomps * self.refit_ncomps,
            ),
            axis=0,
            dtype=bool,
        )
        self.indices_refit = np.array(self.decomposition["index_fit"])[mask_refit]
        # self.indices_refit = self.indices_refit[886:888]  # for debugging
        self.locations_refit = np.take(
            np.array(self.location), self.indices_refit, axis=0
        )

    def _determine_spectra_for_refitting(self) -> None:
        """Determine spectra for refitting in phase 1 of the spatially coherent refitting."""
        say("\ndetermine spectra that need refitting...", logger=self.logger)

        # Flag spectra based on user-defined criteria
        self.determine_spectra_for_flagging()

        # Determine new masks for spectra that do not satisfy the user-defined criteria for broad components and
        # reduced chi-square values. This is done because users can opt to use different values for flagging and
        # refitting for these two criteria.
        self.mask_broad_refit = (
            self._define_mask_broad()
            if self.refit_broad
            else np.zeros(self.length, dtype=bool)
        )
        self.mask_rchi2_refit = self._define_mask(
            key="best_fit_rchi2", limit=self.rchi2_limit_refit, flag=self.refit_rchi2
        )

        # Select spectra for refitting based on user-defined criteria
        self._define_mask_refit()

        n_flagged_blended = self.mask_blended.sum()
        n_flagged_neg_res_peak = self.mask_neg_res_peak.sum()
        n_flagged_broad = self.mask_broad_flagged.sum()
        n_flagged_rchi2 = self.mask_rchi2_flagged.sum()
        n_flagged_residual = self.mask_residual.sum()
        n_flagged_ncomps = self.mask_ncomps.sum()

        n_refit_list = [
            n_refit_blended := n_flagged_blended * self.refit_blended,
            n_refit_neg_res_peak := n_flagged_neg_res_peak * self.refit_neg_res_peak,
            n_refit_broad := self.mask_broad_refit.sum() * self.refit_broad,
            n_refit_rchi2 := self.mask_rchi2_refit.sum() * self.refit_rchi2,
            n_refit_residual := n_flagged_residual * self.refit_residual,
            n_refit_ncomps := n_flagged_ncomps * self.refit_ncomps,
        ]

        n_spectra = sum(x is not None for x in self.decomposition["N_components"])
        n_indices_refit = len(self.indices_refit)
        try:
            n_fraction_refit = n_indices_refit / n_spectra
        except ZeroDivisionError:
            n_fraction_refit = 0

        text = textwrap.dedent(
            f"""
            {n_indices_refit} out of {n_spectra} spectra ({n_fraction_refit:.2%}) selected for refitting:
             - {n_refit_blended} spectra w/ blended components ({n_flagged_blended} flagged)
             - {n_refit_neg_res_peak} spectra w/ negative residual feature ({n_flagged_neg_res_peak} flagged)
             - {n_refit_broad} spectra w/ broad feature ({n_flagged_broad} flagged)
             \t (info: {self.mask_broad_limit.sum()} spectra w/ a FWHM > {int(self.max_fwhm)} channels)
             - {n_refit_rchi2} spectra w/ high rchi2 value ({n_flagged_rchi2} flagged)
             - {n_refit_residual} spectra w/ residual not passing normality test ({n_flagged_residual} flagged)
             - {n_refit_ncomps} spectra w/ differing number of components ({n_flagged_ncomps} flagged)"""
        )
        say(text, logger=self.logger)

        # Check if the stopping criterion is fulfilled
        if self.only_print_flags:
            return
        elif self._stopping_criterion(n_refit_list):
            self._save_final_results()
        else:
            self.list_n_refit.append(n_refit_list)
            self.refitting_iteration += 1
            self._refitting()

    def _stopping_criterion(self, n_refit_list: List) -> bool:
        """Check if spatial refitting iterations should be stopped."""
        #  stop refitting if the user-defined maximum number of iterations are reached
        if self.refitting_iteration >= self.max_refitting_iteration:
            return True
        #  stop refitting if the number of spectra selected for refitting is identical to a previous iteration
        if n_refit_list in self.list_n_refit:
            return True
        #  stop refitting if the number of spectra selected for refitting got higher than in the previous iteration for each user-defined criterion
        if self.refitting_iteration > 0:
            return all(
                n_refit_list[i] >= min(n[i] for n in self.list_n_refit)
                for i in range(len(n_refit_list))
            )

    def _refitting(self) -> None:
        """Refit spectra with multiprocessing routine."""
        say(
            f"\nstart refit iteration #{self.refitting_iteration}...",
            logger=self.logger,
        )

        #  initialize the multiprocessing routine

        import gausspyplus.parallel_processing

        gausspyplus.parallel_processing.parallel_processing.init(
            [self.indices_refit, [self]]
        )

        #  try to refit spectra via the multiprocessing routine
        results_list = gausspyplus.parallel_processing.parallel_processing.func(
            use_ncpus=self.use_ncpus,
            function="refit_phase_2" if self.phase_two else "refit_phase_1",
        )
        print("SUCCESS")

        if self._finalize:
            return results_list

        #  reset the mask for spectra selected for refitting
        self.mask_refitted = np.array([0] * self.n_indices)

        keys = [
            "amplitudes_fit",
            "fwhms_fit",
            "means_fit",
            "amplitudes_fit_err",
            "fwhms_fit_err",
            "means_fit_err",
            "best_fit_rchi2",
            "best_fit_aicc",
            "N_components",
            "pvalue",
            "N_neg_res_peak",
            "N_blended",
        ]

        count_selected, count_refitted = 0, 0

        #  collect results of the multiprocessing routine

        for i, item in enumerate(results_list):
            if not isinstance(item, list):
                say(f"Error for spectrum with index {i}: {item}", logger=self.logger)
                continue

            index, fit_results, indices_neighbors, is_successful_refit = item
            if is_successful_refit:
                count_selected += 1
            self.neighbor_indices[index] = indices_neighbors
            if fit_results is not None:
                count_refitted += 1
                self.decomposition["refit_iteration"][index] += 1
                self.mask_refitted[index] = 1
                for key in keys:
                    self.decomposition[key][index] = fit_results[key]

        #  print statistics of the refitting iteration to the terminal

        refit_percent = 0 if count_selected == 0 else count_refitted / count_selected
        text = textwrap.dedent(
            f"""
            Results of the refit iteration:
            Tried to refit {count_selected} spectra
            Successfully refitted {count_refitted} spectra ({refit_percent:.2%})\n
            ***"""
        )
        say(text, logger=self.logger)

        #  check if one of the stopping criteria is fulfilled

        if self.phase_two:
            if self._stopping_criterion([count_refitted]):
                self.min_p -= self.w_2
                self.list_n_refit = [[self.length]]
                self.mask_refitted = np.array([1] * self.n_indices)
            else:
                self.list_n_refit.append([count_refitted])

            if self.min_p < self.min_weight:
                self._save_final_results()
            else:
                self._check_continuity()
        else:
            self._determine_spectra_for_refitting()

    def _determine_neighbor_indices(
        self, indices_of_all_neighbors: np.ndarray, include_flagged: bool = False
    ) -> np.ndarray:
        """Determine indices of all valid neighboring pixels.

        :param indices_of_all_neighbors: Indices of all valid neighboring spectra in the form [idx1, ..., idxN]
        :param include_flagged: Whether to keep neighboring spectra that were flagged.
        :return: Selection of valid neighboring spectra.
        """

        # TODO: Compute this once in the beginning, so it does not get recomputed all the time?
        # Here we select all indices of spectra that are not NaN and have fit components
        valid_indices = np.array(self.decomposition["index_fit"])[
            np.flatnonzero(self.decomposition["N_components"])
        ]
        if not include_flagged:
            # Whether to exclude all flagged neighboring spectra as well that were not selected for refitting
            indices_flagged = (
                # TODO: Compute this somewhere else so it does not get recomputed all the time?
                np.array(self.decomposition["index_fit"])[self.count_flags.astype(bool)]
                if self.exclude_flagged
                else self.indices_refit
            )
            valid_indices = np.setdiff1d(valid_indices, indices_flagged)

        # Use only neighboring spectra for refitting that are not masked out and have fit components
        indices_neighbors = np.intersect1d(indices_of_all_neighbors, valid_indices)

        if indices_neighbors.size > 1:
            # Sort neighboring fit solutions according to lowest value of reduced chi-square.
            # TODO: change this so that this gets sorted according to the lowest difference of the reduced chi-square
            #  values to the ideal value of 1 to prevent using fit solutions that 'overfit' the data
            sort_criterion = (
                self.count_flags[indices_neighbors]
                if include_flagged
                else np.array(self.decomposition["best_fit_rchi2"])[indices_neighbors]
            )
            indices_neighbors = indices_neighbors[np.argsort(sort_criterion)]

        return indices_neighbors

    def refit_spectrum_phase_1(self, index: int, i: int) -> List:
        """Refit a spectrum based on neighboring unflagged fit solutions.

        :param index: Index ('index_fit' keyword) of the spectrum that will be refit.
        :param i: List index of the entry in the list that is handed over to the multiprocessing routine.
        :return: A list in the form of [index, fit_results, indices_neighbors, is_successful_refit] in case of a
        successful refit; otherwise [index, 'None', indices_neighbors, is_successful_refit] is returned.
        """

        spectrum = Spectrum(
            intensity_values=self.data[index],
            channels=self.channels,
            rms_noise=self.errors[index][0],
            signal_intervals=self.signal_intervals[index],
            noise_spike_intervals=self.noise_spike_intervals[index],
        )

        flags = np.array(["residual", "broad", "blended", "None"])[
            [
                self.refit_neg_res_peak * self.mask_neg_res_peak[index],
                self.refit_broad * self.mask_broad_refit[index],
                self.refit_blended * self.mask_blended[index],
                True,
            ]
        ]

        indices_neighbors_ = get_neighbors(
            location=self.locations_refit[i], shape=self.shape
        )

        indices_of_unflagged_neighbors = self._determine_neighbor_indices(
            indices_of_all_neighbors=indices_neighbors_, include_flagged=False
        )
        indices_neighbors_for_individual_refit = indices_of_unflagged_neighbors
        indices_neighbors_for_grouping = indices_of_unflagged_neighbors

        for include_flagged_spectra in np.unique((False, self.use_all_neighors)):
            if include_flagged_spectra:
                indices_of_flagged_and_unflagged_neighbors = (
                    self._determine_neighbor_indices(
                        indices_of_all_neighbors=indices_neighbors_,
                        include_flagged=True,
                    )
                )
                # The following is necessary to avoid repeating refits with neighboring solutions that we already tried
                indices_neighbors_for_individual_refit = np.setdiff1d(
                    indices_of_flagged_and_unflagged_neighbors,
                    indices_neighbors_for_grouping,
                )
                indices_neighbors_for_grouping = (
                    indices_of_flagged_and_unflagged_neighbors
                )

            # Skip refitting if there are no valid neighbors available
            if indices_neighbors_for_grouping.size == 0:
                continue

            # TODO: Check if condition with indices_neighbors_for_grouping is correct here
            # Skip refitting if there were no changes of neighboring fit solutions in the last iteration
            if (
                np.array_equal(
                    indices_neighbors_for_grouping, self.neighbor_indices[index]
                )
                and not self.mask_refitted[indices_neighbors_for_grouping].sum()
            ):
                continue

            # Try to refit the spectrum with fit solution of individual spectra from selected neighbors
            for flag in flags:
                (
                    fit_results,
                    is_successful_refit,
                ) = self._try_refit_with_individual_neighbors(
                    index=index,
                    spectrum=spectrum,
                    indices_neighbors=indices_neighbors_for_individual_refit,
                    flag=flag,
                )
                if is_successful_refit:
                    return [
                        index,
                        fit_results,
                        indices_neighbors_for_grouping,
                        is_successful_refit,
                    ]

            # Try to refit the spectrum by grouping the fit solutions of all selected neighboring spectra
            if indices_neighbors_for_grouping.size > 1:
                fit_results, is_successful_refit = self._try_refit_with_grouping(
                    index=index,
                    spectrum=spectrum,
                    indices_neighbors=indices_neighbors_for_grouping,
                )
                if is_successful_refit:
                    return [
                        index,
                        fit_results,
                        indices_neighbors_for_grouping,
                        is_successful_refit,
                    ]

        return [index, None, indices_neighbors_for_grouping, False]

    def _try_refit_with_grouping(
        self,
        index: int,
        spectrum: Spectrum,
        indices_neighbors: np.ndarray,
    ) -> Tuple[Optional[Dict], bool]:
        """Try to refit a spectrum by grouping all neighboring unflagged fit solutions.

        :param index: Index ('index_fit' keyword) of the spectrum that will be refit.
        :param spectrum: Spectrum to refit.
        :param indices_neighbors: Array containing the indices of all neighboring fit solutions that should be used
        for the grouping.
        :return: tuple (fit_results, is_successful_refit)
            - fit_results: contains information about the new best fit solution in case of a successful refit,
            otherwise it is `None`.
            - is_successful_refit: states whether the refit was successful.
        """

        #  prepare fit parameter values of all unflagged neighboring fit solutions for the grouping
        amps, means, fwhms = self._get_initial_values(indices_neighbors)
        is_successful_refit = False

        #  Group fit parameter values of all unflagged neighboring fit solutions and try to refit the spectrum with the
        #  new resulting average fit parameter values. First we try to group the fit solutions only by their mean
        #  position values. If this does not yield a new successful refit, we group the fit solutions by their mean
        #  position and FWHM values.

        for split_fwhm in [False, True]:
            fit_components = group_fit_solutions(
                amps_tot=amps,
                means_tot=means,
                fwhms_tot=fwhms,
                split_fwhm=split_fwhm,
                mean_separation=self.mean_separation,
                fwhm_separation=self.fwhm_separation,
            )
            fit_components = determine_average_values(
                intensity_values=spectrum.intensity_values,
                fit_components=fit_components,
                intensity_threshold=self.snr * spectrum.rms_noise,
                mean_separation=self.mean_separation,
                constrain_fwhm=self.constrain_fwhm,
                fwhm_separation=self.fwhm_separation,
            )

            #  try refit with the new average fit solution values

            if len(fit_components.keys()) > 0:
                fit_results = self._gaussian_fitting(
                    spectrum=spectrum,
                    fit_components=fit_components,
                )
                is_successful_refit = True
                if fit_results is None:
                    continue
                if self._choose_new_fit(fit_results, index):
                    return fit_results, is_successful_refit

        return None, is_successful_refit

    def _skip_index_for_refitting(self, index: int, index_neighbor: int) -> bool:
        """Check whether neighboring fit solution should be skipped.

        We want to exclude (most likely futile) refits with initial guesses from the fit solutions of neighboring
        spectra if the same fit solutions were already used in a previous iteration.

        :param index: Index (`index_fit` keyword) of the spectrum that will be refit.
        :param index_neighbor: Index (`index_fit` keyword) of the neighboring fit solution.
        :return: Whether to skip the neighboring fit solution for an attempted refit.
        """

        return (
            self.refitting_iteration > 1
            #  check if spectrum was selected for refitting in any of the  previous iterations
            and self.neighbor_indices[index] is not None
            #  check if neighbor was used in that refitting iteration
            and index_neighbor in self.neighbor_indices[index]
            #  check if neighbor was refit in previous iteration
            and not self.mask_refitted[index_neighbor]
        )

    def _try_refit_with_individual_neighbors(
        self,
        index: int,
        spectrum: Spectrum,
        indices_neighbors: np.ndarray,
        interval: Optional[List] = None,
        n_centroids: Optional[int] = None,
        flag: str = "none",
        updated_fit_results: Optional[Dict] = None,
    ) -> Tuple[Optional[Dict], bool]:
        """Try to refit a spectrum with the fit solution of an unflagged neighboring spectrum.

        :param index: Index ('index_fit' keyword) of the spectrum that will be refit.
        :param spectrum: Spectrum to refit.
        :param indices_neighbors: Array containing the indices of all neighboring fit solutions that should be used
        for the grouping.
        :param interval: List specifying the interval of spectral channels containing the flagged feature in the
        form of [lower, upper]. Only used in phase 2 of the spatially coherent refitting.
        :param n_centroids: Number of centroid positions that should be present in interval.
        :param flag: Flagged criterion that should be refit: 'broad', 'blended', or 'residual'.
        :param updated_fit_results: Only used in phase 2 of the spatially coherent refitting, in case the best
        fit solution was already updated in a previous iteration.
        :return: tuple (fit_results, is_successful_refit)
            - fit_results: Information about the new best fit solution in case of a successful refit. Otherwise
            'None' is returned.
            - is_successful_refit: Information of whether there was a new successful refit.
        """

        fit_components = None
        is_successful_refit = False

        for index_neighbor in indices_neighbors:
            #  check whether to use the neighboring fit solution or skip it
            if self._skip_index_for_refitting(index, index_neighbor):
                continue

            #  try to only replace part of the fit solution with new initial guesses from the neighboring fit solution
            #  for components flagged as broad, blended, or causing a negative residual feature. Otherwise use the
            #  entire fit solution of the neighboring spectrum.

            # TODO: check if this if-elif condition is correct and can be simplified
            if flag in {"broad", "blended", "residual"}:
                fit_components = self._replace_flagged_interval(
                    index=index,
                    index_neighbor=index_neighbor,
                    intensity_values=spectrum.intensity_values,
                    rms=spectrum.rms_noise,
                    flag=flag,
                )
            elif interval is not None:
                fit_components = self._replace_flagged_interval(
                    index=index,
                    index_neighbor=index_neighbor,
                    intensity_values=spectrum.intensity_values,
                    rms=spectrum.rms_noise,
                    interval=interval,
                    updated_fit_results=updated_fit_results,
                )
            else:
                fit_components = self._get_initial_values_from_neighbor(
                    i=index_neighbor, intensity_values=spectrum.intensity_values
                )

            if fit_components is None:
                continue

            #  try to refit with new fit solution

            fit_results = self._gaussian_fitting(
                spectrum=spectrum,
                fit_components=fit_components,
            )
            # TODO: Does setting is_successful_refit to True here make sense?
            is_successful_refit = True
            if fit_results is None:
                continue
            if self._choose_new_fit(
                fit_results=fit_results,
                index=index,
                updated_fit_results=updated_fit_results,
                interval=interval,
                n_centroids=n_centroids,
            ):
                return fit_results, is_successful_refit

        return None, is_successful_refit

    def _get_refit_interval(
        self,
        intensity_values: np.ndarray,
        rms: float,
        amps: List,
        fwhms: List,
        means: List,
        flag: str,
    ) -> List:
        """Get interval of spectral channels containing flagged feature selected for refitting.

        :param intensity_values: Intensity values of spectrum to refit.
        :param rms: Root-mean-square noise value of the spectrum.
        :param amps: List of amplitude values of the fitted components.
        :param fwhms: List of FWHM values of the fitted components.
        :param means: List of mean position values of the fitted components.
        :param flag: Flagged criterion that should be refit: 'broad', 'blended', or 'residual'.
        :return: List specifying the interval of spectral channels containing the flagged feature in the form of
        [lower, upper].
        """

        #  for component flagged as broad select the interval [mean - FWHM, mean + FWHM]
        if flag == "blended":
            params = amps + fwhms + means
            separation_factor = self.decomposition[
                "improve_fit_settings"
            ].separation_factor
            indices = get_fully_blended_gaussians(
                params, separation_factor=separation_factor
            )
            lower = max(0, min(np.array(means)[indices] - np.array(fwhms)[indices]))
            upper = max(np.array(means)[indices] + np.array(fwhms)[indices])
        elif flag == "broad":
            idx = np.argmax(np.array(fwhms))  # idx of broadest component
            lower = max(0, means[idx] - fwhms[idx])
            upper = means[idx] + fwhms[idx]
        elif flag == "residual":
            settings_improve_fit = self.decomposition["improve_fit_settings"]

            #  TODO: What if multiple negative residual features occur in one spectrum?
            idx = check_for_negative_residual(
                model=Model(
                    spectrum=Spectrum(
                        intensity_values=intensity_values,
                        channels=self.channels,
                        rms_noise=rms,
                    )
                ),
                settings_improve_fit=settings_improve_fit,
                get_idx=True,
            )
            if idx is None:
                #  TODO: check if self.channels[-1] causes problems
                return [self.channels[0], self.channels[-1]]
            lower = max(0, means[idx] - fwhms[idx])
            upper = means[idx] + fwhms[idx]

        return [lower, upper]

    def _replace_flagged_interval(
        self,
        index: int,
        index_neighbor: int,
        intensity_values: np.ndarray,
        rms: float,
        flag: str = "none",
        interval: Optional[List] = None,
        updated_fit_results: Optional[Dict] = None,
    ) -> Dict:
        """Update initial guesses for fit components by replacing flagged feature with a neighboring fit solution.

        :param index: Index ('index_fit' keyword) of the spectrum that will be refit.
        :param index_neighbor: Index ('index_fit' keyword) of the neighboring fit solution.
        :param intensity_values: Intensity values of spectrum to refit.
        :param rms: Root-mean-square noise value of the spectrum.
        :param flag: Flagged criterion that should be refit: 'broad', 'blended', or 'residual'.
        :param interval: List specifying the interval of spectral channels containing the flagged feature in the
        form of [lower, upper].
        :param updated_fit_results: Only used in phase 2 of the spatially coherent refitting, in case the best
        fit solution was already updated in a previous iteration.
        :return: Dictionary containing updated initial guesses for the fit solution.
        """

        #  for phase 2 of the spatially coherent refitting; if fit solution was already updated in previous iteration
        if updated_fit_results is not None:
            amps = updated_fit_results["amplitudes_fit"]
            fwhms = updated_fit_results["fwhms_fit"]
            means = updated_fit_results["means_fit"]
            means_err = updated_fit_results["means_fit_err"]
        else:
            amps = self.decomposition["amplitudes_fit"][index]
            fwhms = self.decomposition["fwhms_fit"][index]
            means = self.decomposition["means_fit"][index]
            means_err = self.decomposition["means_fit_err"][index]

        #  remove fit solution(s) of fit component(s) that are causing the flagged feature

        if interval is None:
            interval = self._get_refit_interval(
                intensity_values=intensity_values,
                rms=rms,
                amps=amps,
                fwhms=fwhms,
                means=means,
                flag=flag,
            )
        indices, interval = indices_of_fit_components_in_interval(
            fwhms=fwhms, means=means, interval=interval
        )

        amps = remove_elements_at_indices(amps, indices)
        fwhms = remove_elements_at_indices(fwhms, indices)
        means = remove_elements_at_indices(means, indices)
        means_err = remove_elements_at_indices(means_err, indices)

        #  get new initial guess(es) for removed component(s) from neighboring fit solution

        amps_new = self.decomposition["amplitudes_fit"][index_neighbor]
        fwhms_new = self.decomposition["fwhms_fit"][index_neighbor]
        means_new = self.decomposition["means_fit"][index_neighbor]
        means_err_new = self.decomposition["means_fit_err"][index_neighbor]

        #  check which of the neighboring fit components overlap with the interval containing the flagged feature(s)
        indices, interval = indices_of_fit_components_in_interval(
            fwhms=fwhms_new, means=means_new, interval=interval
        )

        if len(indices) == 0:
            return

        #  discard all neighboring fit components not overlappting with the interval containing the flagged feature(s)
        remove_indices = np.delete(np.arange(len(amps_new)), indices)
        amps_new = remove_elements_at_indices(amps_new, remove_indices)
        fwhms_new = remove_elements_at_indices(fwhms_new, remove_indices)
        means_new = remove_elements_at_indices(means_new, remove_indices)
        means_err_new = remove_elements_at_indices(means_err_new, remove_indices)

        if len(amps_new) == 0:
            return

        #  get best fit with new fit solution(s) for only the interval that contained the removed components

        idx_lower = int(interval[0])
        idx_upper = int(interval[1]) + 2

        fit_components_in_interval = {}
        for amp, fwhm, mean, mean_err in zip(
            amps_new, fwhms_new, means_new, means_err_new
        ):
            fit_components_in_interval = self._add_initial_value_to_dict(
                fit_components=fit_components_in_interval,
                intensity_values=intensity_values[idx_lower:idx_upper],
                amp=amp,
                fwhm=fwhm,
                mean=mean - idx_lower,
                mean_bound=max(self.mean_separation, mean_err),
            )

        fit_results = self._gaussian_fitting(
            spectrum=Spectrum(
                intensity_values=intensity_values[idx_lower:idx_upper],
                channels=np.arange(len(intensity_values[idx_lower:idx_upper])),
                rms_noise=rms,
            ),
            fit_components=fit_components_in_interval,
            params_only=True,
        )

        if fit_results is None:
            return

        #  create new dictionary of fit solution(s) by combining new fit component(s) taken from neighboring spectrum with the remaining fit component(s) outside the flagged interval

        fit_components = {}
        for amp, fwhm, mean, mean_err in zip(
            fit_results["amplitudes_fit"],
            fit_results["fwhms_fit"],
            fit_results["means_fit"],
            fit_results["means_fit_err"],
        ):
            fit_components = self._add_initial_value_to_dict(
                fit_components=fit_components,
                intensity_values=intensity_values,
                amp=amp,
                fwhm=fwhm,
                mean=mean + idx_lower,
                mean_bound=max(self.mean_separation, mean_err),
            )

        for amp, fwhm, mean, mean_err in zip(amps, fwhms, means, means_err):
            fit_components = self._add_initial_value_to_dict(
                fit_components=fit_components,
                intensity_values=intensity_values,
                amp=amp,
                fwhm=fwhm,
                mean=mean,
                mean_bound=max(self.mean_separation, mean_err),
            )

        return fit_components

    def _add_initial_value_to_dict(
        self,
        fit_components: Dict,
        intensity_values: np.ndarray,
        amp: float,
        fwhm: float,
        mean: float,
        mean_bound: float,
    ) -> Dict:
        """Update dictionary of fit components with new component.

        :param fit_components: Dictionary of fit components.
        :param intensity_values: Intensity values of spectrum to refit.
        :param amp: Amplitude value of fit component.
        :param fwhm: FWHM value of fit component.
        :param mean: Mean position value of fit component.
        :param mean_bound: Relative bound (upper and lower) of mean position value of fit component.
        :return: Updated dictionary of fit components.
        """

        #  TODO: add here also mean +/- stddev??

        fit_components[str(len(fit_components) + 1)] = {
            "amp_ini": amp,
            "mean_ini": mean,
            "fwhm_ini": fwhm,
            "amp_bounds": [
                0.0,
                upper_limit_for_amplitude(
                    intensity_values, mean, fwhm, buffer_factor=1.1
                ),
            ],
            "mean_bounds": [max(0.0, mean - mean_bound), mean + mean_bound],
            "fwhm_bounds": [
                max(0.0, fwhm - self.fwhm_separation) if self.constrain_fwhm else 0.0,
                fwhm + self.fwhm_separation if self.constrain_fwhm else None,
            ],
        }
        return fit_components

    def _choose_new_fit(
        self,
        fit_results: Dict,
        index: int,
        updated_fit_results: Optional[Dict] = None,
        interval: Optional[List] = None,
        n_centroids: Optional[int] = None,
    ) -> bool:
        """Decide whether to accept the new fit solution as the new best fit.

        :param fit_results: Dictionary containing the new best fit results after the refit attempt.
        :param index: Index ('index_fit' keyword) of the spectrum.
        :param updated_fit_results: Only used in phase 2 of the spatially coherent refitting, in case the best fit
        solution was already updated in a previous iteration.
        :param interval: List specifying the interval of spectral channels where 'n_centroids' number of centroid
        positions are required.
        :param n_centroids: Number of centroid positions that should be present in interval.
        :return: Decision of whether new fit solution gets accepted as new best fit.
        """

        current_best_fit = updated_fit_results or {
            key: self.decomposition[key][index]
            for key in [
                "N_blended",
                "N_neg_res_peak",
                "best_fit_rchi2",
                "best_fit_aicc",
                "pvalue",
                "fwhms_fit",
                "N_components",
                "means_fit",
            ]
        }

        # First we check how the values of flagged features changed after the refit

        flag_blended_old, flag_blended_new = (
            get_flags(
                n_old=current_best_fit["N_blended"],
                n_new=fit_results["N_blended"],
            )
            if self.flag_blended
            else (0, 0)
        )
        flag_neg_res_peak_old, flag_neg_res_peak_new = (
            get_flags(
                n_old=current_best_fit["N_neg_res_peak"],
                n_new=fit_results["N_neg_res_peak"],
            )
            if self.flag_neg_res_peak
            else (0, 0)
        )
        flag_rchi2_old, flag_rchi2_new = (
            get_flags_rchi2(
                rchi2_old=current_best_fit["best_fit_rchi2"],
                rchi2_new=fit_results["best_fit_rchi2"],
                rchi2_limit=self.rchi2_limit,
            )
            if self.flag_rchi2
            else (0, 0)
        )
        flag_residual_old, flag_residual_new = (
            get_flags_pvalue(
                pvalue_old=current_best_fit["pvalue"],
                pvalue_new=fit_results["pvalue"],
                min_pvalue=self.min_pvalue,
            )
            if self.flag_residual
            else (0, 0)
        )
        flag_broad_old, flag_broad_new = (
            get_flags_broad(
                fwhms_old=current_best_fit["fwhms_fit"],
                fwhms_new=fit_results["fwhms_fit"],
                contains_fwhm_flagged_as_broad=self.mask_broad_flagged[index],
                fwhm_factor=self.fwhm_factor,
                fwhm_separation=self.fwhm_separation,
            )
            if self.flag_broad
            else (0, 0)
        )

        # TODO: Improve the following, try to get rid of mask_indices
        indices, mask_indices = get_neighbors(
            location=self.location[index],
            shape=self.shape,
            return_mask=True,
        )
        ncomps = np.ones(8) * np.nan
        ncomps[mask_indices] = self.ncomps[indices]
        ncomps = np.insert(ncomps, 4, current_best_fit["N_components"])

        flag_ncomps_old, flag_ncomps_new = get_flags_ncomps(
            ndiff_old=abs(
                self.ncomps_wmedian[index] - current_best_fit["N_components"]
            ),
            ndiff_new=abs(self.ncomps_wmedian[index] - fit_results["N_components"]),
            njumps_old=self.ncomps_jumps[index],
            njumps_new=number_of_component_jumps(ncomps, self.max_jump_comps),
            max_diff_comps=self.max_diff_comps,
            n_max_jump_comps=self.n_max_jump_comps,
        )

        flag_centroids_old, flag_centroids_new = (
            get_flags_centroids(
                means_old=current_best_fit["means_fit"],
                means_new=fit_results["means_fit"],
                interval=interval,
                n_centroids=n_centroids,
            )
            if interval is not None
            else (0, 0)
        )

        #  only for phase 2: do not accept the new fit if there was no improvement for the centroid positions required in 'interval'
        if (n_centroids is not None) and (flag_centroids_new > 1):
            return False

        #  compute total flag values
        n_flags_old, n_flags_new = np.array(
            [
                (flag_blended_old, flag_blended_new),
                (flag_neg_res_peak_old, flag_neg_res_peak_new),
                (flag_broad_old, flag_broad_new),
                (flag_rchi2_old, flag_rchi2_new),
                (flag_residual_old, flag_residual_new),
                (flag_ncomps_old, flag_ncomps_new),
                (flag_centroids_old, flag_centroids_new),
            ]
        ).sum(axis=0)

        #  do not accept new fit if the total flag value increased
        if n_flags_new > n_flags_old:
            return False

        # if total flag value is the same or decreased there are two ways for the new best fit to get accepted as the
        # new best fit solution:
        # - accept the new fit if its AICc value is lower than AICc value of the current best fit solution
        # - if the AICc value of new fit is higher than the AICc value of the current best fit solution, only accept the
        # new fit if the values of the residual are normally distributed, i.e. if it passes the Kolmogorov-Smirnov test

        return (fit_results["best_fit_aicc"] <= current_best_fit["best_fit_aicc"]) or (
            fit_results["pvalue"] >= self.min_pvalue
        )

    def _get_values_for_indices(self, indices: np.ndarray, key: str) -> np.ndarray:
        # sum(tuple_of_lists, []) makes a flat list out of the tuple of lists
        return np.array(sum((self.decomposition[key][idx] for idx in indices), []))

    def _get_initial_values(
        self, indices_neighbors: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get sorted parameter values (amps, means, fwhms) of neighboring fit components for the grouping.

        :param indices_neighbors: Array containing the indices of all neighboring fit solutions that should be used
        for the grouping. in the form [idx1, ..., idxN].
        :return: tuple (amps, means, fwhms)
            - Array of amplitude values (sorted according to mean position).
            - Array of sorted mean position values.
            - Array of FWHM values (sorted according to mean position).
        """

        amps = self._get_values_for_indices(
            indices=indices_neighbors, key="amplitudes_fit"
        )
        means = self._get_values_for_indices(indices=indices_neighbors, key="means_fit")
        fwhms = self._get_values_for_indices(indices=indices_neighbors, key="fwhms_fit")
        sort_order = np.argsort(means)
        return amps[sort_order], means[sort_order], fwhms[sort_order]

    def _get_initial_values_from_neighbor(
        self, i: int, intensity_values: np.ndarray
    ) -> Dict:
        """Get dictionary with information about all fit components from neighboring fit solution.

        :param i: Index of neighboring fit solution.
        :param intensity_values: Intensity values of spectrum to refit.
        :return: Dictionary containing information about all fit components from neighboring fit solution.
        """

        fit_components = {}

        for key in range(self.decomposition["N_components"][i]):
            amp = self.decomposition["amplitudes_fit"][i][key]
            mean = self.decomposition["means_fit"][i][key]
            mean_err = self.decomposition["means_fit_err"][i][key]
            fwhm = self.decomposition["fwhms_fit"][i][key]

            mean_min = min(mean - self.mean_separation, mean - mean_err)
            mean_min = max(0, mean_min)  # prevent negative values
            mean_max = max(mean + self.mean_separation, mean + mean_err)

            fwhm_min = 0
            fwhm_max = None

            if self.constrain_fwhm:
                fwhm_min = max(0, fwhm - self.fwhm_separation)
                fwhm_max = fwhm + self.fwhm_separation

            keyname = str(key + 1)
            fit_components[keyname] = {
                "amp_ini": amp,
                "mean_ini": mean,
                "fwhm_ini": fwhm,
                "amp_bounds": [
                    0.0,
                    upper_limit_for_amplitude(
                        intensity_values, mean, fwhm, buffer_factor=1.1
                    ),
                ],
                "mean_bounds": [mean_min, mean_max],
                "fwhm_bounds": [fwhm_min, fwhm_max],
            }

        return fit_components

    # TODO: include settings_improve_fit as parameter and shift _gaussian_fitting to gausspy_py3.gp_plus?
    def _gaussian_fitting(
        self,
        spectrum: Spectrum,
        fit_components: Dict,
        params_only: bool = False,
    ) -> Dict:
        """Perform a new Gaussian decomposition with updated initial guesses.

        :param spectrum: Spectrum to refit.
        :param fit_components: Dictionary containing information about new initial guesses for fit components.
        :param params_only: If set to `True`, the returned dictionary of the fit results will only contain information
        about the amplitudes, FWHM values and mean positions of the fitted components.
        :return: Dictionary containing information about the fit results.
        """

        #  correct dictionary key
        settings_improve_fit = self.decomposition["improve_fit_settings"]
        settings_improve_fit.max_amp = settings_improve_fit.max_amp_factor * np.max(
            spectrum.intensity_values
        )

        #  set limits for fit parameters
        params, params_min, params_max = [], [], []
        for key in ["amp", "fwhm", "mean"]:
            for nr in fit_components.keys():
                params.append(fit_components[nr][f"{key}_ini"])
                params_min.append(fit_components[nr][f"{key}_bounds"][0])
                params_max.append(fit_components[nr][f"{key}_bounds"][1])

        #  get new best fit
        model = get_best_fit_model(
            model=Model(spectrum=spectrum),
            params_fit=params,
            settings_improve_fit=settings_improve_fit,
            params_min=params_min,
            params_max=params_max,
        )

        #  check for unfit residual peaks
        #  TODO: set fitted_residual_peaks to input offset positions??
        fitted_residual_peaks = []
        new_fit = True

        while new_fit:
            model, fitted_residual_peaks = check_for_peaks_in_residual(
                model=model,
                settings_improve_fit=settings_improve_fit,
                fitted_residual_peaks=fitted_residual_peaks,
            )
            new_fit = model.new_best_fit

        if model.n_components == 0:
            return

        fit_results = {
            "amplitudes_fit": model.amps,
            "fwhms_fit": model.fwhms,
            "means_fit": model.means,
            "amplitudes_fit_err": model.amps_uncertainties,
            "fwhms_fit_err": model.fwhms_uncertainties,
            "means_fit_err": model.means_uncertainties,
        }

        if params_only:
            return fit_results

        N_blended = get_fully_blended_gaussians(
            params_fit=model.parameters,
            get_count=True,
            separation_factor=self.decomposition[
                "improve_fit_settings"
            ].separation_factor,
        )
        N_neg_res_peak = check_for_negative_residual(
            model=model, settings_improve_fit=settings_improve_fit, get_count=True
        )

        return {
            **fit_results,
            **{
                "best_fit_rchi2": model.rchi2,
                "best_fit_aicc": model.aicc,
                "pvalue": model.pvalue,
                "N_components": model.n_components,
                "N_blended": N_blended,
                "N_neg_res_peak": N_neg_res_peak,
            },
        }

    def _save_final_results(self) -> None:
        """Save the results of the spatially coherent refitting iterations."""
        pathToFile = os.path.join(self.decomp_dirname, f"{self.fin_filename}.pickle")
        pickle.dump(self.decomposition, open(pathToFile, "wb"), protocol=2)
        say(
            f"'{self.fin_filename}' in '{self.decomp_dirname}'",
            task="save",
            logger=self.logger,
        )

    #
    #  --- Phase 2: Refitting towards coherence in centroid positions ---
    #

    def _get_n_centroid(self, n_centroids: np.ndarray, weights: np.ndarray) -> int:
        """Calculate expected value for number of centroids per grouped centroid interval."""
        choices = list(set(n_centroids))
        # first, check only immediate neighboring spectra
        mask_weight = weights >= (self.w_1 / np.sqrt(2))
        counts_choices = [
            0 if choice == 0 else np.count_nonzero(n_centroids[mask_weight] == choice)
            for choice in choices
        ]

        n_neighbors = np.count_nonzero(mask_weight)
        if np.max(counts_choices) > 0.5 * n_neighbors:
            return choices[np.argmax(counts_choices)]

        # include additional neighbors that are two pixels away
        weights_choices = [
            0 if choice == 0 else sum(weights[n_centroids == choice])
            for choice in choices
        ]
        return choices[np.argmax(weights_choices)]

    def _combine_directions(self, spatial_coherence_checks_per_direction: Dict) -> Dict:
        """Combine directions and build master dictionary."""
        indices_neighbors = np.concatenate(
            [
                spatial_coherence_checks_per_direction[direction]["indices_neighbors"]
                for direction in self.weights.keys()
            ]
        )
        weights = np.concatenate(
            [
                spatial_coherence_checks_per_direction[direction]["weights"]
                for direction in self.weights.keys()
            ]
        )
        intervals = merge_overlapping_intervals(
            [
                interval
                for direction in self.weights.keys()
                for interval in spatial_coherence_checks_per_direction[direction][
                    "means_interval"
                ]
            ]
        )
        means_interval = {
            str(key): [
                max(0, interval[0] - self.mean_separation / 2),
                interval[1] + self.mean_separation / 2,
            ]
            for key, interval in enumerate(intervals, start=1)
        }
        # TODO: The following lines also appear in _check_continuity_centroids; should this be refactored into its own
        #  method?
        means_of_neighbors = [
            self.decomposition["means_fit"][idx] for idx in indices_neighbors
        ]
        # estimate the expected number of centroids for interval
        ncomps_per_interval = {
            key: [
                sum(mean_min < mean < mean_max for mean in means) if bool(means) else 0
                for means in means_of_neighbors
            ]
            for key, (mean_min, mean_max) in means_interval.items()
        }
        # calculate number of centroids per centroid interval of neighbors
        n_centroids = {
            key: self._get_n_centroid(np.array(ncomps), weights)
            for key, ncomps in ncomps_per_interval.items()
        }
        return {
            "indices_neighbors": indices_neighbors,
            "weights": weights,
            "means_interval": means_interval,
            "n_comps": ncomps_per_interval,
            "n_centroids": n_centroids,
        }

    # TODO: Check if this can be a cached property? w_1 and w_2 change so this could be a problem for a cached property
    @property
    def weights(self):
        weights = dict.fromkeys(
            ["horizontal", "vertical"],
            {-2: self.w_2, -1: self.w_1, 1: self.w_1, 2: self.w_2},
        )
        weights.update(
            dict.fromkeys(
                ["diagonal_ul", "diagonal_ur"],
                {
                    -2: self.w_2 / np.sqrt(8),
                    -1: self.w_1 / np.sqrt(2),
                    1: self.w_1 / np.sqrt(2),
                    2: self.w_2 / np.sqrt(8),
                },
            )
        )
        return weights

    def _get_indices_and_weights_of_valid_neighbors(self, loc, idx, direction):
        indices_neighbors_and_center = get_neighbors(
            location=loc,
            exclude_location=False,
            shape=self.shape,
            n_neighbors=2,
            direction=direction,
        )
        is_neighbor = indices_neighbors_and_center != idx
        has_fit_components = np.array(
            [
                self.decomposition["N_components"][i]
                for i in indices_neighbors_and_center
            ]
        ).astype(bool)
        relative_position_to_center = np.arange(
            len(indices_neighbors_and_center)
        ) - np.flatnonzero(~is_neighbor)
        return (
            indices_neighbors_and_center[is_neighbor & has_fit_components],
            np.array(
                [
                    self.weights[direction][pos]
                    for pos in relative_position_to_center[
                        is_neighbor & has_fit_components
                    ]
                ]
            ),
        )

    def _check_continuity_centroids(self, idx: int, loc: Tuple) -> Dict:
        """Check for coherence of centroid positions of neighboring spectra.

        See Sect. 3.3.2. and Fig. 10 in Riener+ 2019 for more details.

        :param idx: Index ('index_fit' keyword) of the central spectrum.
        :param loc: Location (ypos, xpos) of the central spectrum.
        :return: Dictionary containing results of the spatial coherence check for neighboring centroid positions.
        """

        spatial_coherence_checks_per_direction = {}

        for direction in self.weights.keys():
            (
                indices_neighbors,
                weights_neighbors,
            ) = self._get_indices_and_weights_of_valid_neighbors(loc, idx, direction)

            if len(indices_neighbors) == 0:
                spatial_coherence_checks_per_direction[direction] = {
                    "indices_neighbors": indices_neighbors,
                    "weights": weights_neighbors,
                    "means_interval": [],
                }
                continue

            amps, means, fwhms = self._get_initial_values(indices_neighbors)
            grouping = group_fit_solutions(
                amps_tot=amps,
                means_tot=means,
                fwhms_tot=fwhms,
                split_fwhm=False,
                mean_separation=self.mean_separation,
                fwhm_separation=self.fwhm_separation,
            )

            means_interval = {
                key: [
                    max(0, min(value["means"]) - self.mean_separation / 2),
                    max(value["means"]) + self.mean_separation / 2,
                ]
                for key, value in grouping.items()
            }

            means_of_neighbors = [
                self.decomposition["means_fit"][idx] for idx in indices_neighbors
            ]
            ncomps_per_interval = {
                key: [
                    sum(mean_min < mean < mean_max for mean in means)
                    if bool(means)
                    else 0
                    for means in means_of_neighbors
                ]
                for key, (mean_min, mean_max) in means_interval.items()
            }

            # Calculate weight of required components per centroid interval.
            factor_required = {
                key: sum(np.array(val, dtype=bool) * weights_neighbors)
                for key, val in ncomps_per_interval.items()
            }

            # Keep only centroid intervals that have a certain minimum weight
            means_interval = [
                means_interval[key]
                for key in factor_required
                if factor_required[key] > self.min_p
            ]

            spatial_coherence_checks_per_direction[direction] = {
                "indices_neighbors": indices_neighbors,
                "weights": weights_neighbors,
                "means_interval": means_interval,
            }

        return self._combine_directions(spatial_coherence_checks_per_direction)

    def _check_for_required_components(
        self, idx: int, spatial_coherence_checks: Dict
    ) -> Dict:
        """Check the presence of the required centroid positions within the determined interval."""
        means = self.decomposition["means_fit"][idx]
        keys_for_refit = [
            key
            for key, interval in spatial_coherence_checks["means_interval"].items()
            if sum(interval[0] < x < interval[1] for x in means)
            != spatial_coherence_checks["n_centroids"][key]
        ]
        return {
            "indices_neighbors": spatial_coherence_checks["indices_neighbors"],
            "weights": spatial_coherence_checks["weights"],
            "means_interval": {
                str(i): spatial_coherence_checks["means_interval"][key]
                for i, key in enumerate(keys_for_refit, start=1)
            },
            "n_centroids": {
                str(i): spatial_coherence_checks["n_centroids"][key]
                for i, key in enumerate(keys_for_refit, start=1)
            },
        }

    def _select_neighbors_to_use_for_refit(
        self, indices: np.ndarray, means_interval: Dict, n_centroids: Dict
    ) -> Dict:
        """Select neighboring fit solutions with right number of centroid positions as refit solutions."""
        return {
            key: [
                idx
                for idx in indices
                if n_centroids[key]
                == sum(
                    interval[0] < x < interval[1]
                    for x in self.decomposition["means_fit"][idx]
                )
            ]
            for key, interval in means_interval.items()
        }

    def _determine_all_neighbors(self) -> None:
        """Determine the indices of all valid neighbors."""
        say("\ndetermine neighbors for all spectra...", logger=self.logger)

        mask_all = np.array(
            [0 if x is None else 1 for x in self.decomposition["N_components"]]
        ).astype("bool")
        self.indices_all = np.array(self.decomposition["index_fit"])[mask_all]
        if self.pixel_range is not None:
            self.indices_all = np.array(self.decomposition["index_fit"])[~self.nan_mask]
        self.locations_all = np.take(np.array(self.location), self.indices_all, axis=0)

        for i, loc in tqdm(zip(self.indices_all, self.locations_all)):
            indices_neighbors_total = np.array([])
            for direction in self.weights.keys():
                indices_neighbors = get_neighbors(
                    location=loc,
                    exclude_location=True,
                    shape=self.shape,
                    n_neighbors=2,
                    direction=direction,
                )
                indices_neighbors_total = np.append(
                    indices_neighbors_total, indices_neighbors
                )
            indices_neighbors_total = indices_neighbors_total.astype("int")
            self.neighbor_indices_all[i] = indices_neighbors_total

    def _check_indices_refit(self) -> None:
        """Check which spectra show incoherence in their fitted centroid positions and require refitting."""
        say("\ncheck which spectra require refitting...", logger=self.logger)
        if self.refitting_iteration == 1:
            self._determine_all_neighbors()

        if np.count_nonzero(self.mask_refitted) == len(self.mask_refitted):
            self.indices_refit = self.indices_all.copy()
            self.locations_refit = self.locations_all.copy()
            return

        self.indices_refit = np.array(
            [
                i
                for i in self.indices_all
                if np.count_nonzero(self.mask_refitted[self.neighbor_indices_all[i]])
            ],
            dtype="int",
        )
        self.locations_refit = np.take(
            np.array(self.location), self.indices_refit, axis=0
        )

    def _check_continuity(self) -> None:
        """Check continuity of centroid positions.

        See Fig. 9 and Sect. 3.3.2. in Riener+ 2019 for more details.

        """
        self.refitting_iteration += 1
        say(
            f"\nthreshold for required components: {self.min_p:.3f}", logger=self.logger
        )
        self.determine_spectra_for_flagging()
        self._check_indices_refit()
        self._refitting()

    def refit_spectrum_phase_2(self, index: int, i: int) -> List:
        """Parallelized function for refitting spectra whose fitted centroid position values show spatial incoherence.

        :param index: Index ('index_fit' keyword) of the spectrum that will be refit.
        :param i: List index of the entry in the list that is handed over to the multiprocessing routine
        :return: A list in the form of [index, best_fit_results, indices_neighbors, is_successful_refit] in case of
        a successful refit; otherwise [index, 'None', indices_neighbors, is_successful_refit] is returned.
        """

        is_successful_refit = False
        fit_results, best_fit_results = None, None
        #  TODO: check if this is correct:
        indices_neighbors = []
        spatial_coherence_refit_requirements = self._check_for_required_components(
            idx=index,
            spatial_coherence_checks=self._check_continuity_centroids(
                idx=index, loc=self.locations_refit[i]
            ),
        )

        if self._finalize:
            return [
                index,
                spatial_coherence_refit_requirements["means_interval"],
                spatial_coherence_refit_requirements["n_centroids"],
            ]

        if len(spatial_coherence_refit_requirements["means_interval"].keys()) == 0:
            return [index, None, indices_neighbors, is_successful_refit]

        spatial_coherence_refit_requirements[
            "indices_refit"
        ] = self._select_neighbors_to_use_for_refit(
            # TODO: Check if spatial_coherence_refit_requirements['weights'] >= self.w_min condition was already checked earlier
            indices=spatial_coherence_refit_requirements["indices_neighbors"][
                spatial_coherence_refit_requirements["weights"]
                >= (self.w_1 / np.sqrt(2))
            ],
            means_interval=spatial_coherence_refit_requirements["means_interval"],
            n_centroids=spatial_coherence_refit_requirements["n_centroids"],
        )

        for key, indices_neighbors in spatial_coherence_refit_requirements[
            "indices_refit"
        ].items():
            fit_results, refit = self._try_refit_with_individual_neighbors(
                index=index,
                spectrum=Spectrum(
                    intensity_values=self.data[index],
                    channels=self.channels,
                    rms_noise=self.errors[index][0],
                    signal_intervals=self.signal_intervals[index],
                    noise_spike_intervals=self.noise_spike_intervals[index],
                ),
                indices_neighbors=indices_neighbors,
                interval=spatial_coherence_refit_requirements["means_interval"][key],
                n_centroids=spatial_coherence_refit_requirements["n_centroids"][key],
                updated_fit_results=fit_results,
            )

            if fit_results is not None:
                best_fit_results = fit_results

        return [index, best_fit_results, indices_neighbors, is_successful_refit]
