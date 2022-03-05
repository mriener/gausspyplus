import collections
import functools
import itertools
import os
import pickle
import textwrap
from pathlib import Path
from typing import Dict, Optional, List, Tuple, Union

import numpy as np
import scipy.ndimage as ndimage

from networkx.algorithms.components.connected import connected_components
from tqdm import tqdm

from gausspyplus.config_file import get_values_from_config_file
from gausspyplus.gausspy_py3.gp_plus import get_fully_blended_gaussians, check_for_peaks_in_residual, get_best_fit, check_for_negative_residual, remove_components_from_sublists
from gausspyplus.utils.determine_intervals import merge_overlapping_intervals
from gausspyplus.utils.gaussian_functions import combined_gaussian, split_params, CONVERSION_STD_TO_FWHM
from gausspyplus.utils.grouping_functions import to_graph, get_neighbors
from gausspyplus.utils.noise_estimation import mask_channels
from gausspyplus.utils.output import set_up_logger, say, make_pretty_header


class SpatialFitting(object):
    def __init__(self,
                 path_to_pickle_file: Optional[Union[str, Path]] = None,
                 path_to_decomp_file: Optional[Union[str, Path]] = None,
                 fin_filename: Optional[Union[str, Path]] = None,
                 config_file: Union[str, Path] = ''):
        """Class implementing the two phases of spatially coherent refitting discussed in Riener+ 2019.

        Parameters
        ----------
        path_to_pickle_file :
        path_to_decomp_file :
        fin_filename :
        config_file :

        """
        self.path_to_pickle_file = path_to_pickle_file
        self.path_to_decomp_file = path_to_decomp_file
        self.dirpath_gpy = None
        self.fin_filename = fin_filename

        self.exclude_flagged = False
        self.max_fwhm = None
        self.rchi2_limit = None
        self.rchi2_limit_refit = None
        self.max_diff_comps = 1
        self.max_jump_comps = 2
        self.n_max_jump_comps = 1
        self.max_refitting_iteration = 30

        self.use_all_neighors = False
        self.flag_blended = None
        self.flag_neg_res_peak = None
        self.flag_rchi2 = None
        self.flag_residual = None
        self.flag_broad = None
        self.flag_ncomps = None
        self.refit_blended = False
        self.refit_neg_res_peak = False
        self.refit_rchi2 = False
        self.refit_residual = False
        self.refit_broad = False
        self.refit_ncomps = False

        self.mean_separation = 2.  # minimum distance between peaks in channels
        self.fwhm_separation = 4.
        self.constrain_fwhm = False
        self.snr = 3.
        self.fwhm_factor = 2.
        self.fwhm_factor_refit = None
        self.broad_neighbor_fraction = 0.5
        self.min_weight = 0.5
        self.weight_factor = 2
        self.min_pvalue = 0.01
        self.use_ncpus = None
        self.verbose = True
        self.suffix = ''
        self.log_output = True
        self.only_print_flags = False
        self.pixel_range = None
        self._w_start = 1.
        self._finalize = False

        if config_file:
            get_values_from_config_file(
                self, config_file, config_key='spatial fitting')

    def _check_settings(self) -> None:
        """Check user settings and raise error messages or apply corrections."""
        if self.path_to_pickle_file is None:
            raise Exception("Need to specify 'path_to_pickle_file'")
        if self.path_to_decomp_file is None:
            raise Exception("Need to specify 'path_to_decomp_file'")
        self.decomp_dirname = os.path.dirname(self.path_to_decomp_file)
        self.file = os.path.basename(self.path_to_decomp_file)
        self.filename, self.file_extension = os.path.splitext(self.file)

        if self.fin_filename is None:
            suffix = '_sf-p1'
            self.fin_filename = self.filename + suffix + self.suffix
            if self.phase_two:
                suffix = '_sf-p2'
                if self.filename.endswith('_sf-p1'):
                    self.fin_filename = self.filename.replace('_sf-p1', '_sf-p2') + self.suffix
                else:
                    self.fin_filename = self.filename + suffix + self.suffix

        if self.dirpath_gpy is None:
            self.dirpath_gpy = os.path.dirname(self.decomp_dirname)

        if self.rchi2_limit_refit is None:
            self.rchi2_limit_refit = self.rchi2_limit
        if self.fwhm_factor_refit is None:
            self.fwhm_factor_refit = self.fwhm_factor

        if all(refit is False for refit in [self.refit_blended,
                                            self.refit_neg_res_peak,
                                            self.refit_rchi2,
                                            self.refit_residual,
                                            self.refit_broad,
                                            self.refit_ncomps]):
            raise Exception("Need to set at least one 'refit_*' parameter to 'True'")

        if self.flag_blended is None:
            self.flag_blended = self.refit_blended
        if self.flag_neg_res_peak is None:
            self.flag_neg_res_peak = self.refit_neg_res_peak
        if self.flag_rchi2 is None:
            self.flag_rchi2 = self.refit_rchi2
        if self.flag_rchi2 and (self.rchi2_limit is None):
            raise Exception("Need to set 'rchi2_limit' if 'flag_rchi2=True' or 'refit_rchi2=True'")
        if self.flag_residual is None:
            self.flag_residual = self.refit_residual
        if self.flag_broad is None:
            self.flag_broad = self.refit_broad
        if self.flag_ncomps is None:
            self.flag_ncomps = self.refit_ncomps

    def _initialize(self) -> None:
        """Read in data files and initialize parameters."""
        with open(self.path_to_pickle_file, "rb") as pickle_file:
            pickledData = pickle.load(pickle_file, encoding='latin1')

        self.indexList = pickledData['index']
        self.data = pickledData['data_list']
        self.errors = pickledData['error']
        if 'header' in pickledData.keys():
            self.header = pickledData['header']
            self.shape = (self.header['NAXIS2'], self.header['NAXIS1'])
            self.length = self.header['NAXIS2'] * self.header['NAXIS1']
            self.location = pickledData['location']
            self.n_channels = self.header['NAXIS3']
        else:
            self.length = len(self.data)
            self.n_channels = len(self.data[0])
        self.channels = np.arange(self.n_channels)
        if self.max_fwhm is None:
            self.max_fwhm = int(self.n_channels / 3)

        self.signalRanges = pickledData['signal_ranges']
        self.noiseSpikeRanges = pickledData['noise_spike_ranges']

        with open(self.path_to_decomp_file, "rb") as pickle_file:
            self.decomposition = pickle.load(pickle_file, encoding='latin1')

        self.nIndices = len(self.decomposition['index_fit'])

        self.decomposition['refit_iteration'] = [0] * self.nIndices
        self.decomposition['gaussians_rchi2'] = [None] * self.nIndices
        self.decomposition['gaussians_aicc'] = [None] * self.nIndices

        self.neighbor_indices = np.array([None]*self.nIndices)
        self.neighbor_indices_all = np.array([None]*self.nIndices)

        self.nanMask = np.isnan([np.nan if i is None else i for i in self.decomposition['N_components']])

        if self.pixel_range is not None:
            self._mask_out_beyond_pixel_range()

        self.nanIndices = np.array(self.decomposition['index_fit'])[self.nanMask]

        self.signal_mask = [None for _ in range(self.nIndices)]
        for i, (noiseSpikeRanges, signalRanges) in enumerate(zip(self.noiseSpikeRanges, self.signalRanges)):
            if signalRanges is not None:
                self.signal_mask[i] = mask_channels(
                    n_channels=self.n_channels,
                    ranges=signalRanges,
                    remove_intervals=noiseSpikeRanges
                )

        #  starting condition so that refitting iteration can start
        # self.mask_refitted = np.ones(1)
        self.mask_refitted = np.array([1]*self.nIndices)
        self.list_n_refit = []
        self.refitting_iteration = 0

        normalization_factor = 1 / (2 * (self.weight_factor + 1))
        self.w_2 = normalization_factor
        self.w_1 = self.weight_factor * normalization_factor
        self.min_p = self._w_start - self.w_2

    def _mask_out_beyond_pixel_range(self) -> None:
        locations = list(itertools.product(
            range(self.pixel_range['y'][0], self.pixel_range['y'][1]),
            range(self.pixel_range['x'][0], self.pixel_range['x'][1])))

        for idx, loc in enumerate(self.location):
            if loc not in locations:
                self.nanMask[idx] = np.nan

    def _info_text(self, refit=False):
        text_phase_1 = '' if not refit else textwrap.dedent(f"""
            For phase 1:
            Exclude flagged spectra as possible refit solutions in first refit attempts: {self.exclude_flagged}
            Use also flagged spectra as refit solutions in case no new best fit could be obtained from unflagged spectra: {self.use_all_neighors}""")
        return text_phase_1 + textwrap.dedent(f"""
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
             - Differing number of components: {(self.flag_ncomps, self.refit_ncomps)[refit]}""")

    def _getting_ready(self) -> None:
        """Set up logger and write initial output to terminal."""
        if self.log_output:
            self.logger = set_up_logger(self.dirpath_gpy, self.filename, method='g+_spatial_refitting')
        else:
            self.logger = False
        say(message=make_pretty_header(f'Spatial refitting - Phase {1 + self.phase_two}'), logger=self.logger)
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

        Parameters
        ----------
        continuity : Set to 'True' for phase 2 of the spatially coherent refitting (coherence of centroid positions).

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

    def _define_mask(self, key: str, limit: Union[int, float], flag: bool) -> np.ndarray:
        """Create boolean mask with data values exceeding the defined limits set to 'True'.

        This mask is only created if 'flag=True'.

        Parameters
        ----------
        key : Dictionary key of the parameter: 'N_blended', 'N_neg_res_peak', or 'best_fit_rchi2'.
        limit : Upper limit of the corresponding value.
        flag : User-defined flag for the corresponding dictionary parameter.

        Returns
        -------
        mask : Boolean mask with values exceeding 'limit' set to 'True'.

        """
        if not flag:
            return np.zeros(self.length).astype('bool')

        array = np.array(self.decomposition[key])
        array[self.nanMask] = 0
        return array > limit

    def _define_mask_residual(self, key: str, limit: Union[int, float], flag: bool) -> np.ndarray:
        if not flag:
            return np.zeros(self.length).astype('bool')

        array = np.array(self.decomposition[key])
        array[self.nanMask] = limit
        return array < limit

    def _define_mask_broad_limit(self, flag: bool) -> Union[np.ndarray, int]:
        """Create boolean mask identifying the location of broad fit components.

        The mask is 'True' at the location of spectra that contain fit components exceeding the 'max_fwhm' value.

        This mask is only created if 'flag=True'.

        Parameters
        ----------
        flag : User-defined 'flag_broad' parameter.

        Returns
        -------
        mask : numpy.ndarray

        """
        # TODO: n_broad seems to be incorrect -> shouldn't that be a count?
        n_broad = np.zeros(self.length)

        if not flag:
            return n_broad.astype('bool'), n_broad

        for i, fwhms in enumerate(self.decomposition['fwhms_fit']):
            if fwhms is None:
                continue
            n_broad[i] = np.count_nonzero(np.array(fwhms) > self.max_fwhm)
        n_broad[self.nanMask] = 0
        mask = n_broad > 0
        return mask, n_broad

    def _broad_components(self, values: np.ndarray) -> Union[float, int]:
        """Check for the presence of broad fit components.

        This check is performed by comparing the broadest fit components of a spectrum with its 8 immediate neighbors.

        A fit component is defined as broad if its FWHM value exceeds the FWHM value of the largest fit components of
        more than 'self.broad_neighbor_fraction' of its neighbors by at least a factor of 'self.fwhm_factor'.

        In addition we impose that the minimum difference between the compared FWHM values has to exceed
        'self.fwhm_separation' to avoid flagging narrow components.

        Parameters
        ----------
        values : Array of FWHM values of the broadest fit components for a spectrum and its 8 immediate neighbors.

        Returns
        -------
        FWHM value in case of a broad fit component, 0 otherwise.

        """
        central_value = values[4]
        #  Skip if central spectrum was masked out.
        if np.isnan(central_value):
            return 0
        values = np.delete(values, 4)
        #  Remove all neighbors that are NaN.
        values = values[~np.isnan(values)]
        #  Skip if there are no valid available neighbors.
        if values.size == 0:
            return 0
        #  Compare the largest FWHM value of the central spectrum with the largest FWHM values of its neighbors.
        counter = 0
        for value in values:
            if np.isnan(value):
                continue
            if central_value > value * self.fwhm_factor and (central_value - value) > self.fwhm_separation:
                counter += 1
        if counter > values.size * self.broad_neighbor_fraction:
            return central_value
        return 0

    def _define_mask_broad(self, flag: bool) -> np.ndarray:
        """Create a boolean mask indicating the location of broad fit components.

        If 'self.flag_broad=False' no locations are masked.

        Parameters
        ----------
        flag : User-defined 'self.flag_broad' parameter.

        Returns
        -------
        mask_broad : numpy.ndarray

        """
        if not flag:
            return np.zeros(self.length).astype('bool')

        broad_1d = np.empty(self.length)
        broad_1d.fill(np.nan)
        mask_broad = np.zeros(self.length)

        #  check if the fit component with the largest FWHM value of a spectrum satisfies the criteria to be flagged as
        #  a broad component by comparing it to the remaining components of the spectrum.
        for i, fwhms in enumerate(self.decomposition['fwhms_fit']):
            if fwhms is None:
                continue
            if len(fwhms) == 0:
                continue
            #  in case there is only one fit parameter there are no other components to compare; we need to compare it
            #  with the components of the immediate neighbors
            broad_1d[i] = max(fwhms)
            if len(fwhms) == 1:
                continue
            #  in case of multiple fit parameters select the one with the largest FWHM value and check whether it
            #  exceeds the second largest FWHM value in that spectrum by a factor of 'self.fwhm_factor'; also check if
            #  the absolute difference of their values exceeds 'self.fwhm_separation' to avoid narrow components.
            fwhms = sorted(fwhms)
            if (fwhms[-1] > self.fwhm_factor * fwhms[-2]) and (fwhms[-1] - fwhms[-2]) > self.fwhm_separation:
                mask_broad[i] = 1

        #  check if the fit component with the largest FWHM value of a spectrum satisfies the criteria to be flagged as
        #  a broad component by comparing it to the largest FWHM values of its 8 immediate neighbors.

        broad_2d = broad_1d.astype('float').reshape(self.shape)

        footprint = np.ones((3, 3))

        broad_fwhm_values = ndimage.generic_filter(
            input=broad_2d,
            function=self._broad_components,
            footprint=footprint,
            mode='constant',
            cval=np.nan
        ).flatten()
        # mask_broad = mask_broad.astype('bool')
        mask_broad += broad_fwhm_values  #.astype('bool')
        mask_broad[self.nanMask] = 0
        mask_broad = mask_broad.astype('bool')

        return mask_broad

    def _weighted_median(self, data):
        # TODO: add type hints
        """Adapted from: https://gist.github.com/tinybike/d9ff1dad515b66cc0d87"""
        w_1 = 1
        w_2 = w_1 / np.sqrt(2)
        weights = np.array([w_2, w_1, w_2, w_1, w_1, w_2, w_1, w_2])
        central_value = data[4]
        #  Skip if central spectrum was masked out.
        if np.isnan(central_value):
            return 0
        data = np.delete(data, 4)
        #  Remove all neighbors that are NaN.
        mask = ~np.isnan(data)
        data = data[mask]
        weights = weights[mask]
        #  Skip if there are no valid available neighbors.
        if data.size == 0:
            return 0

        s_data, s_weights = map(np.array, zip(*sorted(zip(data, weights))))
        midpoint = 0.5 * sum(s_weights)
        if any(weights > midpoint):
            return (data[weights == np.max(weights)])[0]
        cs_weights = np.cumsum(s_weights)
        idx = np.where(cs_weights <= midpoint)[0][-1]
        return np.mean(s_data[idx:idx+2]) if cs_weights[idx] == midpoint else s_data[idx + 1]

    def _number_of_component_jumps(self, values: np.ndarray) -> int:
        """Determine the number of component jumps towards neighboring fits.

        A component jump occurs if the number of components is different by more than 'self.max_jump_comps' components.

        Parameters
        ----------
        values : Array of the number of fit components for a spectrum and its 8 immediate neighbors.

        Returns
        -------
        Number of component jumps.

        """
        central_value = values[4]
        if np.isnan(central_value):
            return 0
        values = np.delete(values, 4)
        counter = 0
        for value in values:
            if np.isnan(value):
                continue
            if np.abs(central_value - value) > self.max_jump_comps:
                counter += 1
        return counter

    def _define_mask_neighbor_ncomps(self, flag: bool) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """Create a boolean mask indicating the location of component jumps.

        Parameters
        ----------
        flag : User-defined 'self.flag_ncomps' parameter.

        Returns
        -------
        mask_neighbor : Boolean mask indicating the location of component jump.
        ncomps_jumps : Array containing the information of how many component jumps occur at which location.
        ncomps_1d : Array containing the information about the number of fitted components per location.

        """
        if not flag:
            return np.zeros(self.length).astype('bool'), None, None

        nanmask_1d = self.nanMask
        nanmask_2d = nanmask_1d.reshape(self.shape)
        ncomps_1d = np.empty(self.length)
        ncomps_1d.fill(np.nan)
        ncomps_1d[~self.nanMask] = np.array(self.decomposition['N_components'])[~self.nanMask]
        ncomps_2d = ncomps_1d.astype('float').reshape(self.shape)
        ncomps_2d[nanmask_2d] = np.nan

        mask_neighbor = np.zeros(self.length)
        footprint = np.ones((3, 3))

        ncomps_wmedian = ndimage.generic_filter(
            input=ncomps_2d,
            function=self._weighted_median,
            footprint=footprint,
            mode='constant',
            cval=np.nan
        ).flatten()
        mask_neighbor[~self.nanMask] = ncomps_wmedian[~self.nanMask] > self.max_diff_comps

        ncomps_jumps = ndimage.generic_filter(
            input=ncomps_2d,
            function=self._number_of_component_jumps,
            footprint=footprint,
            mode='reflect',
            cval=np.nan
        ).flatten()
        mask_neighbor[~self.nanMask] = ncomps_jumps[~self.nanMask] > self.n_max_jump_comps

        mask_neighbor = mask_neighbor.astype('bool')

        # TODO: check the return type: above 3 params are returned but in the following 4??
        return mask_neighbor, ncomps_wmedian, ncomps_jumps, ncomps_1d

    def determine_spectra_for_flagging(self) -> None:
        """Flag spectra not satisfying user-defined flagging criteria."""
        self.mask_blended = self._define_mask('N_blended', 0, self.flag_blended)
        self.mask_neg_res_peak = self._define_mask('N_neg_res_peak', 0, self.flag_neg_res_peak)
        self.mask_rchi2_flagged = self._define_mask('best_fit_rchi2', self.rchi2_limit, self.flag_rchi2)
        self.mask_residual = self._define_mask_residual('pvalue', self.min_pvalue, self.flag_residual)
        self.mask_broad_flagged = self._define_mask_broad(self.flag_broad)
        self.mask_broad_limit, self.n_broad = self._define_mask_broad_limit(self.flag_broad)
        self.mask_ncomps, self.ncomps_wmedian, self.ncomps_jumps, self.ncomps =\
            self._define_mask_neighbor_ncomps(self.flag_ncomps)

        if self._finalize:
            return

        mask_flagged = (self.mask_blended
                        + self.mask_neg_res_peak
                        + self.mask_broad_flagged
                        + self.mask_rchi2_flagged
                        + self.mask_residual
                        + self.mask_ncomps)

        self.count_flags = (self.mask_blended.astype('int')
                            + self.mask_neg_res_peak.astype('int')
                            + self.mask_broad_flagged.astype('int')
                            + self.mask_rchi2_flagged.astype('int')
                            + self.mask_residual.astype('int')
                            + self.mask_ncomps.astype('int'))

        self.indices_flagged = np.array(self.decomposition['index_fit'])[mask_flagged]

        if self.phase_two:
            text = textwrap.dedent(f"""
                Flags:
                - {np.count_nonzero(self.mask_blended)} spectra w/ blended components
                - {np.count_nonzero(self.mask_neg_res_peak)} spectra w/ negative residual feature
                - {np.count_nonzero(self.mask_broad_flagged)} spectra w/ broad feature
                \t (info: {np.count_nonzero(self.mask_broad_limit)} spectra w/ a FWHM > {int(self.max_fwhm)} channels)
                - {np.count_nonzero(self.mask_rchi2_flagged)} spectra w/ high rchi2 value
                - {np.count_nonzero(self.mask_residual)} spectra w/ residual not passing normality test
                - {np.count_nonzero(self.mask_ncomps)} spectra w/ differing number of components""")
            say(text, logger=self.logger)

    def _define_mask_refit(self) -> None:
        """Select spectra to refit in phase 1 of the spatially coherent refitting."""
        mask_refit = np.zeros(self.length).astype('bool')
        if self.refit_blended:
            mask_refit += self.mask_blended
        if self.refit_neg_res_peak:
            mask_refit += self.mask_neg_res_peak
        if self.refit_broad:
            mask_refit += self.mask_broad_refit
        if self.refit_rchi2:
            mask_refit += self.mask_rchi2_refit
        if self.refit_residual:
            mask_refit += self.mask_residual
        if self.refit_ncomps:
            mask_refit += self.mask_ncomps

        self.indices_refit = np.array(self.decomposition['index_fit'])[mask_refit]
        # self.indices_refit = self.indices_refit[886:888]  # for debugging
        self.locations_refit = np.take(np.array(self.location), self.indices_refit, axis=0)

    def _get_n_refit(self, flag: bool, n_refit: int) -> int:
        return n_refit if flag else 0

    def _determine_spectra_for_refitting(self) -> None:
        """Determine spectra for refitting in phase 1 of the spatially coherent refitting."""
        say('\ndetermine spectra that need refitting...', logger=self.logger)

        #  flag spectra based on user-defined criteria
        self.determine_spectra_for_flagging()

        #  determine new masks for spectra that do not satisfy the user-defined criteria for broad components and
        #  reduced chi-square values; this is done because users can opt to use different values for flagging and
        #  refitting for these two criteria
        self.mask_broad_refit = self._define_mask_broad(flag=self.refit_broad)
        self.mask_rchi2_refit = self._define_mask(
            key='best_fit_rchi2',
            limit=self.rchi2_limit_refit,
            flag=self.refit_rchi2
        )

        #  select spectra for refitting based on user-defined criteria
        self._define_mask_refit()

        #  print the results of the flagging/refitting selections to the terminal

        n_spectra = sum(x is not None for x in self.decomposition['N_components'])
        n_indices_refit = len(self.indices_refit)
        n_flagged_blended = np.count_nonzero(self.mask_blended)
        n_flagged_neg_res_peak = np.count_nonzero(self.mask_neg_res_peak)
        n_flagged_broad = np.count_nonzero(self.mask_broad_flagged)
        n_flagged_rchi2 = np.count_nonzero(self.mask_rchi2_flagged)
        n_flagged_residual = np.count_nonzero(self.mask_residual)
        n_flagged_ncomps = np.count_nonzero(self.mask_ncomps)

        n_refit_blended = self._get_n_refit(self.refit_blended, n_flagged_blended)
        n_refit_neg_res_peak = self._get_n_refit(self.refit_neg_res_peak, n_flagged_neg_res_peak)
        n_refit_broad = self._get_n_refit(self.refit_broad, np.count_nonzero(self.mask_broad_refit))
        n_refit_rchi2 = self._get_n_refit(self.refit_rchi2, np.count_nonzero(self.mask_rchi2_refit))
        n_refit_residual = self._get_n_refit(self.refit_residual, np.count_nonzero(self.mask_residual))
        n_refit_ncomps = self._get_n_refit(self.refit_ncomps, n_flagged_ncomps)

        try:
            n_fraction_refit = n_indices_refit / n_spectra
        except ZeroDivisionError:
            n_fraction_refit = 0

        n_refit_list = [
            n_refit_blended,
            n_refit_neg_res_peak,
            n_refit_broad,
            n_refit_rchi2,
            n_refit_residual,
            n_refit_ncomps
        ]

        text = textwrap.dedent(f"""
            {n_indices_refit} out of {n_spectra} spectra ({n_fraction_refit:.2%}) selected for refitting:
             - {n_refit_blended} spectra w/ blended components ({n_flagged_blended} flagged)
             - {n_refit_neg_res_peak} spectra w/ negative residual feature ({n_flagged_neg_res_peak} flagged)
             - {n_refit_broad} spectra w/ broad feature ({n_flagged_broad} flagged)
             \t (info: {np.count_nonzero(self.mask_broad_limit)} spectra w/ a FWHM > {int(self.max_fwhm)} channels)
             - {n_refit_rchi2} spectra w/ high rchi2 value ({n_flagged_rchi2} flagged)
             - {n_refit_residual} spectra w/ residual not passing normality test ({n_flagged_residual} flagged)
             - {n_refit_ncomps} spectra w/ differing number of components ({n_flagged_ncomps} flagged)""")
        say(text, logger=self.logger)

        #  check if the stopping criterion is fulfilled

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
            return all(n_refit_list[i] >= min(n[i] for n in self.list_n_refit) for i in range(len(n_refit_list)))

    def _refitting(self) -> None:
        """Refit spectra with multiprocessing routine."""
        say('\nstart refit iteration #{}...'.format(
            self.refitting_iteration), logger=self.logger)

        #  initialize the multiprocessing routine

        import gausspyplus.parallel_processing
        gausspyplus.parallel_processing.init([self.indices_refit, [self]])

        #  try to refit spectra via the multiprocessing routine

        if self.phase_two:
            results_list = gausspyplus.parallel_processing.func(use_ncpus=self.use_ncpus, function='refit_phase_2')
        else:
            results_list = gausspyplus.parallel_processing.func(use_ncpus=self.use_ncpus, function='refit_phase_1')
        print('SUCCESS')

        if self._finalize:
            return results_list

        #  reset the mask for spectra selected for refitting
        self.mask_refitted = np.array([0]*self.nIndices)

        keys = ['amplitudes_fit', 'fwhms_fit', 'means_fit',
                'amplitudes_fit_err', 'fwhms_fit_err', 'means_fit_err',
                'best_fit_rchi2', 'best_fit_aicc', 'N_components',
                'gaussians_rchi2', 'gaussians_aicc', 'pvalue',
                'N_neg_res_peak', 'N_blended']

        count_selected, count_refitted = 0, 0

        #  collect results of the multiprocessing routine

        for i, item in enumerate(results_list):
            if not isinstance(item, list):
                say(f"Error for spectrum with index {i}: {item}", logger=self.logger)
                continue

            index, result, indices_neighbors, refit = item
            if refit:
                count_selected += 1
            self.neighbor_indices[index] = indices_neighbors
            if result is not None:
                count_refitted += 1
                self.decomposition['refit_iteration'][index] += 1
                self.mask_refitted[index] = 1
                for key in keys:
                    self.decomposition[key][index] = result[key]

        #  print statistics of the refitting iteration to the terminal

        refit_percent = 0 if count_selected == 0 else count_refitted / count_selected
        text = textwrap.dedent(f"""
            Results of the refit iteration:
            Tried to refit {count_selected} spectra
            Successfully refitted {count_refitted} spectra ({refit_percent:.2%})\n
            ***""")
        say(text, logger=self.logger)

        #  check if one of the stopping criteria is fulfilled

        if self.phase_two:
            if self._stopping_criterion([count_refitted]):
                self.min_p -= self.w_2
                self.list_n_refit = [[self.length]]
                self.mask_refitted = np.array([1]*self.nIndices)
            else:
                self.list_n_refit.append([count_refitted])

            if self.min_p < self.min_weight:
                self._save_final_results()
            else:
                self._check_continuity()
        else:
            self._determine_spectra_for_refitting()

    def _determine_neighbor_indices(self, neighbors: List, include_flagged: bool = False) -> Tuple[np.ndarray, bool]:
        """Determine indices of all valid neighboring pixels.

        Parameters
        ----------
        neighbors : List containing information about the location of N neighboring spectra in the form
            [(y1, x1), ..., (yN, xN)].

        Returns
        -------
        indices_neighbors : Array containing the corresponding indices of the N neighboring spectra in the form
            [idx1, ..., idxN].

        """
        all_neighbors = False
        indices_neighbors_all = np.array([])
        for neighbor in neighbors:
            indices_neighbors_all = np.append(
                indices_neighbors_all, np.ravel_multi_index(neighbor, self.shape)).astype('int')
        indices_neighbors = indices_neighbors_all.copy()

        #  check if neighboring pixels were selected for refitting, are masked out, or contain no fits and thus cannot be used

        #  whether to exclude all flagged neighboring spectra as well that
        #  were not selected for refitting
        if self.exclude_flagged:
            indices_bad = self.indices_flagged
        else:
            indices_bad = self.indices_refit

        for idx in indices_neighbors:
            if (idx in indices_bad) or (idx in self.nanIndices) or (self.decomposition['N_components'][idx] == 0):
                indices_neighbors = np.delete(indices_neighbors, np.where(indices_neighbors == idx))

        if indices_neighbors.size > 1:
            #  sort neighboring fit solutions according to lowest value of reduced chi-square
            #  TODO: change this so that this gets sorted according to the lowest difference of the reduced chi-square
            #   values to the ideal value of 1 to prevent using fit solutions that 'overfit' the data
            sort = np.argsort(np.array(self.decomposition['best_fit_rchi2'])[indices_neighbors])
            indices_neighbors = indices_neighbors[sort]

        elif (indices_neighbors.size == 0) or include_flagged:
            #  in case there are no unflagged neighbors, use all flagged neighbors instead
            all_neighbors = True
            indices_neighbors = indices_neighbors_all.copy()
            indices_neighbors = self._neighbor_indices_including_flagged(indices_neighbors=indices_neighbors)

        return indices_neighbors, all_neighbors

    def _neighbor_indices_including_flagged(self, indices_neighbors: np.ndarray) -> np.ndarray:
        # TODO: check type hints
        for idx in indices_neighbors:
            if (idx in self.nanIndices) or (self.decomposition['N_components'][idx] == 0):
                indices_neighbors = np.delete(indices_neighbors, np.where(indices_neighbors == idx))
        #  sort the flagged neighbors according to the least number of flags
        sort = np.argsort(self.count_flags[indices_neighbors])
        return indices_neighbors[sort]

    def refit_spectrum_phase_1(self, index: int, i: int) -> List:
        """Refit a spectrum based on neighboring unflagged fit solutions.

        Parameters
        ----------
        index : Index ('index_fit' keyword) of the spectrum that will be refit.
        i : List index of the entry in the list that is handed over to the multiprocessing routine

        Returns
        -------
        A list in the form of [index, dictResults, indices_neighbors, refit] in case of a successful refit; otherwise
            [index, 'None', indices_neighbors, refit] is returned.

        """
        refit = False
        dictResults = None
        flags = []

        loc = self.locations_refit[i]
        spectrum = self.data[index]
        rms = self.errors[index][0]
        signal_ranges = self.signalRanges[index]
        noise_spike_ranges = self.noiseSpikeRanges[index]
        signal_mask = self.signal_mask[index]

        #  determine all neighbors that should be used for the refitting

        neighbors = get_neighbors(loc, shape=self.shape)
        # TODO: Is it okay, that flagged spectra are not included here?
        indices_neighbors, all_neighbors = self._determine_neighbor_indices(neighbors=neighbors, include_flagged=False)

        if (indices_neighbors.size == 0) or (all_neighbors and not self.use_all_neighors):
            return [index, None, indices_neighbors, refit]

        # skip refitting if there were no changes to the last iteration
        if (np.array_equal(indices_neighbors, self.neighbor_indices[index]) and self.mask_refitted[indices_neighbors].sum() < 1):
            return [index, None, indices_neighbors, refit]

        if self.refit_neg_res_peak and self.mask_neg_res_peak[index]:
            flags.append('residual')
        elif self.refit_broad and self.mask_broad_refit[index]:
            flags.append('broad')
        elif self.refit_blended and self.mask_blended[index]:
            flags.append('blended')

        flags.append('None')

        #  try to refit the spectrum with fit solution of individual unflagged neighboring spectra

        for flag in flags:
            dictResults, refit = self._try_refit_with_individual_neighbors(
                index=index,
                spectrum=spectrum,
                rms=rms,
                indices_neighbors=indices_neighbors,
                signal_ranges=signal_ranges,
                noise_spike_ranges=noise_spike_ranges,
                signal_mask=signal_mask,
                flag=flag
            )

            if dictResults is not None:
                return [index, dictResults, indices_neighbors, refit]

        #  try to refit the spectrum by grouping the fit solutions of all unflagged neighboring spectra

        if indices_neighbors.size > 1:
            dictResults, refit = self._try_refit_with_grouping(
                index=index,
                spectrum=spectrum,
                rms=rms,
                indices_neighbors=indices_neighbors,
                signal_ranges=signal_ranges,
                noise_spike_ranges=noise_spike_ranges,
                signal_mask=signal_mask
            )

        if (not all_neighbors and self.use_all_neighors):
            #  even though we now use indices_neighbors_all we still return indices_neighbors to avoid repeating the refitting
            indices_neighbors_all, all_neighbors = self._determine_neighbor_indices(neighbors=neighbors,
                                                                                    include_flagged=True)
            indices_neighbors_flagged = np.setdiff1d(indices_neighbors_all, indices_neighbors).astype('int')

            if indices_neighbors_flagged.size == 0:
                return [index, None, indices_neighbors, refit]

            for flag in flags:
                dictResults, refit = self._try_refit_with_individual_neighbors(
                    index=index,
                    spectrum=spectrum,
                    rms=rms,
                    indices_neighbors=indices_neighbors_flagged,
                    signal_ranges=signal_ranges,
                    noise_spike_ranges=noise_spike_ranges,
                    signal_mask=signal_mask,
                    flag=flag)

                if dictResults is not None:
                    return [index, dictResults, indices_neighbors, refit]

            if indices_neighbors_all.size > 1:
                dictResults, refit = self._try_refit_with_grouping(
                    index=index,
                    spectrum=spectrum,
                    rms=rms,
                    indices_neighbors=indices_neighbors_all,
                    signal_ranges=signal_ranges,
                    noise_spike_ranges=noise_spike_ranges,
                    signal_mask=signal_mask
                )

        return [index, dictResults, indices_neighbors, refit]

    def _try_refit_with_grouping(self,
                                 index: int,
                                 spectrum: np.ndarray,
                                 rms: float,
                                 indices_neighbors: np.ndarray,
                                 signal_ranges: List,
                                 noise_spike_ranges: List,
                                 signal_mask: np.ndarray) -> Tuple[Optional[Dict], bool]:
        """Try to refit a spectrum by grouping all neighboring unflagged fit solutions.

        Parameters
        ----------
        index : Index ('index_fit' keyword) of the spectrum that will be refit.
        spectrum : Spectrum to refit.
        rms : Root-mean-square noise value of the spectrum.
        indices_neighbors : Array containing the indices of all neighboring fit solutions that should be used for the
            grouping.
        signal_ranges : Nested list containing info about ranges of the spectrum that were estimated to contain signal.
            The goodness-of-fit calculations are only performed for the spectral channels within these ranges.
        noise_spike_ranges : Nested list containing info about ranges of the spectrum that were estimated to contain
            noise spike features. These will get masked out from goodness-of-fit calculations.
        signal_mask : Boolean array containing the information of signal_ranges.

        Returns
        -------
        dictResults : Information about the new best fit solution in case of a successful refit. Otherwise 'None' is
            returned.
        refit : Information of whether there was a new successful refit.

        """
        #  prepare fit parameter values of all unflagged neighboring fit solutions for the grouping
        amps, means, fwhms = self._get_initial_values(indices_neighbors)
        refit = False

        #  Group fit parameter values of all unflagged neighboring fit solutions and try to refit the spectrum with the
        #  new resulting average fit parameter values. First we try to group the fit solutions only by their mean
        #  position values. If this does not yield a new successful refit, we group the fit solutions by their mean
        #  position and FWHM values.

        for split_fwhm in [False, True]:
            dictComps = self._grouping(amps_tot=amps, means_tot=means, fwhms_tot=fwhms, split_fwhm=split_fwhm)
            dictComps = self._determine_average_values(spectrum=spectrum, rms=rms, dictComps=dictComps)

            #  try refit with the new average fit solution values

            if len(dictComps.keys()) > 0:
                dictResults = self._gaussian_fitting(
                    spectrum=spectrum,
                    rms=rms,
                    dictComps=dictComps,
                    signal_ranges=signal_ranges,
                    noise_spike_ranges=noise_spike_ranges,
                    signal_mask=signal_mask
                )
                refit = True
                if dictResults is None:
                    continue
                if self._choose_new_fit(dictResults, index):
                    return dictResults, refit

        return None, refit

    def _skip_index_for_refitting(self, index: int, index_neighbor: int) -> bool:
        """Check whether neighboring fit solution should be skipped.

        We want to exclude (most likely futile) refits with initial guesses from the fit solutions of neighboring
        spectra if the same fit solutions were already used in a previous iteration.

        Parameters
        ----------
        index : Index ('index_fit' keyword) of the spectrum that will be refit.
        index_neighbor : Index ('index_fit' keyword) of the neighboring fit solution.

        Returns
        -------
        Whether to skip the neighboring fit solution for an attempted refit.

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

    def _try_refit_with_individual_neighbors(self,
                                             index: int,
                                             spectrum: np.ndarray,
                                             rms: float,
                                             indices_neighbors: np.ndarray,
                                             signal_ranges: List,
                                             noise_spike_ranges: List,
                                             signal_mask: np.ndarray,
                                             interval: Optional[List] = None,
                                             n_centroids: Optional[int] = None,
                                             flag: str = 'none',
                                             dct_new_fit: Optional[Dict] = None) -> Tuple[Optional[Dict], bool]:
        """Try to refit a spectrum with the fit solution of an unflagged neighboring spectrum.

        Parameters
        ----------
        index : Index ('index_fit' keyword) of the spectrum that will be refit.
        spectrum : Spectrum to refit.
        rms : Root-mean-square noise value of the spectrum.
        indices_neighbors : Array containing the indices of all neighboring fit solutions that should be used for the
            grouping.
        signal_ranges : Nested list containing info about ranges of the spectrum that were estimated to contain signal.
            The goodness-of-fit calculations are only performed for the spectral channels within these ranges.
        noise_spike_ranges : Nested list containing info about ranges of the spectrum that were estimated to contain
            noise spike features. These will get masked out from goodness-of-fit calculations.
        signal_mask : Boolean array containing the information of signal_ranges.
        interval : List specifying the interval of spectral channels containing the flagged feature in the form of
            [lower, upper]. Only used in phase 2 of the spatially coherent refitting.
        n_centroids : Number of centroid positions that should be present in interval.
        flag : Flagged criterion that should be refit: 'broad', 'blended', or 'residual'.
        dct_new_fit : Only used in phase 2 of the spatially coherent refitting, in case the best fit solution was
            already updated in a previous iteration.

        Returns
        -------
        dictResults : Information about the new best fit solution in case of a successful refit. Otherwise 'None' is
            returned.
        refit : Information of whether there was a new successful refit.

        """
        dictComps = None
        refit = False

        for index_neighbor in indices_neighbors:
            #  check whether to use the neighboring fit solution or skip it
            if self._skip_index_for_refitting(index, index_neighbor):
                continue

            #  try to only replace part of the fit solution with new initial guesses from the neighboring fit solution
            #  for components flagged as broad, blended, or causing a negative residual feature. Otherwise use the
            #  entire fit solution of the neighboring spectrum.

            # TODO: check if this if-elif condition is correct and can be simplified
            if flag in {'broad', 'blended', 'residual'}:
                dictComps = self._replace_flagged_interval(
                    index=index,
                    index_neighbor=index_neighbor,
                    spectrum=spectrum,
                    rms=rms,
                    flag=flag
                )
            elif interval is not None:
                dictComps = self._replace_flagged_interval(
                    index=index,
                    index_neighbor=index_neighbor,
                    spectrum=spectrum,
                    rms=rms,
                    interval=interval,
                    dct_new_fit=dct_new_fit
                )
            else:
                dictComps = self._get_initial_values_from_neighbor(i=index_neighbor, spectrum=spectrum)

            if dictComps is None:
                continue

            #  try to refit with new fit solution

            dictResults = self._gaussian_fitting(
                spectrum=spectrum,
                rms=rms,
                dictComps=dictComps,
                signal_ranges=signal_ranges,
                noise_spike_ranges=noise_spike_ranges,
                signal_mask=signal_mask)
            refit = True
            if dictResults is None:
                continue
            if self._choose_new_fit(dictResults=dictResults,
                                    index=index,
                                    dct_new_fit=dct_new_fit,
                                    interval=interval,
                                    n_centroids=n_centroids):
                return dictResults, refit

        return None, refit

    def _get_refit_interval(self,
                            spectrum: np.ndarray,
                            rms: float,
                            amps: List,
                            fwhms: List,
                            means: List,
                            flag: str) -> List:
        """Get interval of spectral channels containing flagged feature selected for refitting.

        Parameters
        ----------
        spectrum : Spectrum to refit.
        rms : Root-mean-square noise value of the spectrum.
        amps : List of amplitude values of the fitted components.
        fwhms : List of FWHM values of the fitted components.
        means : List of mean position values of the fitted components.
        flag : Flagged criterion that should be refit: 'broad', 'blended', or 'residual'.

        Returns
        -------
        List specifying the interval of spectral channels containing the flagged feature in the form of [lower, upper].

        """
        #  for component flagged as broad select the interval [mean - FWHM, mean + FWHM]
        if flag == 'blended':
            params = amps + fwhms + means
            separation_factor = self.decomposition['improve_fit_settings']['separation_factor']
            indices = get_fully_blended_gaussians(params, separation_factor=separation_factor)
            lower = max(0, min(np.array(means)[indices] - np.array(fwhms)[indices]))
            upper = max(np.array(means)[indices] + np.array(fwhms)[indices])
        elif flag == 'broad':
            idx = np.argmax(np.array(fwhms))  # idx of broadest component
            lower = max(0, means[idx] - fwhms[idx])
            upper = means[idx] + fwhms[idx]
        elif flag == 'residual':
            dct = self.decomposition['improve_fit_settings'].copy()

            best_fit_list = [None for _ in range(10)]
            best_fit_list[0] = amps + fwhms + means
            best_fit_list[2] = len(amps)
            residual = spectrum - combined_gaussian(amps=amps, fwhms=fwhms, means=means, x=self.channels)
            best_fit_list[4] = residual

            #  TODO: What if multiple negative residual features occur in one spectrum?
            idx = check_for_negative_residual(
                vel=self.channels,
                data=spectrum,
                errors=rms,
                best_fit_list=best_fit_list,
                dct=dct,
                get_idx=True
            )
            if idx is None:
                #  TODO: check if self.channels[-1] causes problems
                return [self.channels[0], self.channels[-1]]
            lower = max(0, means[idx] - fwhms[idx])
            upper = means[idx] + fwhms[idx]

        return [lower, upper]

    def _replace_flagged_interval(self,
                                  index: int,
                                  index_neighbor: int,
                                  spectrum: np.ndarray,
                                  rms: float,
                                  flag: str = 'none',
                                  interval: Optional[List] = None,
                                  dct_new_fit: Optional[Dict] = None) -> Dict:
        """Update initial guesses for fit components by replacing flagged feature with a neighboring fit solution.

        Parameters
        ----------
        index : Index ('index_fit' keyword) of the spectrum that will be refit.
        index_neighbor : Index ('index_fit' keyword) of the neighboring fit solution.
        spectrum : Spectrum to refit.
        rms : Root-mean-square noise value of the spectrum.
        flag : Flagged criterion that should be refit: 'broad', 'blended', or 'residual'.
        interval : List specifying the interval of spectral channels containing the flagged feature in the form of
            [lower, upper].
        dct_new_fit : Only used in phase 2 of the spatially coherent refitting, in case the best fit solution was
            already updated in a previous iteration.

        Returns
        -------
        dictComps : Dictionary containing updated initial guesses for the fit solution.

        """
        #  for phase 2 of the spatially coherent refitting; if fit solution was already updated in previous iteration
        if dct_new_fit is not None:
            amps = dct_new_fit['amplitudes_fit']
            fwhms = dct_new_fit['fwhms_fit']
            means = dct_new_fit['means_fit']

            amps_err = dct_new_fit['amplitudes_fit_err']
            fwhms_err = dct_new_fit['fwhms_fit_err']
            means_err = dct_new_fit['means_fit_err']
        else:
            amps = self.decomposition['amplitudes_fit'][index]
            fwhms = self.decomposition['fwhms_fit'][index]
            means = self.decomposition['means_fit'][index]

            amps_err = self.decomposition['amplitudes_fit_err'][index]
            fwhms_err = self.decomposition['fwhms_fit_err'][index]
            means_err = self.decomposition['means_fit_err'][index]

        #  remove fit solution(s) of fit component(s) that are causing the flagged feature

        if interval is None:
            interval = self._get_refit_interval(
                spectrum=spectrum,
                rms=rms,
                amps=amps,
                fwhms=fwhms,
                means=means,
                flag=flag
            )
        indices, interval = self._components_in_interval(fwhms=fwhms, means=means, interval=interval)

        amps, fwhms, means = remove_components_from_sublists(lst=[amps, fwhms, means], remove_indices=indices)
        amps_err, fwhms_err, means_err = remove_components_from_sublists(lst=[amps_err, fwhms_err, means_err],
                                                                         remove_indices=indices)

        #  get new initial guess(es) for removed component(s) from neighboring fit solution

        amps_new = self.decomposition['amplitudes_fit'][index_neighbor]
        fwhms_new = self.decomposition['fwhms_fit'][index_neighbor]
        means_new = self.decomposition['means_fit'][index_neighbor]

        amps_err_new = self.decomposition['amplitudes_fit'][index_neighbor]
        fwhms_err_new = self.decomposition['fwhms_fit'][index_neighbor]
        means_err_new = self.decomposition['means_fit'][index_neighbor]

        #  check which of the neighboring fit components overlap with the interval containing the flagged feature(s)
        indices, interval = self._components_in_interval(fwhms=fwhms_new, means=means_new, interval=interval)

        if len(indices) == 0:
            return None

        #  discard all neighboring fit components not overlappting with the interval containing the flagged feature(s)
        remove_indices = np.delete(np.arange(len(amps_new)), indices)
        amps_new, fwhms_new, means_new = remove_components_from_sublists(lst=[amps_new, fwhms_new, means_new],
                                                                         remove_indices=remove_indices)
        amps_err_new, fwhms_err_new, means_err_new = remove_components_from_sublists(
            lst=[amps_err_new, fwhms_err_new, means_err_new],
            remove_indices=remove_indices
        )

        if len(amps_new) == 0:
            return None

        #  get best fit with new fit solution(s) for only the interval that contained the removed components

        idx_lower = int(interval[0])
        idx_upper = int(interval[1]) + 2

        dictCompsInterval = {}
        for amp, fwhm, mean, mean_err in zip(amps_new, fwhms_new, means_new, means_err_new):
            dictCompsInterval = self._add_initial_value_to_dict(
                dictComps=dictCompsInterval,
                spectrum=spectrum[idx_lower:idx_upper],
                amp=amp,
                fwhm=fwhm,
                mean=mean - idx_lower,
                mean_bound=max(self.mean_separation, mean_err)
            )

        channels = np.arange(len(spectrum[idx_lower:idx_upper]))

        dictFit = self._gaussian_fitting(
            spectrum=spectrum[idx_lower:idx_upper],
            rms=rms,
            dictComps=dictCompsInterval,
            signal_ranges=None,
            noise_spike_ranges=None,
            signal_mask=None,
            params_only=True,
            channels=channels
        )

        if dictFit is None:
            return None

        #  create new dictionary of fit solution(s) by combining new fit component(s) taken from neighboring spectrum with the remaining fit component(s) outside the flagged interval

        dictComps = {}
        for amp, fwhm, mean, mean_err in zip(dictFit['amplitudes_fit'],
                                             dictFit['fwhms_fit'],
                                             dictFit['means_fit'],
                                             dictFit['means_fit_err']):
            dictComps = self._add_initial_value_to_dict(
                dictComps=dictComps,
                spectrum=spectrum,
                amp=amp,
                fwhm=fwhm,
                mean=mean + idx_lower,
                mean_bound=max(self.mean_separation, mean_err)
            )

        for amp, fwhm, mean, mean_err in zip(amps, fwhms, means, means_err):
            dictComps = self._add_initial_value_to_dict(
                dictComps=dictComps,
                spectrum=spectrum,
                amp=amp,
                fwhm=fwhm,
                mean=mean,
                mean_bound=max(self.mean_separation, mean_err)
            )

        return dictComps

    def _components_in_interval(self, fwhms: List, means: List, interval: List) -> Tuple[List, List]:
        """Find indices of components overlapping with the interval and update the interval range to accommodate full extent of the components.

        Component i is selected if means[i] +/- fwhms[i] overlaps with the
        interval.

        The interval is updated to accommodate all spectral channels contained in the range means[i] +/- fwhms[i].

        Parameters
        ----------
        fwhms : List of FWHM values of fit components.
        means : List of mean position values of fit components.
        interval : List specifying the interval of spectral channels containing the flagged feature in the form of
            [lower, upper].

        Returns
        -------
        indices : List with indices of components overlapping with interval.
        interval_new : Updated interval that accommodates all spectral channels contained in the range
            means[i] +/- fwhms[i].

        """
        lower_interval, upper_interval = interval.copy()
        lower_interval_new, upper_interval_new = interval.copy()
        indices = []

        for i, (mean, fwhm) in enumerate(zip(means, fwhms)):
            lower = max(0, mean - fwhm)
            upper = mean + fwhm
            if (lower_interval <= lower <= upper_interval) or (lower_interval <= upper <= upper_interval):
                lower_interval_new = min(lower_interval_new, lower)
                upper_interval_new = max(upper_interval_new, upper)
                indices.append(i)
        return indices, [lower_interval_new, upper_interval_new]

    @staticmethod
    # TODO: move this to another general module
    def upper_limit_for_amplitude(spectrum: np.ndarray, mean: float, fwhm: float, buffer_factor: float = 1.) -> float:
        stddev = fwhm / CONVERSION_STD_TO_FWHM
        idx_low = max(0, int(mean - stddev))
        idx_upp = int(mean + stddev) + 2
        return buffer_factor * np.max(spectrum[idx_low:idx_upp])

    def _add_initial_value_to_dict(self,
                                   dictComps: Dict,
                                   spectrum: np.ndarray,
                                   amp: float,
                                   fwhm: float,
                                   mean: float,
                                   mean_bound: float) -> Dict:
        """Update dictionary of fit components with new component.

        Parameters
        ----------
        dictComps : Dictionary of fit components.
        spectrum : Spectrum to refit.
        amp : Amplitude value of fit component.
        fwhm : FWHM value of fit component.
        mean : Mean position value of fit component.
        mean_bound : Relative bound (upper and lower) of mean position value of fit component.

        Returns
        -------
        dictComps : Updated dictionary of fit components.

        """
        #  TODO: add here also mean +/- stddev??

        dictComps[str(len(dictComps) + 1)] = {
            'amp_ini': amp,
            'mean_ini': mean,
            'fwhm_ini': fwhm,
            'amp_bounds': [
                0.,
                SpatialFitting.upper_limit_for_amplitude(spectrum, mean, fwhm, buffer_factor=1.1)
            ],
            'mean_bounds': [
                max(0., mean - mean_bound),
                mean + mean_bound
            ],
            'fwhm_bounds': [
                max(0., fwhm - self.fwhm_separation) if self.constrain_fwhm else 0.,
                fwhm + self.fwhm_separation if self.constrain_fwhm else None
            ]
        }
        return dictComps

    def _get_dictionary_value(self, key: str, index: int, dct_new_fit: Optional[Dict] = None):
        """Return a dictionary value.
        # TODO: type hint for return
        # TODO: replace this with dictionary method -> default

        Parameters
        ----------
        key : Key of the dictionary.
        index : Index ('index_fit' keyword) of the spectrum that gets/was refit.
        dct_new_fit : If this dictionary is supplied, the value is extracted from it (only used in phase 2 of the
            spatially coherent refitting); otherwise the value is extracted from the 'self.decomposition' dictionary

        """
        return dct_new_fit[key] if dct_new_fit is not None else self.decomposition[key][index]

    def _get_flags(self,
                   dictResults: Dict,
                   index: int,
                   key: Optional[str] = 'None',
                   flag: Optional[bool] = None,
                   dct_new_fit: Optional[Dict] = None) -> Tuple[int, int]:
        """Check how the refit affected the number of blended or negative residual features.

        This check will only be performed if the 'self.flag_blended=True' or 'self.flag_neg_res_peak=True'.

        Parameters
        ----------
        dictResults : Dictionary containing the new best fit results after the refit attempt.
        index : Index ('index_fit' keyword) of the spectrum that gets/was refit.
        key : Dictionary keys, either 'N_blended' or 'N_neg_res_peak'.
        flag : User-selected flag criterion, either 'self.flag_blended', or 'self.flag_neg_res_peak'
        dct_new_fit : Only used in phase 2 of the spatially coherent refitting, in case the best fit solution was
            already updated in a previous iteration.

        Returns
        -------
        flag_old : Count of flagged features present in spectrum before refit.
        flag_new : Count of flagged features present in spectrum after refit.

        """
        flag_old, flag_new = (0 for _ in range(2))

        if not flag:
            return flag_old, flag_new

        n_old = self._get_dictionary_value(key=key, index=index, dct_new_fit=dct_new_fit)
        n_new = dictResults[key]
        #  flag if old fitting results showed flagged feature
        if n_old > 0:
            flag_old = 1
        #  punish new fit if it contains more of the flagged features
        if n_new > n_old:
            flag_new = flag_old + 1
        #  same flags if the new and old fitting results show the same number of features
        elif n_new == n_old:
            flag_new = flag_old

        return flag_old, flag_new

    def _get_flags_rchi2(self, dictResults: Dict, index: int, dct_new_fit: Optional[Dict] = None) -> Tuple[int, int]:
        """Check how the reduced chi-square value of a spectrum changed after the refit.

        This check will only be performed if the 'self.flag_rchi2=True'.

        Parameters
        ----------
        dictResults : Dictionary containing the new best fit results after the refit attempt.
        index : Index ('index_fit' keyword) of the spectrum that gets/was refit.
        dct_new_fit : Only used in phase 2 of the spatially coherent refitting, in case the best fit solution was
            already updated in a previous iteration.

        Returns
        -------
        flag_old : Flag value before the refit.
        flag_new : Flag value after the refit.

        """
        flag_old, flag_new = (0 for _ in range(2))

        if not self.flag_rchi2:
            return flag_old, flag_new

        rchi2_old = self._get_dictionary_value(key='best_fit_rchi2', index=index, dct_new_fit=dct_new_fit)
        rchi2_new = dictResults['best_fit_rchi2']

        if rchi2_old > self.rchi2_limit:
            flag_old += 1
        if rchi2_new > self.rchi2_limit:
            flag_new += 1

        #  reward new fit if it is closer to rchi2 = 1 and thus likely less "overfit"
        if max(rchi2_old, rchi2_new) < self.rchi2_limit and abs(rchi2_new - 1) < abs(rchi2_old - 1):
            flag_old += 1

        return flag_old, flag_new

    def _get_flags_pvalue(self, dictResults: Dict, index: int, dct_new_fit: Optional[Dict] = None) -> Tuple[int, int]:
        flag_old, flag_new = (0 for _ in range(2))

        if not self.flag_residual:
            return flag_old, flag_new

        pvalue_old = self._get_dictionary_value(key='pvalue', index=index, dct_new_fit=dct_new_fit)
        pvalue_new = dictResults['pvalue']

        if pvalue_old < self.min_pvalue:
            flag_old += 1
        if pvalue_new < self.min_pvalue:
            flag_new += 1

        #  punish fit if pvalue got worse
        if pvalue_new < pvalue_old:
            flag_new += 1

        return flag_old, flag_new

    def _get_flags_broad(self, dictResults: Dict, index: int, dct_new_fit: Optional[Dict] = None) -> Tuple[int, int]:
        """Check how the refit affected the number of components flagged as broad.

        This check will only be performed if the 'self.flag_broad=True'.

        Parameters
        ----------
        dictResults : Dictionary containing the new best fit results after the refit attempt.
        index : Index ('index_fit' keyword) of the spectrum that gets/was refit.
        dct_new_fit : Only used in phase 2 of the spatially coherent refitting, in case the best fit solution was
            already updated in a previous iteration.

        Returns
        -------
        flag_old : Flag value before the refit.
        flag_new : Flag value after the refit.

        """
        flag_old, flag_new = (0 for _ in range(2))

        if not self.flag_broad:
            return flag_old, flag_new

        if self.mask_broad_flagged[index]:
            flag_old = 1
            fwhm_max_old = max(self._get_dictionary_value(key='fwhms_fit', index=index, dct_new_fit=dct_new_fit))
            fwhm_max_new = max(np.array(dictResults['fwhms_fit']))
            #  no changes to the fit
            if fwhm_max_new == fwhm_max_old:
                flag_new = 1
            #  punish fit if component got even broader
            elif fwhm_max_new > fwhm_max_old:
                flag_new = 2
        else:
            fwhms = dictResults['fwhms_fit']
            if len(fwhms) > 1:
                #  punish fit if broad component was introduced
                fwhms = sorted(dictResults['fwhms_fit'])
                if (fwhms[-1] > self.fwhm_factor * fwhms[-2]) and (fwhms[-1] - fwhms[-2]) > self.fwhm_separation:
                    flag_new = 1

        return flag_old, flag_new

    def _get_flags_ncomps(self, dictResults: Dict, index: int, dct_new_fit: Optional[Dict] = None) -> Tuple[int, int]:
        """Check how the number of component jumps changed after the refit.

        TODO: Remove unused dictResults -> also from code where function is called!

        Parameters
        ----------
        dictResults : Dictionary containing the new best fit results after the refit attempt.
        index : Index ('index_fit' keyword) of the spectrum that gets/was refit.
        dct_new_fit : Only used in phase 2 of the spatially coherent refitting, in case the best fit solution was
            already updated in a previous iteration.

        Returns
        -------
        flag_old : Flag value before the refit.
        flag_new : Flag value after the refit.

        """
        flag_old, flag_new = (0 for _ in range(2))

        if not self.flag_ncomps:
            return flag_old, flag_new

        njumps_old = self.ncomps_jumps[index]

        loc = self.location[index]
        indices = get_neighbors(loc, exclude_p=True, shape=self.shape, nNeighbors=1, get_indices=True)
        mask_indices = get_neighbors(loc, exclude_p=True, shape=self.shape, nNeighbors=1, get_mask=True)

        ncomps = np.ones(8) * np.nan
        ncomps[mask_indices] = self.ncomps[indices]
        ncomps_central = self._get_dictionary_value(key='N_components', index=index, dct_new_fit=dct_new_fit)
        ncomps = np.insert(ncomps, 4, ncomps_central)
        njumps_new = self._number_of_component_jumps(ncomps)

        ncomps_wmedian = self.ncomps_wmedian[index]
        ndiff_old = abs(ncomps_wmedian - self.ncomps[index])
        ndiff_new = abs(ncomps_wmedian - ncomps_central)

        if (njumps_old > self.n_max_jump_comps) or (ndiff_old > self.max_diff_comps):
            flag_old = 1
        if (njumps_new > self.n_max_jump_comps) or (ndiff_new > self.max_diff_comps):
            flag_new = 1
        if (njumps_new > njumps_old) or (ndiff_new > ndiff_old):
            flag_new += 1

        return flag_old, flag_new

    def _get_flags_centroids(self,
                             dictResults: Dict,
                             index: int,
                             dct_new_fit: Optional[Dict] = None,
                             interval: Optional[List] = None,
                             n_centroids: Optional[int] = None) -> Tuple[int, int]:
        """Check how the presence of centroid positions changed after the refit.

        This check is only performed in phase 2 of the spatially coherent refitting.

        Parameters
        ----------
        dictResults : Dictionary containing the new best fit results after the refit attempt.
        index : Index ('index_fit' keyword) of the spectrum that gets/was refit.
        dct_new_fit : Only used in phase 2 of the spatially coherent refitting, in case the best fit solution was
            already updated in a previous iteration.
        interval : List specifying the interval of spectral channels where 'n_centroids' number of centroid positions
            are required.
        n_centroids : Number of centroid positions that should be present in interval.

        Returns
        -------
        flag_old : Flag value before the refit.
        flag_new : Flag value after the refit.

        """
        flag_old, flag_new = (0 for _ in range(2))

        if interval is None:
            return flag_old, flag_new

        means_old = self._get_dictionary_value(key='means_fit', index=index, dct_new_fit=dct_new_fit)
        means_new = dictResults['means_fit']

        flag_old, flag_new = (2 for _ in range(2))

        n_centroids_old = sum(interval[0] < x < interval[1] for x in means_old)
        n_centroids_new = sum(interval[0] < x < interval[1] for x in means_new)

        #  reward new fit if it has the required number of centroid positions within 'interval'
        if n_centroids_new == n_centroids:
            flag_new = 0
        #  reward new fit if its number of centroid positions within 'interval' got closer to the required value
        elif abs(n_centroids_new - n_centroids) < abs(n_centroids_old - n_centroids):
            flag_new = 1
        #  punish new fit if its number of centroid positions within 'interval' compared to the required value got worse than in the old fit
        elif abs(n_centroids_new - n_centroids) > abs(n_centroids_old - n_centroids):
            flag_old = 1

        return flag_old, flag_new

    def _choose_new_fit(self,
                        dictResults: Dict,
                        index: int,
                        dct_new_fit: Optional[Dict] = None,
                        interval: Optional[List] = None,
                        n_centroids: Optional[int] = None) -> bool:
        """Decide whether to accept the new fit solution as the new best fit.

        Parameters
        ----------
        dictResults : Dictionary containing the new best fit results after the refit attempt.
        index : Index ('index_fit' keyword) of the spectrum that gets/was refit.
        dct_new_fit : Only used in phase 2 of the spatially coherent refitting, in case the best fit solution was
            already updated in a previous iteration.
        interval : List specifying the interval of spectral channels where 'n_centroids' number of centroid positions
            are required.
        n_centroids : Number of centroid positions that should be present in interval.

        Returns
        -------
        Decision of whether new fit solution gets accepted as new best fit.

        """
        #  check how values/numbers of flagged features changed after the refit

        flag_blended_old, flag_blended_new = self._get_flags(
            dictResults=dictResults,
            index=index,
            key='N_blended',
            flag=self.flag_blended,
            dct_new_fit=dct_new_fit
        )

        flag_neg_res_peak_old, flag_neg_res_peak_new = self._get_flags(
            dictResults=dictResults,
            index=index,
            key='N_neg_res_peak',
            flag=self.flag_neg_res_peak,
            dct_new_fit=dct_new_fit
        )

        flag_rchi2_old, flag_rchi2_new = self._get_flags_rchi2(
            dictResults=dictResults,
            index=index,
            dct_new_fit=dct_new_fit
        )

        flag_residual_old, flag_residual_new = self._get_flags_pvalue(
            dictResults=dictResults,
            index=index,
            dct_new_fit=dct_new_fit
        )

        flag_broad_old, flag_broad_new = self._get_flags_broad(
            dictResults=dictResults,
            index=index,
            dct_new_fit=dct_new_fit
        )

        flag_ncomps_old, flag_ncomps_new = self._get_flags_ncomps(
            dictResults=dictResults,
            index=index,
            dct_new_fit=dct_new_fit
        )

        flag_centroids_old, flag_centroids_new = self._get_flags_centroids(
            dictResults=dictResults,
            index=index,
            dct_new_fit=dct_new_fit,
            interval=interval,
            n_centroids=n_centroids
        )

        #  only for phase 2: do not accept the new fit if there was no improvement for the centroid positions required in 'interval'
        if (n_centroids is not None) and (flag_centroids_new > 1):
            return False

        #  compute total flag values

        n_flags_old = (flag_blended_old
                       + flag_neg_res_peak_old
                       + flag_broad_old
                       + flag_rchi2_old
                       + flag_residual_old
                       + flag_ncomps_old
                       + flag_centroids_old)

        n_flags_new = (flag_blended_new
                       + flag_neg_res_peak_new
                       + flag_broad_new
                       + flag_rchi2_new
                       + flag_residual_new
                       + flag_ncomps_new
                       + flag_centroids_new)

        #  do not accept new fit if the total flag value increased
        if n_flags_new > n_flags_old:
            return False

        # if total flag value is the same or decreased there are two ways for the new best fit to get accepted as the
        # new best fit solution:
        # - accept the new fit if its AICc value is lower than AICc value of the current best fit solution
        # - if the AICc value of new fit is higher than the AICc value of the current best fit solution, only accept the
        # new fit if the values of the residual are normally distributed, i.e. if it passes the Kolmogorov-Smirnov test

        aicc_old = self._get_dictionary_value(key='best_fit_aicc', index=index, dct_new_fit=dct_new_fit)
        aicc_new = dictResults['best_fit_aicc']
        # residual_signal_mask = dictResults['residual_signal_mask']
        pvalue = dictResults['pvalue']

        if (aicc_new > aicc_old) and (pvalue < self.min_pvalue):
            return False

        return True

    def _get_values_for_indices(self, indices: np.ndarray, key: str) -> np.ndarray:
        # sum(tuple_of_lists, []) makes a flat list out of the tuple of lists
        return np.array(sum((self.decomposition[key][idx] for idx in indices), []))

    def _get_initial_values(self, indices_neighbors: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get sorted parameter values (amps, means, fwhms) of neighboring fit components for the grouping.

        Parameters
        ----------
        indices_neighbors : Array containing the indices of all neighboring fit solutions that should be used for the
            grouping. in the form [idx1, ..., idxN].

        Returns
        -------
        amps : Array of amplitude values (sorted according to mean position).
        means : Array of sorted mean position values.
        fwhms : Array of FWHM values (sorted according to mean position).

        """
        amps = self._get_values_for_indices(indices=indices_neighbors, key='amplitudes_fit')
        means = self._get_values_for_indices(indices=indices_neighbors, key='means_fit')
        fwhms = self._get_values_for_indices(indices=indices_neighbors, key='fwhms_fit')
        sort_order = np.argsort(means)
        return amps[sort_order], means[sort_order], fwhms[sort_order]

    def _grouping(self,
                  amps_tot: np.ndarray,
                  means_tot: np.ndarray,
                  fwhms_tot: np.ndarray,
                  split_fwhm: bool = True) -> collections.OrderedDict:
        """Grouping according to mean position values only or mean position values and FWHM values.

        Parameters
        ----------
        amps_tot : Array of amplitude values (sorted according to mean position).
        means_tot : Array of sorted mean position values.
        fwhms_tot : Array of FWHM values (sorted according to mean position).
        split_fwhm : Whether to group according to mean position and FWHM values ('True') or only according to mean
            position values ('False').

        Returns
        -------
        dictCompsOrdered : Ordered dictionary containing the results of the grouping.

        """
        #  group with regards to mean positions only
        means_diff = np.append(np.array([0.]), means_tot[1:] - means_tot[:-1])

        split_indices = np.where(means_diff > self.mean_separation)[0]
        split_means_tot = np.split(means_tot, split_indices)
        split_fwhms_tot = np.split(fwhms_tot, split_indices)
        split_amps_tot = np.split(amps_tot, split_indices)

        dictComps = {}

        for amps, fwhms, means in zip(split_amps_tot, split_fwhms_tot, split_means_tot):
            if (len(means) == 1) or not split_fwhm:
                key = f"{len(dictComps) + 1}"
                dictComps[key] = {"amps": amps, "means": means, "fwhms": fwhms}
                continue

            #  also group with regards to FWHM values

            lst_of_grouped_indices = []
            for i in range(len(means)):
                grouped_indices_means = np.where((np.abs(means - means[i]) < self.mean_separation))[0]
                grouped_indices_fwhms = np.where((np.abs(fwhms - fwhms[i]) < self.fwhm_separation))[0]
                ind = np.intersect1d(grouped_indices_means, grouped_indices_fwhms)
                lst_of_grouped_indices.append(list(ind))

            #  merge all sublists from lst_of_grouped_indices that share common indices

            G = to_graph(lst_of_grouped_indices)
            lst = list(connected_components(G))
            lst = [list(l) for l in lst]

            for sublst in lst:
                key = f"{len(dictComps) + 1}"
                dictComps[key] = {"amps": amps[sublst], "means": means[sublst], "fwhms": fwhms[sublst]}

        dictCompsOrdered = collections.OrderedDict()
        for i, k in enumerate(sorted(dictComps, key=lambda k: len(dictComps[k]['amps']), reverse=True)):
            dictCompsOrdered[str(i + 1)] = dictComps[k]

        return dictCompsOrdered

    def _get_initial_values_from_neighbor(self, i: int, spectrum: np.ndarray) -> Dict:
        """Get dictionary with information about all fit components from neighboring fit solution.

        Parameters
        ----------
        i : Index of neighboring fit solution.
        spectrum : Spectrum to refit.

        Returns
        -------
        dictComps : Dictionary containing information about all fit components from neighboring fit solution.

        """
        dictComps = {}

        for key in range(self.decomposition['N_components'][i]):
            amp = self.decomposition['amplitudes_fit'][i][key]
            mean = self.decomposition['means_fit'][i][key]
            mean_err = self.decomposition['means_fit_err'][i][key]
            fwhm = self.decomposition['fwhms_fit'][i][key]

            mean_min = min(mean - self.mean_separation, mean - mean_err)
            mean_min = max(0, mean_min)  # prevent negative values
            mean_max = max(mean + self.mean_separation, mean + mean_err)

            fwhm_min = 0
            fwhm_max = None

            if self.constrain_fwhm:
                fwhm_min = max(0, fwhm - self.fwhm_separation)
                fwhm_max = fwhm + self.fwhm_separation

            keyname = str(key + 1)
            dictComps[keyname] = {
                'amp_ini': amp,
                'mean_ini': mean,
                'fwhm_ini': fwhm,
                'amp_bounds': [0.0, SpatialFitting.upper_limit_for_amplitude(spectrum, mean, fwhm, buffer_factor=1.1)],
                'mean_bounds': [mean_min, mean_max],
                'fwhm_bounds': [fwhm_min, fwhm_max],
            }

        return dictComps

    def _determine_average_values(self,
                                  spectrum: np.ndarray,
                                  rms: float,
                                  dictComps: collections.OrderedDict) -> collections.OrderedDict:
        """Determine average values for fit components obtained by grouping.

        Parameters
        ----------
        spectrum : Spectrum to refit.
        rms : Root-mean-square noise value of the spectrum.
        dictComps : Ordered dictionary containing results of the grouping.

        Returns
        -------
        dictComps : Updated ordered dictionary containing average values for the fit components obtained via the
            grouping.

        """
        for key in dictComps.copy().keys():
            amps = np.array(dictComps[key]['amps'])
            #  TODO: also exclude all groups with two points?
            if len(amps) == 1:
                dictComps.pop(key)
                continue
            means = np.array(dictComps[key]['means'])
            fwhms = np.array(dictComps[key]['fwhms'])

            # TODO: take the median instead of the mean??
            amp_ini = np.mean(amps)
            mean_ini = np.mean(means)
            fwhm_ini = np.mean(fwhms)

            if (amp_max := SpatialFitting.upper_limit_for_amplitude(spectrum, mean_ini, fwhm_ini)) < self.snr*rms:
                dictComps.pop(key)
                continue

            #  determine fitting constraints for mean value
            lower_interval = max(abs(mean_ini - np.min(means)), self.mean_separation)
            mean_min = max(0, mean_ini - lower_interval)

            upper_interval = max(abs(mean_ini - np.max(means)), self.mean_separation)
            mean_max = mean_ini + upper_interval

            fwhm_min = 0
            fwhm_max = None

            if self.constrain_fwhm:
                #  determine fitting constraints for fwhm value
                lower_interval = max(abs(fwhm_ini - np.min(fwhms)), self.fwhm_separation)
                fwhm_min = max(0, fwhm_ini - lower_interval)

                upper_interval = max(abs(fwhm_ini - np.max(fwhms)), self.fwhm_separation)
                fwhm_max = fwhm_ini + upper_interval

            dictComps[key]['amp_ini'] = amp_ini
            dictComps[key]['mean_ini'] = mean_ini
            dictComps[key]['fwhm_ini'] = fwhm_ini

            dictComps[key]['amp_bounds'] = [0., 1.1*amp_max]
            dictComps[key]['mean_bounds'] = [mean_min, mean_max]
            # dictComps[key]['fwhm_bounds'] = [0., None]
            dictComps[key]['fwhm_bounds'] = [fwhm_min, fwhm_max]
        return dictComps

    def _gaussian_fitting(self,
                          spectrum: np.ndarray,
                          rms: float,
                          dictComps: Dict,
                          signal_ranges: List,
                          noise_spike_ranges: List,
                          signal_mask: np.ndarray,
                          params_only: bool = False,
                          channels: Optional[np.ndarray] = None) -> Dict:
        """Perform a new Gaussian decomposition with updated initial guesses.

        Parameters
        ----------
        spectrum : Spectrum to refit.
        rms : Root-mean-square noise value of the spectrum.
        dictComps : Dictionary containing information about new initial guesses for fit components.
        signal_ranges : Nested list containing info about ranges of the spectrum that were estimated to contain signal.
            The goodness-of-fit calculations are only performed for the spectral channels within these ranges.
        noise_spike_ranges : Nested list containing info about ranges of the spectrum that were estimated to contain
            noise spike features. These will get masked out from goodness-of-fit calculations.
        signal_mask : Boolean array containing the information of signal_ranges.
        params_only : If set to 'True', the returned dictionary of the fit results will only contain information about
            the amplitudes, FWHM values and mean positions of the fitted components.
        channels : Array containing the number of spectral channels.

        Returns
        -------
        dictResults : Dictionary containing information about the fit results.

        """
        if channels is None:
            n_channels = self.n_channels
            channels = self.channels
        else:
            n_channels = len(channels)

        if noise_spike_ranges:
            noise_spike_mask = mask_channels(
                n_channels=n_channels,
                ranges=[[0, n_channels]],
                remove_intervals=noise_spike_ranges
            )
        else:
            noise_spike_mask = None

        errors = np.ones(n_channels)*rms

        #  correct dictionary key
        dct = self.decomposition['improve_fit_settings'].copy()
        dct['max_amp'] = dct['max_amp_factor'] * np.max(spectrum)

        #  set limits for fit parameters
        params, params_min, params_max = ([] for _ in range(3))
        for key in ['amp', 'fwhm', 'mean']:
            for nr in dictComps.keys():
                params.append(dictComps[nr][f'{key}_ini'])
                params_min.append(dictComps[nr][f'{key}_bounds'][0])
                params_max.append(dictComps[nr][f'{key}_bounds'][1])

        #  get new best fit
        best_fit_list = get_best_fit(
            vel=channels,
            data=spectrum,
            errors=errors,
            params_fit=params,
            dct=dct,
            first=True,
            signal_ranges=signal_ranges,
            signal_mask=signal_mask,
            params_min=params_min,
            params_max=params_max,
            noise_spike_mask=noise_spike_mask
        )

        # #  get a new best fit that is unconstrained
        # params = best_fit_list[0]
        #
        # best_fit_list = get_best_fit(
        #     self.channels, spectrum, errors, params, dct, first=True,
        #     signal_ranges=signal_ranges, signal_mask=signal_mask)

        #  check for unfit residual peaks
        #  TODO: set fitted_residual_peaks to input offset positions??
        fitted_residual_peaks = []
        new_fit = True

        while new_fit:
            best_fit_list[7] = False
            best_fit_list, fitted_residual_peaks = check_for_peaks_in_residual(
                vel=channels,
                data=spectrum,
                errors=errors,
                best_fit_list=best_fit_list,
                dct=dct,
                fitted_residual_peaks=fitted_residual_peaks,
                signal_ranges=signal_ranges,
                signal_mask=signal_mask,
                noise_spike_mask=noise_spike_mask
            )
            new_fit = best_fit_list[7]

        params = best_fit_list[0]
        params_errs = best_fit_list[1]
        ncomps = best_fit_list[2]
        best_fit = best_fit_list[3]
        residual_signal_mask = best_fit_list[4][signal_mask]
        rchi2 = best_fit_list[5]
        aicc = best_fit_list[6]
        pvalue = best_fit_list[10]

        if ncomps == 0:
            return None

        amps, fwhms, means = split_params(params=params, ncomps=ncomps)
        amps_errs, fwhms_errs, means_errs = split_params(params=params_errs, ncomps=ncomps)

        keys = ['amplitudes_fit', 'fwhms_fit', 'means_fit',
                'amplitudes_fit_err', 'fwhms_fit_err', 'means_fit_err']
        vals = [amps, fwhms, means, amps_errs, fwhms_errs, means_errs]
        dictResults = dict(zip(keys, vals))

        if params_only:
            return dictResults

        # mask = mask_covering_gaussians(
        #     means, fwhms, n_channels, remove_intervals=noise_spike_ranges)
        # rchi2_gauss, aicc_gauss = goodness_of_fit(
        #     spectrum, best_fit, rms, ncomps, mask=mask, get_aicc=True)
        # TODO: completely remove rchi2_gauss and aicc_gauss (the function mask_covering_gaussians is already removed)
        rchi2_gauss, aicc_gauss = None, None

        N_blended = get_fully_blended_gaussians(
            params_fit=params,
            get_count=True,
            separation_factor=self.decomposition['improve_fit_settings']['separation_factor']
        )
        N_neg_res_peak = check_for_negative_residual(
            vel=channels,
            data=spectrum,
            errors=rms,
            best_fit_list=best_fit_list,
            dct=dct,
            get_count=True)

        keys = ["best_fit_rchi2", "best_fit_aicc", "residual_signal_mask",
                "gaussians_rchi2", "gaussians_aicc", "pvalue",
                "N_components", "N_blended", "N_neg_res_peak"]
        values = [rchi2, aicc, residual_signal_mask,
                  rchi2_gauss, aicc_gauss, pvalue,
                  ncomps, N_blended, N_neg_res_peak]
        for key, val in zip(keys, values):
            dictResults[key] = val

        return dictResults

    def _save_final_results(self) -> None:
        """Save the results of the spatially coherent refitting iterations."""
        pathToFile = os.path.join(self.decomp_dirname, f'{self.fin_filename}.pickle')
        pickle.dump(self.decomposition, open(pathToFile, 'wb'), protocol=2)
        say("\033[92mSAVED FILE:\033[0m '{}' in '{}'".format(
            self.fin_filename, self.decomp_dirname), logger=self.logger)

    #
    #  --- Phase 2: Refitting towards coherence in centroid positions ---
    #

    def _get_n_centroid(self, n_centroids: np.ndarray, weights: np.ndarray) -> int:
        """Calculate expected value for number of centroids per grouped centroid interval."""
        choices = list(set(n_centroids))
        # first, check only immediate neighboring spectra
        mask_weight = weights >= (self.w_1 / np.sqrt(2))
        counts_choices = [0 if choice == 0 else np.count_nonzero(n_centroids[mask_weight] == choice)
                          for choice in choices]

        n_neighbors = np.count_nonzero(mask_weight)
        if np.max(counts_choices) > 0.5 * n_neighbors:
            return choices[np.argmax(counts_choices)]

        # include additional neighbors that are two pixels away
        weights_choices = [0 if choice == 0 else sum(weights[n_centroids == choice]) for choice in choices]
        return choices[np.argmax(weights_choices)]

    def _combine_directions(self, dct: Dict) -> Dict:
        """Combine directions and build master dictionary."""
        indices_neighbors = np.concatenate(
            [dct[direction]['indices_neighbors'] for direction in self.weights.keys()])
        weights = np.concatenate(
            [dct[direction]['weights'] for direction in self.weights.keys()])
        intervals = merge_overlapping_intervals(
            [interval for direction in self.weights.keys() for interval in dct[direction]['means_interval']])
        means_interval = {str(key): [max(0, interval[0] - self.mean_separation / 2),
                                     interval[1] + self.mean_separation / 2]
                          for key, interval in enumerate(intervals, start=1)}
        # TODO: The following lines also appear in _check_continuity_centroids; should this be refactored into its own
        #  method?
        means_of_neighbors = [self.decomposition['means_fit'][idx] for idx in indices_neighbors]
        # estimate the expected number of centroids for interval
        ncomps_per_interval = {key: [sum(mean_min < mean < mean_max for mean in means) if bool(means) else 0
                                     for means in means_of_neighbors]
                               for key, (mean_min, mean_max) in means_interval.items()}
        # calculate number of centroids per centroid interval of neighbors
        n_centroids = {key: self._get_n_centroid(np.array(ncomps), weights)
                       for key, ncomps in ncomps_per_interval.items()}
        return {
            'indices_neighbors': indices_neighbors,
            'weights': weights,
            'means_interval': means_interval,
            'n_comps': ncomps_per_interval,
            'n_centroids': n_centroids
        }

    @functools.cached_property
    def weights(self):
        weights = dict.fromkeys(['horizontal', 'vertical'], {-2: self.w_2, -1: self.w_1, 1: self.w_1, 2: self.w_2})
        weights.update(dict.fromkeys(['diagonal_ul', 'diagonal_ur'], {
            -2: self.w_2 / np.sqrt(8),
            -1: self.w_1 / np.sqrt(2),
            1: self.w_1 / np.sqrt(2),
            2: self.w_2 / np.sqrt(8)
        }))
        return weights

    def _get_indices_and_weights_of_valid_neighbors(self, loc, idx, direction):
        indices_neighbors_and_center = get_neighbors(
            loc,
            exclude_p=False,
            shape=self.shape,
            nNeighbors=2,
            direction=direction,
            get_indices=True
        )
        is_neighbor = indices_neighbors_and_center != idx
        has_fit_components = np.array([self.decomposition['N_components'][i]
                                       for i in indices_neighbors_and_center]).astype(bool)
        relative_position_to_center = np.arange(len(indices_neighbors_and_center)) - np.flatnonzero(~is_neighbor)
        return (indices_neighbors_and_center[is_neighbor & has_fit_components],
                np.array([self.weights[direction][pos]
                          for pos in relative_position_to_center[is_neighbor & has_fit_components]]))

    def _check_continuity_centroids(self, idx: int, loc: Tuple) -> Dict:
        """Check for coherence of centroid positions of neighboring spectra.

        See Sect. 3.3.2. and Fig. 10 in Riener+ 2019 for more details.

        Parameters
        ----------
        idx : Index ('index_fit' keyword) of the central spectrum.
        loc : Location (ypos, xpos) of the central spectrum.

        Returns
        -------
        dct : Dictionary containing results of the spatial coherence check for neighboring centroid positions.

        """
        dct = {}

        for direction in self.weights.keys():
            indices_neighbors, weights_neighbors = self._get_indices_and_weights_of_valid_neighbors(loc, idx, direction)

            if len(indices_neighbors) == 0:
                dct[direction] = {
                    'indices_neighbors': indices_neighbors,
                    'weights': weights_neighbors,
                    'means_interval': [],
                }
                continue

            amps, means, fwhms = self._get_initial_values(indices_neighbors)
            grouping = self._grouping(amps_tot=amps, means_tot=means, fwhms_tot=fwhms, split_fwhm=False)

            means_interval = {key: [max(0, min(value['means']) - self.mean_separation / 2),
                                    max(value['means']) + self.mean_separation / 2]
                              for key, value in grouping.items()}

            means_of_neighbors = [self.decomposition['means_fit'][idx] for idx in indices_neighbors]
            ncomps_per_interval = {key: [sum(mean_min < mean < mean_max for mean in means) if bool(means) else 0
                                         for means in means_of_neighbors]
                                   for key, (mean_min, mean_max) in means_interval.items()}

            # Calculate weight of required components per centroid interval.
            factor_required = {key: sum(np.array(val, dtype=bool) * weights_neighbors)
                               for key, val in ncomps_per_interval.items()}

            # Keep only centroid intervals that have a certain minimum weight
            means_interval = [means_interval[key] for key in factor_required
                              if factor_required[key] > self.min_p]

            dct[direction] = {
                'indices_neighbors': indices_neighbors,
                'weights': weights_neighbors,
                'means_interval': means_interval,
            }

        return self._combine_directions(dct)

    def _check_for_required_components(self, idx: int, dct: Dict) -> Dict:
        """Check the presence of the required centroid positions within the determined interval."""
        means = self.decomposition['means_fit'][idx]
        keys_for_refit = [key for key, interval in dct['means_interval'].items()
                          if sum(interval[0] < x < interval[1] for x in means) != dct['n_centroids'][key]]
        return {
            'indices_neighbors': dct['indices_neighbors'],
            'weights': dct['weights'],
            'means_interval': {str(i): dct['means_interval'][key]
                               for i, key in enumerate(keys_for_refit, start=1)},
            'n_centroids': {str(i): dct['n_centroids'][key]
                            for i, key in enumerate(keys_for_refit, start=1)},
        }

    def _select_neighbors_to_use_for_refit(self,
                                           indices: np.ndarray,
                                           means_interval: Dict,
                                           n_centroids: Dict) -> Dict:
        """Select neighboring fit solutions with right number of centroid positions as refit solutions."""
        return {key: [idx for idx in indices
                      if n_centroids[key] == sum(interval[0] < x < interval[1]
                                                 for x in self.decomposition['means_fit'][idx])]
                for key, interval in means_interval.items()}

    def _determine_all_neighbors(self) -> None:
        """Determine the indices of all valid neighbors."""
        say("\ndetermine neighbors for all spectra...", logger=self.logger)

        mask_all = np.array([0 if x is None else 1 for x in self.decomposition['N_components']]).astype('bool')
        self.indices_all = np.array(self.decomposition['index_fit'])[mask_all]
        if self.pixel_range is not None:
            self.indices_all = np.array(self.decomposition['index_fit'])[~self.nanMask]
        self.locations_all = np.take(np.array(self.location), self.indices_all, axis=0)

        for i, loc in tqdm(zip(self.indices_all, self.locations_all)):
            indices_neighbors_total = np.array([])
            for direction in ['horizontal', 'vertical', 'diagonal_ul', 'diagonal_ur']:
                indices_neighbors = get_neighbors(
                    loc,
                    exclude_p=True,
                    shape=self.shape,
                    nNeighbors=2,
                    direction=direction,
                    get_indices=True
                )
                indices_neighbors_total = np.append(indices_neighbors_total, indices_neighbors)
            indices_neighbors_total = indices_neighbors_total.astype('int')
            self.neighbor_indices_all[i] = indices_neighbors_total

    def _check_indices_refit(self) -> None:
        """Check which spectra show incoherence in their fitted centroid positions and require refitting."""
        say('\ncheck which spectra require refitting...', logger=self.logger)
        if self.refitting_iteration == 1:
            self._determine_all_neighbors()

        if np.count_nonzero(self.mask_refitted) == len(self.mask_refitted):
            self.indices_refit = self.indices_all.copy()
            self.locations_refit = self.locations_all.copy()
            return

        indices_remove = np.array([])

        for i in self.indices_all:
            if np.count_nonzero(self.mask_refitted[self.neighbor_indices_all[i]]) == 0:
                indices_remove = np.append(indices_remove, i).astype('int')

        self.indices_refit = np.delete(self.indices_all.copy(), indices_remove)
        self.locations_refit = np.take(np.array(self.location), self.indices_refit, axis=0)

    def _check_continuity(self) -> None:
        """Check continuity of centroid positions.

        See Fig. 9 and Sect. 3.3.2. in Riener+ 2019 for more details.

        """
        self.refitting_iteration += 1
        say(f'\nthreshold for required components: {self.min_p:.3f}', logger=self.logger)

        self.determine_spectra_for_flagging()

        self._check_indices_refit()
        self._refitting()

    def refit_spectrum_phase_2(self, index: int, i: int) -> List:
        """Parallelized function for refitting spectra whose fitted centroid position values show spatial incoherence.

        Parameters
        ----------
        index : Index ('index_fit' keyword) of the spectrum that will be refit.
        i : List index of the entry in the list that is handed over to the multiprocessing routine

        Returns
        -------
        A list in the form of [index, dictResults_best, indices_neighbors, refit] in case of a successful refit;
            otherwise [index, 'None', indices_neighbors, refit] is returned.

        """
        refit = False
        loc = self.locations_refit[i]
        spectrum = self.data[index]
        rms = self.errors[index][0]
        signal_ranges = self.signalRanges[index]
        noise_spike_ranges = self.noiseSpikeRanges[index]
        signal_mask = self.signal_mask[index]

        dictResults, dictResults_best = (None for _ in range(2))
        #  TODO: check if this is correct:
        indices_neighbors = []

        dictComps = self._check_continuity_centroids(index, loc)
        dct_refit = self._check_for_required_components(index, dictComps)

        if self._finalize:
            return [index, dct_refit['means_interval'], dct_refit['n_centroids']]

        if len(dct_refit['means_interval'].keys()) == 0:
            return [index, None, indices_neighbors, refit]

        dct_refit['indices_refit'] = self._select_neighbors_to_use_for_refit(
            # TODO: Check if dct_refit['weights'] >= self.w_min condition was already checked earlier
            indices=dct_refit['indices_neighbors'][dct_refit['weights'] >= (self.w_1 / np.sqrt(2))],
            means_interval=dct_refit['means_interval'],
            n_centroids=dct_refit['n_centroids'],
        )

        for key, indices_neighbors in dct_refit['indices_refit'].items():
            dictResults, refit = self._try_refit_with_individual_neighbors(
                index=index,
                spectrum=spectrum,
                rms=rms,
                indices_neighbors=indices_neighbors,
                signal_ranges=signal_ranges,
                noise_spike_ranges=noise_spike_ranges,
                signal_mask=signal_mask,
                interval=dct_refit['means_interval'][key],
                n_centroids=dct_refit['n_centroids'][key],
                dct_new_fit=dictResults
            )

            if dictResults is not None:
                dictResults_best = dictResults

        return [index, dictResults_best, indices_neighbors, refit]
