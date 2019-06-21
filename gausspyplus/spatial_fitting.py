# @Author: riener
# @Date:   2019-01-22T08:00:18+01:00
# @Filename: spatial_fitting.py
# @Last modified by:   riener
# @Last modified time: 2019-04-08T10:18:41+02:00

import collections
import os
import pickle

import numpy as np
import scipy.ndimage as ndimage

from functools import reduce
from networkx.algorithms.components.connected import connected_components
from tqdm import tqdm

from .config_file import get_values_from_config_file
from .gausspy_py3.gp_plus import split_params, get_fully_blended_gaussians, check_for_peaks_in_residual, get_best_fit, check_for_negative_residual, remove_components_from_sublists
from .utils.determine_intervals import mask_covering_gaussians
from .utils.fit_quality_checks import goodness_of_fit
from .utils.gaussian_functions import combined_gaussian
from .utils.grouping_functions import to_graph, get_neighbors
from .utils.noise_estimation import mask_channels
from .utils.output import set_up_logger, say


class SpatialFitting(object):
    def __init__(self, path_to_pickle_file=None,
                 path_to_decomp_file=None, fin_filename=None,
                 config_file=''):
        """Class implementing the two phases of spatially coherent refitting discussed in Riener+ 2019.

        Parameters
        ----------
        path_to_pickle_file : type
            Description of parameter `path_to_pickle_file`.
        path_to_decomp_file : type
            Description of parameter `path_to_decomp_file`.
        fin_filename : type
            Description of parameter `fin_filename`.
        config_file : type
            Description of parameter `config_file`.

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

        if config_file:
            get_values_from_config_file(
                self, config_file, config_key='spatial fitting')

    def check_settings(self):
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
            raise Exception(
                "Need to set at least one 'refit_*' parameter to 'True'")

        if self.flag_blended is None:
            self.flag_blended = self.refit_blended
        if self.flag_neg_res_peak is None:
            self.flag_neg_res_peak = self.refit_neg_res_peak
        if self.flag_rchi2 is None:
            self.flag_rchi2 = self.refit_rchi2
        if self.flag_rchi2 and (self.rchi2_limit is None):
            raise Exception(
                "Need to set 'rchi2_limit' if 'flag_rchi2=True' or 'refit_rchi2=True'")
        if self.flag_residual is None:
            self.flag_residual = self.refit_residual
        if self.flag_broad is None:
            self.flag_broad = self.refit_broad
        if self.flag_ncomps is None:
            self.flag_ncomps = self.refit_ncomps

    def initialize(self):
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

        self.nanMask = np.isnan([np.nan if i is None else i
                                 for i in self.decomposition['N_components']])
        self.nanIndices = np.array(
            self.decomposition['index_fit'])[self.nanMask]

        self.signal_mask = [None for _ in range(self.nIndices)]
        for i, (noiseSpikeRanges, signalRanges) in enumerate(
                zip(self.noiseSpikeRanges, self.signalRanges)):
            if signalRanges is not None:
                self.signal_mask[i] = mask_channels(
                    self.n_channels, signalRanges,
                    remove_intervals=noiseSpikeRanges)

        #  starting condition so that refitting iteration can start
        # self.mask_refitted = np.ones(1)
        self.mask_refitted = np.array([1]*self.nIndices)
        self.list_n_refit = []
        self.refitting_iteration = 0

        normalization_factor = 1 / (2 * (self.weight_factor + 1))
        self.w_2 = normalization_factor
        self.w_1 = self.weight_factor * normalization_factor
        self.w_min = self.w_1 / np.sqrt(2)
        self.min_p = 1 - self.w_2

    def getting_ready(self):
        """Set up logger and write initial output to terminal."""
        self.logger = False
        if self.log_output:
            self.logger = set_up_logger(
                self.dirpath_gpy, self.filename, method='g+_spatial_refitting')

        phase = 1
        if self.phase_two:
            phase = 2

        string = 'Spatial refitting - Phase {}'.format(phase)
        banner = len(string) * '='
        heading = '\n' + banner + '\n' + string + '\n' + banner
        say(heading, logger=self.logger)

        string = str(
            '\nFlagging:'
            '\n - Blended components: {a}'
            '\n - Negative residual features: {b}'
            '\n - Broad components: {c}'
            '\n   flagged if FWHM of broadest component in spectrum is:'
            '\n   >= {d} times the FWHM of second broadest component'
            '\n   or'
            '\n   >= {d} times any FWHM in >= {e:.0%} of its neigbors'
            '\n - High reduced chi2 values (> {f}): {g}'
            '\n - Non-Gaussian distributed residuals: {h}'
            '\n - Differing number of components: {i}').format(
                a=self.flag_blended,
                b=self.flag_neg_res_peak,
                c=self.flag_broad,
                d=self.fwhm_factor,
                e=self.broad_neighbor_fraction,
                f=self.rchi2_limit,
                g=self.flag_rchi2,
                h=self.flag_residual,
                i=self.flag_ncomps)
        say(string, logger=self.logger)

        string = str(
            '\nExclude flagged spectra as possible refit solutions: {}'.format(
                self.exclude_flagged))
        if not self.phase_two:
            say(string, logger=self.logger)

        string = str(
            '\nRefitting:'
            '\n - Blended components: {a}'
            '\n - Negative residual features: {b}'
            '\n - Broad components: {c}'
            '\n   try to refit if FWHM of broadest component in spectrum is:'
            '\n   >= {d} times the FWHM of second broadest component'
            '\n   or'
            '\n   >= {d} times any FWHM in >= {e:.0%} of its neigbors'
            '\n - High reduced chi2 values (> {f}): {g}'
            '\n - Non-Gaussian distributed residuals: {h}'
            '\n - Differing number of components: {i}').format(
                a=self.refit_blended,
                b=self.refit_neg_res_peak,
                c=self.refit_broad,
                d=self.fwhm_factor_refit,
                e=self.broad_neighbor_fraction,
                f=self.rchi2_limit_refit,
                g=self.refit_rchi2,
                h=self.refit_residual,
                i=self.refit_ncomps)
        if not self.phase_two:
            say(string, logger=self.logger)

    def spatial_fitting(self, continuity=False):
        """Start the spatially coherent refitting.

        Parameters
        ----------
        continuity : bool (default: 'False')
            Set to 'True' for phase 2 of the spatially coherent refitting (coherence of centroid positions).

        """
        self.phase_two = continuity
        self.check_settings()
        self.initialize()
        self.getting_ready()
        if self.phase_two:
            self.list_n_refit.append([self.length])
            self.check_continuity()
        else:
            self.determine_spectra_for_refitting()

    def define_mask(self, key, limit, flag):
        """Create boolean mask with data values exceeding the defined limits set to 'True'.

        This mask is only created if 'flag=True'.

        Parameters
        ----------
        key : str
            Dictionary key of the parameter: 'N_blended', 'N_neg_res_peak', or 'best_fit_rchi2'.
        limit : int or float
            Upper limit of the corresponding value.
        flag : bool
            User-defined flag for the corresponding dictionary parameter.

        Returns
        -------
        mask : numpy.ndarray
            Boolean mask with values exceeding 'limit' set to 'True'.

        """
        if not flag:
            return np.zeros(self.length).astype('bool')

        array = np.array(self.decomposition[key])
        array[self.nanMask] = 0
        mask = array > limit
        return mask

    def define_mask_pvalue(self, key, limit, flag):
        if not flag:
            return np.zeros(self.length).astype('bool')

        array = np.array(self.decomposition[key])
        array[self.nanMask] = limit
        mask = array < limit
        return mask

    def define_mask_broad_limit(self, flag):
        """Create boolean mask identifying the location of broad fit components.

        The mask is 'True' at the location of spectra that contain fit components exceeding the 'max_fwhm' value.

        This mask is only created if 'flag=True'.

        Parameters
        ----------
        flag : bool
            User-defined 'flag_broad' parameter.

        Returns
        -------
        mask : numpy.ndarray

        """
        n_broad = np.zeros(self.length)

        if not flag:
            return n_broad.astype('bool'), n_broad

        for i, fwhms in enumerate(self.decomposition['fwhms_fit']):
            if fwhms is None:
                continue
            n_broad[i] = np.count_nonzero(np.array(fwhms) > self.max_fwhm)
        mask = n_broad > 0
        return mask, n_broad

    def broad_components(self, values):
        """Check for the presence of broad fit components.

        This check is performed by comparing the broadest fit components of a spectrum with its 8 immediate neighbors.

        A fit component is defined as broad if its FWHM value exceeds the FWHM value of the largest fit components of more than 'self.broad_neighbor_fraction' of its neighbors by at least a factor of 'self.fwhm_factor'.

        In addition we impose that the minimum difference between the compared FWHM values has to exceed 'self.fwhm_separation' to avoid flagging narrow components.

        Parameters
        ----------
        values : numpy.ndarray
            Array of FWHM values of the broadest fit components for a spectrum and its 8 immediate neighbors.

        Returns
        -------
        float or int
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
            if central_value > value * self.fwhm_factor and\
                    (central_value - value) > self.fwhm_separation:
                counter += 1
        if counter > values.size * self.broad_neighbor_fraction:
            return central_value
        return 0

    def define_mask_broad(self, flag):
        """Create a boolean mask indicating the location of broad fit components.

        If 'self.flag_broad=False' no locations are masked.

        Parameters
        ----------
        flag : bool
            User-defined 'self.flag_broad' parameter.

        Returns
        -------
        mask_broad : numpy.ndarray

        """
        if not flag:
            return np.zeros(self.length).astype('bool')

        broad_1d = np.empty(self.length)
        broad_1d.fill(np.nan)
        mask_broad = np.zeros(self.length)

        #  check if the fit component with the largest FWHM value of a spectrum satisfies the criteria to be flagged as a broad component by comparing it to the remaining components of the spectrum.
        for i, fwhms in enumerate(self.decomposition['fwhms_fit']):
            if fwhms is None:
                continue
            if len(fwhms) == 0:
                continue
            #  in case there is only one fit parameter there are no other components to compare; we need to compare it with the components of the immediate neighbors
            broad_1d[i] = max(fwhms)
            if len(fwhms) == 1:
                continue
            #  in case of multiple fit parameters select the one with the largest FWHM value and check whether it exceeds the second largest FWHM value in that spectrum by a factor of 'self.fwhm_factor'; also check if the absolute difference of their values exceeds 'self.fwhm_separation' to avoid narrow components.
            fwhms = sorted(fwhms)
            if (fwhms[-1] > self.fwhm_factor * fwhms[-2]) and\
                    (fwhms[-1] - fwhms[-2]) > self.fwhm_separation:
                mask_broad[i] = 1

        #  check if the fit component with the largest FWHM value of a spectrum satisfies the criteria to be flagged as a broad component by comparing it to the largest FWHM values of its 8 immediate neighbors.

        broad_2d = broad_1d.astype('float').reshape(self.shape)

        footprint = np.ones((3, 3))

        broad_fwhm_values = ndimage.generic_filter(
            broad_2d, self.broad_components, footprint=footprint,
            mode='constant', cval=np.nan).flatten()
        mask_broad = mask_broad.astype('bool')
        mask_broad += broad_fwhm_values.astype('bool')

        return mask_broad

    def weighted_median(self, data):
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
            w_median = (data[weights == np.max(weights)])[0]
        else:
            cs_weights = np.cumsum(s_weights)
            idx = np.where(cs_weights <= midpoint)[0][-1]
            if cs_weights[idx] == midpoint:
                w_median = np.mean(s_data[idx:idx + 2])
            else:
                w_median = s_data[idx + 1]
        return w_median

    def number_of_component_jumps(self, values):
        """Determine the number of component jumps towards neighboring fits.

        A component jump occurs if the number of components is different by more than 'self.max_jump_comps' components.

        Parameters
        ----------
        values : numpy.ndarray
            Array of the number of fit components for a spectrum and its 8 immediate neighbors.

        Returns
        -------
        int
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

    def define_mask_neighbor_ncomps(self, flag):
        """Create a boolean mask indicating the location of component jumps.

        Parameters
        ----------
        nanmask_1d : numpy.ndarray
            Boolean mask containing the information about already flagged spectra, which are not considered in the calculation unless 'self.only_print_flags=True'.
        flag : bool
            User-defined 'self.flag_ncomps' parameter.

        Returns
        -------
        mask_neighbor : numpy.ndarray
            Boolean mask indicating the location of component jump.
        ncomps_jumps : numpy.ndarray
            Array containing the information of how many component jumps occur at which location.
        ncomps_1d : numpy.ndarray
            Array containing the information about the number of fitted components per location.

        """
        if not flag:
            return np.zeros(self.length).astype('bool'), None, None

        # nanmask_1d += self.nanMask  # not really necessary
        # if self.only_print_flags:
        nanmask_1d = self.nanMask
        nanmask_2d = nanmask_1d.reshape(self.shape)
        ncomps_1d = np.empty(self.length)
        ncomps_1d.fill(np.nan)
        ncomps_1d[~self.nanMask] = np.array(
            self.decomposition['N_components'])[~self.nanMask]
        ncomps_2d = ncomps_1d.astype('float').reshape(self.shape)
        ncomps_2d[nanmask_2d] = np.nan

        mask_neighbor = np.zeros(self.length)
        footprint = np.ones((3, 3))

        ncomps_wmedian = ndimage.generic_filter(
            ncomps_2d, self.weighted_median, footprint=footprint,
            mode='constant', cval=np.nan).flatten()
        mask_neighbor[~self.nanMask] = ncomps_wmedian[~self.nanMask] > self.max_diff_comps

        ncomps_jumps = ndimage.generic_filter(
            ncomps_2d, self.number_of_component_jumps, footprint=footprint,
            mode='reflect', cval=np.nan).flatten()
        mask_neighbor[~self.nanMask] = ncomps_jumps[~self.nanMask] > self.n_max_jump_comps

        mask_neighbor = mask_neighbor.astype('bool')

        return mask_neighbor, ncomps_wmedian, ncomps_jumps, ncomps_1d

    def determine_spectra_for_flagging(self):
        """Flag spectra not satisfying user-defined flagging criteria."""
        self.mask_blended = self.define_mask(
            'N_blended', 0, self.flag_blended)
        self.mask_residual = self.define_mask(
            'N_neg_res_peak', 0, self.flag_neg_res_peak)
        self.mask_rchi2_flagged = self.define_mask(
            'best_fit_rchi2', self.rchi2_limit, self.flag_rchi2)
        self.mask_pvalue = self.define_mask_pvalue(
            'pvalue', self.min_pvalue, self.flag_residual)
        self.mask_broad_flagged = self.define_mask_broad(self.flag_broad)
        self.mask_broad_limit, self.n_broad = self.define_mask_broad_limit(
            self.flag_broad)
        self.mask_ncomps, self.ncomps_wmedian, self.ncomps_jumps, self.ncomps =\
            self.define_mask_neighbor_ncomps(self.flag_ncomps)

        mask_flagged = self.mask_blended + self.mask_residual\
            + self.mask_broad_flagged + self.mask_rchi2_flagged\
            + self.mask_pvalue + self.mask_ncomps
        self.indices_flagged = np.array(
            self.decomposition['index_fit'])[mask_flagged]

        if self.phase_two:
            n_flagged_blended = np.count_nonzero(self.mask_blended)
            n_flagged_neg_res_peak = np.count_nonzero(self.mask_residual)
            n_flagged_broad = np.count_nonzero(self.mask_broad_flagged)
            n_flagged_rchi2 = np.count_nonzero(self.mask_rchi2_flagged)
            n_flagged_residual = np.count_nonzero(self.mask_pvalue)
            n_flagged_ncomps = np.count_nonzero(self.mask_ncomps)

            text = str(
                "\n Flags:"
                "\n - {a} spectra w/ blended components"
                "\n - {b} spectra w/ negative residual feature"
                "\n - {c} spectra w/ broad feature"
                "\n   (info: {d} spectra w/ a FWHM > {e} channels)"
                "\n - {f} spectra w/ high rchi2 value"
                "\n - {h} spectra w/ residual not passing normality test"
                "\n - {g} spectra w/ differing number of components").format(
                    a=n_flagged_blended,
                    b=n_flagged_neg_res_peak,
                    c=n_flagged_broad,
                    d=np.count_nonzero(self.mask_broad_limit),
                    e=int(self.max_fwhm),
                    f=n_flagged_rchi2,
                    g=n_flagged_ncomps,
                    h=n_flagged_residual
                )

            say(text, logger=self.logger)

    def define_mask_refit(self):
        """Select spectra to refit in phase 1 of the spatially coherent refitting."""
        mask_refit = np.zeros(self.length).astype('bool')
        if self.refit_blended:
            mask_refit += self.mask_blended
        if self.refit_neg_res_peak:
            mask_refit += self.mask_residual
        if self.refit_broad:
            mask_refit += self.mask_broad_refit
        if self.refit_rchi2:
            mask_refit += self.mask_rchi2_refit
        if self.refit_residual:
            mask_refit += self.mask_pvalue
        if self.refit_ncomps:
            mask_refit += self.mask_ncomps

        self.indices_refit = np.array(
            self.decomposition['index_fit'])[mask_refit]
        # self.indices_refit = self.indices_refit[10495:10500]  # for debugging
        self.locations_refit = np.take(
            np.array(self.location), self.indices_refit, axis=0)

    def get_n_refit(self, flag, n_refit):
        if flag:
            return n_refit
        else:
            return 0

    def determine_spectra_for_refitting(self):
        """Determine spectra for refitting in phase 1 of the spatially coherent refitting."""
        say('\ndetermine spectra that need refitting...', logger=self.logger)

        #  flag spectra based on user-defined criteria
        self.determine_spectra_for_flagging()

        #  determine new masks for spectra that do not satisfy the user-defined criteria for broad components and reduced chi-square values; this is done because users can opt to use different values for flagging and refitting for these two criteria
        self.mask_broad_refit = self.define_mask_broad(
            self.refit_broad)
        self.mask_rchi2_refit = self.define_mask(
            'best_fit_rchi2', self.rchi2_limit_refit, self.refit_rchi2)

        #  select spectra for refitting based on user-defined criteria
        self.define_mask_refit()

        #  print the results of the flagging/refitting selections to the terminal

        n_spectra = sum([1 for x in self.decomposition['N_components']
                         if x is not None])
        n_indices_refit = len(self.indices_refit)
        n_flagged_blended = np.count_nonzero(self.mask_blended)
        n_flagged_neg_res_peak = np.count_nonzero(self.mask_residual)
        n_flagged_broad = np.count_nonzero(self.mask_broad_flagged)
        n_flagged_rchi2 = np.count_nonzero(self.mask_rchi2_flagged)
        n_flagged_residual = np.count_nonzero(self.mask_pvalue)
        n_flagged_ncomps = np.count_nonzero(self.mask_ncomps)

        n_refit_blended = self.get_n_refit(
            self.refit_blended, n_flagged_blended)
        n_refit_neg_res_peak = self.get_n_refit(
            self.refit_neg_res_peak, n_flagged_neg_res_peak)
        n_refit_broad = self.get_n_refit(
            self.refit_broad, np.count_nonzero(self.mask_broad_refit))
        n_refit_rchi2 = self.get_n_refit(
            self.refit_rchi2, np.count_nonzero(self.mask_rchi2_refit))
        n_refit_residual = self.get_n_refit(
            self.refit_residual, np.count_nonzero(self.mask_pvalue))
        n_refit_ncomps = self.get_n_refit(
            self.refit_ncomps, n_flagged_ncomps)

        n_refit_list = [
            n_refit_blended, n_refit_neg_res_peak, n_refit_broad,
            n_refit_rchi2, n_refit_residual, n_refit_ncomps]

        text = str(
            "\n{a} out of {b} spectra ({c:.2%}) selected for refitting:"
            "\n - {d} spectra w/ blended components ({e} flagged)"
            "\n - {f} spectra w/ negative residual feature ({g} flagged)"
            "\n - {h} spectra w/ broad feature ({i} flagged)"
            "\n   (info: {j} spectra w/ a FWHM > {k} channels)"
            "\n - {m} spectra w/ high rchi2 value ({n} flagged)"
            "\n - {q} spectra w/ residual not passing normality test ({r} flagged)"
            "\n - {o} spectra w/ differing number of components ({p} flagged)").format(
                a=n_indices_refit,
                b=n_spectra,
                c=n_indices_refit/n_spectra,
                d=n_refit_blended,
                e=n_flagged_blended,
                f=n_refit_neg_res_peak,
                g=n_flagged_neg_res_peak,
                h=n_refit_broad,
                i=n_flagged_broad,
                j=np.count_nonzero(self.mask_broad_limit),
                k=int(self.max_fwhm),
                m=n_refit_rchi2,
                n=n_flagged_rchi2,
                o=n_refit_ncomps,
                p=n_flagged_ncomps,
                q=n_refit_residual,
                r=n_flagged_residual
            )

        say(text, logger=self.logger)

        #  check if the stopping criterion is fulfilled

        if self.only_print_flags:
            return
        elif self.stopping_criterion(n_refit_list):
            self.save_final_results()
        else:
            self.list_n_refit.append(n_refit_list)
            self.refitting_iteration += 1
            self.refitting()

    def stopping_criterion(self, n_refit_list):
        """Check if spatial refitting iterations should be stopped."""
        #  stop refitting if the user-defined maximum number of iterations are reached
        if self.refitting_iteration >= self.max_refitting_iteration:
            return True
        #  stop refitting if the number of spectra selected for refitting is identical to a previous iteration
        if n_refit_list in self.list_n_refit:
            return True
        #  stop refitting if the number of spectra selected for refitting got higher than in the previous iteration for each user-defined criterion
        if self.refitting_iteration > 0:
            stop = True
            for i in range(len(n_refit_list)):
                if n_refit_list[i] < min([n[i] for n in self.list_n_refit]):
                    stop = False
            return stop

    def refitting(self):
        """Refit spectra with multiprocessing routine."""
        say('\nstart refit iteration #{}...'.format(
            self.refitting_iteration), logger=self.logger)

        #  initialize the multiprocessing routine

        import gausspyplus.parallel_processing
        gausspyplus.parallel_processing.init([self.indices_refit, [self]])

        #  try to refit spectra via the multiprocessing routine

        if self.phase_two:
            results_list = gausspyplus.parallel_processing.func(
                use_ncpus=self.use_ncpus, function='refit_phase_2')
        else:
            results_list = gausspyplus.parallel_processing.func(
                use_ncpus=self.use_ncpus, function='refit_phase_1')
        print('SUCCESS')

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
                say("Error for spectrum with index {}: {}".format(i, item),
                    logger=self.logger)
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

        if count_selected == 0:
            refit_percent = 0
        else:
            refit_percent = count_refitted/count_selected

        text = str(
            "\nResults of the refit iteration:"
            "\nTried to refit {a} spectra"
            "\nSuccessfully refitted {b} spectra ({c:.2%})"
            "\n\n***").format(
                a=count_selected,
                b=count_refitted,
                c=refit_percent)

        say(text, logger=self.logger)

        #  check if one of the stopping criteria is fulfilled

        if self.phase_two:
            if self.stopping_criterion([count_refitted]):
                self.min_p -= self.w_2
                self.list_n_refit = [[self.length]]
                self.mask_refitted = np.array([1]*self.nIndices)
            else:
                self.list_n_refit.append([count_refitted])

            if self.min_p < self.min_weight:
                self.save_final_results()
            else:
                self.check_continuity()
        else:
            self.determine_spectra_for_refitting()

    def determine_neighbor_indices(self, neighbors):
        """Determine indices of all valid neighboring pixels.

        Parameters
        ----------
        neighbors : list
            List containing information about the location of N neighboring spectra in the form [(y1, x1), ..., (yN, xN)].

        Returns
        -------
        indices_neighbors : numpy.ndarray
            Array containing the corresponding indices of the N neighboring spectra in the form [idx1, ..., idxN].

        """
        indices_neighbors = np.array([])
        for neighbor in neighbors:
            indices_neighbors = np.append(
                indices_neighbors, np.ravel_multi_index(neighbor, self.shape)).astype('int')

        #  check if neighboring pixels were selected for refitting, are masked out, or contain no fits and thus cannot be used

        #  whether to exclude all flagged neighboring spectra as well that
        #  were not selected for refitting
        if self.exclude_flagged:
            indices_bad = self.indices_flagged
        else:
            indices_bad = self.indices_refit

        for idx in indices_neighbors:
            if (idx in indices_bad) or (idx in self.nanIndices) or\
                    (self.decomposition['N_components'][idx] == 0):
                indices_neighbors = np.delete(
                    indices_neighbors, np.where(indices_neighbors == idx))
        return indices_neighbors

    def refit_spectrum_phase_1(self, index, i):
        """Refit a spectrum based on neighboring unflagged fit solutions.

        Parameters
        ----------
        index : int
            Index ('index_fit' keyword) of the spectrum that will be refit.
        i : int
            List index of the entry in the list that is handed over to the
            multiprocessing routine

        Returns
        -------
        list
            A list in the form of [index, dictResults, indices_neighbors, refit]
            in case of a successful refit; otherwise [index, 'None', indices_neighbors, refit] is returned.

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
        indices_neighbors = self.determine_neighbor_indices(neighbors)

        if indices_neighbors.size == 0:
            return [index, None, indices_neighbors, refit]

        # skip refitting if there were no changes to the last iteration
        if np.array_equal(indices_neighbors, self.neighbor_indices[index]):
            if self.mask_refitted[indices_neighbors].sum() < 1:
                return [index, None, indices_neighbors, refit]

        if self.refit_neg_res_peak and self.mask_residual[index]:
            flags.append('residual')
        elif self.refit_broad and self.mask_broad_refit[index]:
            flags.append('broad')
        elif self.refit_blended and self.mask_blended[index]:
            flags.append('blended')

        flags.append('None')

        #  try to refit the spectrum with fit solution of individual unflagged neighboring spectra

        for flag in flags:
            dictResults, refit = self.try_refit_with_individual_neighbors(
                index, spectrum, rms, indices_neighbors, signal_ranges,
                noise_spike_ranges, signal_mask, flag=flag)

            if dictResults is not None:
                return [index, dictResults, indices_neighbors, refit]

        #  try to refit the spectrum by grouping the fit solutions of all unflagged neighboring spectra

        if indices_neighbors.size > 1:
            dictResults, refit = self.try_refit_with_grouping(
                index, spectrum, rms, indices_neighbors, signal_ranges,
                noise_spike_ranges, signal_mask)

        return [index, dictResults, indices_neighbors, refit]

    def try_refit_with_grouping(self, index, spectrum, rms,
                                indices_neighbors, signal_ranges,
                                noise_spike_ranges, signal_mask):
        """Try to refit a spectrum by grouping all neighboring unflagged fit solutions.

        Parameters
        ----------
        index : int
            Index ('index_fit' keyword) of the spectrum that will be refit.
        spectrum : numpy.ndarray
            Spectrum to refit.
        rms : float
            Root-mean-square noise value of the spectrum.
        indices_neighbors : numpy.ndarray
            Array containing the indices of all neighboring fit solutions that should be used for the grouping.
        signal_ranges : list
            Nested list containing info about ranges of the spectrum that were estimated to contain signal. The goodness-of-fit calculations are only performed for the spectral channels within these ranges.
        noise_spike_ranges : list
            Nested list containing info about ranges of the spectrum that were estimated to contain noise spike features. These will get masked out from goodness-of-fit calculations.
        signal_mask : numpy.ndarray
            Boolean array containing the information of signal_ranges.

        Returns
        -------
        dictResults : dict
            Information about the new best fit solution in case of a successful refit. Otherwise 'None' is returned.
        refit : bool
            Information of whether there was a new successful refit.

        """
        #  prepare fit parameter values of all unflagged neighboring fit solutions for the grouping
        amps, means, fwhms = self.get_initial_values(indices_neighbors)
        refit = False

        #  Group fit parameter values of all unflagged neighboring fit solutions and try to refit the spectrum with the new resulting average fit parameter values. First we try to group the fit solutions only by their mean position values. If this does not yield a new successful refit, we group the fit solutions by their mean position and FWHM values.

        for split_fwhm in [False, True]:
            dictComps = self.grouping(
                amps, means, fwhms, split_fwhm=split_fwhm)
            dictComps = self.determine_average_values(
                spectrum, rms, dictComps)

            #  try refit with the new average fit solution values

            if len(dictComps.keys()) > 0:
                dictResults = self.gaussian_fitting(
                    spectrum, rms, dictComps, signal_ranges, noise_spike_ranges, signal_mask)
                refit = True
                if dictResults is None:
                    continue
                if self.choose_new_fit(dictResults, index):
                    return dictResults, refit

        return None, refit

    def skip_index_for_refitting(self, index, index_neighbor):
        """Check whether neighboring fit solution should be skipped.

        We want to exclude (most likely futile) refits with initial guesses from the fit solutions of neighboring spectra if the same fit solutions were already used in a previous iteration.

        Parameters
        ----------
        index : int
            Index ('index_fit' keyword) of the spectrum that will be refit.
        index_neighbor : int
            Index ('index_fit' keyword) of the neighboring fit solution.

        Returns
        -------
        bool
            Whether to skip the neighboring fit solution for an attempted refit.

        """
        if self.refitting_iteration > 1:
            #  check if spectrum was selected for refitting in any of the  previous iterations
            if self.neighbor_indices[index] is not None:
                #  check if neighbor was used in that refitting iteration
                if index_neighbor in self.neighbor_indices[index]:
                    #  check if neighbor was refit in previous iteration
                    if not self.mask_refitted[index_neighbor]:
                        return True
        return False

    def try_refit_with_individual_neighbors(self, index, spectrum, rms,
                                            indices_neighbors, signal_ranges,
                                            noise_spike_ranges, signal_mask,
                                            interval=None, n_centroids=None, flag='none', dct_new_fit=None):
        """Try to refit a spectrum with the fit solution of an unflagged neighboring spectrum.

        Parameters
        ----------
        index : int
            Index ('index_fit' keyword) of the spectrum that will be refit.
        spectrum : numpy.ndarray
            Spectrum to refit.
        rms : float
            Root-mean-square noise value of the spectrum.
        indices_neighbors : numpy.ndarray
            Array containing the indices of all neighboring fit solutions that should be used for the grouping.
        signal_ranges : list
            Nested list containing info about ranges of the spectrum that were estimated to contain signal. The goodness-of-fit calculations are only performed for the spectral channels within these ranges.
        noise_spike_ranges : list
            Nested list containing info about ranges of the spectrum that were estimated to contain noise spike features. These will get masked out from goodness-of-fit calculations.
        signal_mask : numpy.ndarray
            Boolean array containing the information of signal_ranges.
        interval : list
            List specifying the interval of spectral channels containing the flagged feature in the form of [lower, upper]. Only used in phase 2 of the spatially coherent refitting.
        n_centroids : int
            Number of centroid positions that should be present in interval.
        flag : str
            Flagged criterion that should be refit: 'broad', 'blended', or 'residual'.
        dct_new_fit : dict
            Only used in phase 2 of the spatially coherent refitting, in case the best fit solution was already updated in a previous iteration.

        Returns
        -------
        dictResults : dict
            Information about the new best fit solution in case of a successful refit. Otherwise 'None' is returned.
        refit : bool
            Information of whether there was a new successful refit.

        """
        dictComps = None
        refit = False

        #  sort neighboring fit solutions according to lowest value of reduced chi-square
        #  TODO: change this so that this gets sorted according to the lowest difference of the reduced chi-square values to the ideal value of 1 to prevent using fit solutions that 'overfit' the data
        sort = np.argsort(
            np.array(self.decomposition['best_fit_rchi2'])[indices_neighbors])

        for index_neighbor in indices_neighbors[sort]:
            #  check whether to use the neighboring fit solution or skip it
            if self.skip_index_for_refitting(index, index_neighbor):
                continue

            #  try to only replace part of the fit solution with new initial guesses from the neighboring fit solution for components flagged as broad, blended, or causing a negative residual feature. Otherwise use the entire fit solution of the neighboring spectrum.

            if flag in ['broad', 'blended', 'residual']:
                dictComps = self.replace_flagged_interval(
                    index, index_neighbor, spectrum, rms, flag=flag)
            elif interval is not None:
                dictComps = self.replace_flagged_interval(
                    index, index_neighbor, spectrum, rms, interval=interval,
                    dct_new_fit=dct_new_fit)
            else:
                dictComps = self.get_initial_values_from_neighbor(
                    index_neighbor, spectrum)

            if dictComps is None:
                continue

            #  try to refit with new fit solution

            dictResults = self.gaussian_fitting(
                spectrum, rms, dictComps, signal_ranges, noise_spike_ranges,
                signal_mask)
            refit = True
            if dictResults is None:
                continue
            if self.choose_new_fit(
                    dictResults, index, dct_new_fit=dct_new_fit,
                    interval=interval, n_centroids=n_centroids):
                return dictResults, refit

        return None, refit

    def get_refit_interval(self, spectrum, rms, amps, fwhms, means, flag):
        """Get interval of spectral channels containing flagged feature selected for refitting.

        Parameters
        ----------
        spectrum : numpy.ndarray
            Spectrum to refit.
        rms : float
            Root-mean-square noise value of the spectrum.
        amps : list
            List of amplitude values of the fitted components.
        fwhms : ist
            List of FWHM values of the fitted components.
        means : ist
            List of mean position values of the fitted components.
        flag : str
            Flagged criterion that should be refit: 'broad', 'blended', or 'residual'.

        Returns
        -------
        list
            List specifying the interval of spectral channels containing the flagged feature in the form of [lower, upper].

        """
        #  for component flagged as broad select the interval [mean - FWHM, mean + FWHM]
        if flag == 'broad':
            idx = np.argmax(np.array(fwhms))  # idx of broadest component
            lower = max(0, means[idx] - fwhms[idx])
            upper = means[idx] + fwhms[idx]
        #  for blended components get the interval containing all spectral channels within mean +/- FWHM intervals of all components flagged as blended
        elif flag == 'blended':
            params = amps + fwhms + means
            separation_factor = self.decomposition[
                'improve_fit_settings']['separation_factor']
            indices = get_fully_blended_gaussians(
                params, separation_factor=separation_factor)
            lower = max(0, min(
                np.array(means)[indices] - np.array(fwhms)[indices]))
            upper = max(
                np.array(means)[indices] + np.array(fwhms)[indices])
        #  for negative residual features get the mean +/- FWHM interval of the broadest component that overlaps with the location of the negative residual feature
        elif flag == 'residual':
            dct = self.decomposition['improve_fit_settings'].copy()

            best_fit_list = [None for _ in range(10)]
            best_fit_list[0] = amps + fwhms + means
            best_fit_list[2] = len(amps)
            residual = spectrum - combined_gaussian(
                amps, fwhms, means, self.channels)
            best_fit_list[4] = residual

            #  TODO: What if multiple negative residual features occur in one spectrum?
            idx = check_for_negative_residual(
                self.channels, spectrum, rms, best_fit_list, dct, get_idx=True)
            if idx is None:
                #  TODO: check if self.channels[-1] causes problems
                return [self.channels[0], self.channels[-1]]
            lower = max(0, means[idx] - fwhms[idx])
            upper = means[idx] + fwhms[idx]

        return [lower, upper]

    def replace_flagged_interval(self, index, index_neighbor, spectrum, rms,
                                 flag='none', interval=None, dct_new_fit=None):
        """Update initial guesses for fit components by replacing flagged feature with a neighboring fit solution.

        Parameters
        ----------
        index : int
            Index ('index_fit' keyword) of the spectrum that will be refit.
        index_neighbor : int
            Index ('index_fit' keyword) of the neighboring fit solution.
        spectrum : numpy.ndarray
            Spectrum to refit.
        rms : float
            Root-mean-square noise value of the spectrum.
        flag : str
            Flagged criterion that should be refit: 'broad', 'blended', or 'residual'.
        interval : list
            List specifying the interval of spectral channels containing the flagged feature in the form of [lower, upper].
        dct_new_fit : dict
            Only used in phase 2 of the spatially coherent refitting, in case the best fit solution was already updated in a previous iteration.

        Returns
        -------
        dictComps : dict
            Dictionary containing updated initial guesses for the fit solution.

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
            interval = self.get_refit_interval(
                spectrum, rms, amps, fwhms, means, flag=flag)
        indices, interval = self.components_in_interval(
            fwhms, means, interval)

        amps, fwhms, means = remove_components_from_sublists(
            [amps, fwhms, means], indices)
        amps_err, fwhms_err, means_err = remove_components_from_sublists(
            [amps_err, fwhms_err, means_err], indices)

        #  get new initial guess(es) for removed component(s) from neighboring fit solution

        amps_new = self.decomposition['amplitudes_fit'][index_neighbor]
        fwhms_new = self.decomposition['fwhms_fit'][index_neighbor]
        means_new = self.decomposition['means_fit'][index_neighbor]

        amps_err_new = self.decomposition['amplitudes_fit'][index_neighbor]
        fwhms_err_new = self.decomposition['fwhms_fit'][index_neighbor]
        means_err_new = self.decomposition['means_fit'][index_neighbor]

        #  check which of the neighboring fit components overlap with the interval containing the flagged feature(s)
        indices, interval = self.components_in_interval(
            fwhms_new, means_new, interval)

        if len(indices) == 0:
            return None

        #  discard all neighboring fit components not overlappting with the interval containing the flagged feature(s)
        remove_indices = np.delete(np.arange(len(amps_new)), indices)
        amps_new, fwhms_new, means_new = remove_components_from_sublists(
            [amps_new, fwhms_new, means_new], remove_indices)
        amps_err_new, fwhms_err_new, means_err_new =\
            remove_components_from_sublists(
                [amps_err_new, fwhms_err_new, means_err_new], remove_indices)

        if len(amps_new) == 0:
            return None

        #  get best fit with new fit solution(s) for only the interval that contained the removed components

        idx_lower = int(interval[0])
        idx_upper = int(interval[1]) + 2

        dictCompsInterval = {}
        for amp, fwhm, mean, mean_err in zip(
                amps_new, fwhms_new, means_new, means_err_new):
            dictCompsInterval = self.add_initial_value_to_dict(
                dictCompsInterval, spectrum[idx_lower:idx_upper], amp,
                fwhm, mean - idx_lower, mean_err)

        channels = np.arange(len(spectrum[idx_lower:idx_upper]))

        dictFit = self.gaussian_fitting(
            spectrum[idx_lower:idx_upper], rms, dictCompsInterval, None, None,
            None, params_only=True, channels=channels)

        if dictFit is None:
            return None

        #  create new dictionary of fit solution(s) by combining new fit component(s) taken from neighboring spectrum with the remaining fit component(s) outside the flagged interval

        dictComps = {}
        for amp, fwhm, mean, mean_err in zip(
                dictFit['amplitudes_fit'], dictFit['fwhms_fit'],
                dictFit['means_fit'], dictFit['means_fit_err']):
            dictComps = self.add_initial_value_to_dict(
                dictComps, spectrum, amp, fwhm, mean + idx_lower, mean_err)

        for amp, fwhm, mean, mean_err in zip(amps, fwhms, means, means_err):
            dictComps = self.add_initial_value_to_dict(
                dictComps, spectrum, amp, fwhm, mean, mean_err)

        return dictComps

    def components_in_interval(self, fwhms, means, interval):
        """Find indices of components overlapping with the interval and update the interval range to accommodate full extent of the components.

        Component i is selected if means[i] +/- fwhms[i] overlaps with the
        interval.

        The interval is updated to accommodate all spectral channels contained in the range means[i] +/- fwhms[i].

        Parameters
        ----------
        fwhms : list
            List of FWHM values of fit components.
        means : list
            List of mean position values of fit components.
        interval : list
            List specifying the interval of spectral channels containing the flagged feature in the form of [lower, upper].

        Returns
        -------
        indices : list
            List with indices of components overlapping with interval.
        interval_new : list
            Updated interval that accommodates all spectral channels contained in the range means[i] +/- fwhms[i].

        """
        lower_interval, upper_interval = interval.copy()
        lower_interval_new, upper_interval_new = interval.copy()
        indices = []

        for i, (mean, fwhm) in enumerate(zip(means, fwhms)):
            lower = max(0, mean - fwhm)
            upper = mean + fwhm
            if (lower_interval <= lower <= upper_interval) or\
                    (lower_interval <= upper <= upper_interval):
                lower_interval_new = min(lower_interval_new, lower)
                upper_interval_new = max(upper_interval_new, upper)
                indices.append(i)
        return indices, [lower_interval_new, upper_interval_new]

    def add_initial_value_to_dict(self, dictComps, spectrum,
                                  amp, fwhm, mean, mean_err):
        """Update dictionary of fit components with new component.

        Parameters
        ----------
        dictComps : dict
            Dictionary of fit components.
        spectrum : numpy.ndarray
            Spectrum to refit.
        amp : float
            Amplitude value of fit component.
        fwhm : type
            FWHM value of fit component.
        mean : float
            Mean position value of fit component.
        mean_err : float
            Error of mean position value of fit component.

        Returns
        -------
        dictComps : dict
            Updated dictionary of fit components.

        """
        stddev = fwhm / 2.354820045

        #  determine upper limit for amplitude value
        idx_low = max(0, int(mean - stddev))
        idx_upp = int(mean + stddev) + 2
        amp_max = np.max(spectrum[idx_low:idx_upp])

        #  determine lower and upper limits for mean position
        #  TODO: add here also mean +/- stddev??
        mean_min = min(mean - self.mean_separation, mean - mean_err)
        mean_min = max(0, mean_min)  # prevent negative values
        mean_max = max(mean + self.mean_separation, mean + mean_err)

        # fwhm_min = max(0., fwhm - self.fwhm_separation)
        # fwhm_max = fwhm + self.fwhm_separation

        keyname = str(len(dictComps) + 1)
        dictComps[keyname] = {}
        dictComps[keyname]['amp_ini'] = amp
        dictComps[keyname]['mean_ini'] = mean
        dictComps[keyname]['fwhm_ini'] = fwhm

        dictComps[keyname]['amp_bounds'] = [0., 1.1*amp_max]
        dictComps[keyname]['mean_bounds'] = [mean_min, mean_max]
        dictComps[keyname]['fwhm_bounds'] = [0., None]
        # dictComps[keyname]['fwhm_bounds'] = [fwhm_min, fwhm_max]

        return dictComps

    def get_dictionary_value(self, key, index, dct_new_fit=None):
        """Return a dictionary value.

        Parameters
        ----------
        key : str
            Key of the dictionary.
        index : int
            Index ('index_fit' keyword) of the spectrum that gets/was refit.
        dct_new_fit : dict
            If this dictionary is supplied, the value is extracted from it (only used in phase 2 of the spatially coherent refitting); otherwise the value is extracted from the 'self.decomposition' dictionary

        """
        if dct_new_fit is not None:
            return dct_new_fit[key]
        else:
            return self.decomposition[key][index]

    def get_flags(self, dictResults, index, key='None', flag=None,
                  dct_new_fit=None):
        """Check how the refit affected the number of blended or negative residual features.

        This check will only be performed if the 'self.flag_blended=True' or 'self.flag_neg_res_peak=True'.

        Parameters
        ----------
        dictResults : dict
            Dictionary containing the new best fit results after the refit attempt.
        index : int
            Index ('index_fit' keyword) of the spectrum that gets/was refit.
        key : str
            Dictionary keys, either 'N_blended' or 'N_neg_res_peak'.
        flag : bool
            User-selected flag criterion, either 'self.flag_blended', or 'self.flag_neg_res_peak'
        dct_new_fit : dict
            Only used in phase 2 of the spatially coherent refitting, in case the best fit solution was already updated in a previous iteration.

        Returns
        -------
        flag_old : int
            Count of flagged features present in spectrum before refit.
        flag_new : int
            Count of flagged features present in spectrum after refit.

        """
        flag_old, flag_new = (0 for _ in range(2))

        if not flag:
            return flag_old, flag_new

        n_old = self.get_dictionary_value(key, index, dct_new_fit=dct_new_fit)
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

    def get_flags_rchi2(self, dictResults, index, dct_new_fit=None):
        """Check how the reduced chi-square value of a spectrum changed after the refit.

        This check will only be performed if the 'self.flag_rchi2=True'.

        Parameters
        ----------
        dictResults : dict
            Dictionary containing the new best fit results after the refit attempt.
        index : int
            Index ('index_fit' keyword) of the spectrum that gets/was refit.
        dct_new_fit : dict
            Only used in phase 2 of the spatially coherent refitting, in case the best fit solution was already updated in a previous iteration.

        Returns
        -------
        flag_old : int
            Flag value before the refit.
        flag_new : int
            Flag value after the refit.

        """
        flag_old, flag_new = (0 for _ in range(2))

        if not self.flag_rchi2:
            return flag_old, flag_new

        rchi2_old = self.get_dictionary_value(
            'best_fit_rchi2', index, dct_new_fit=dct_new_fit)
        rchi2_new = dictResults['best_fit_rchi2']

        if rchi2_old > self.rchi2_limit:
            flag_old += 1
        if rchi2_new > self.rchi2_limit:
            flag_new += 1

        #  reward new fit if it is closer to rchi2 = 1 and thus likely less "overfit"
        if max(rchi2_old, rchi2_new) < self.rchi2_limit:
            if abs(rchi2_new - 1) < abs(rchi2_old - 1):
                flag_old += 1

        return flag_old, flag_new

    def get_flags_pvalue(self, dictResults, index, dct_new_fit=None):
        flag_old, flag_new = (0 for _ in range(2))

        if not self.flag_residual:
            return flag_old, flag_new

        pvalue_old = self.get_dictionary_value(
            'pvalue', index, dct_new_fit=dct_new_fit)
        pvalue_new = dictResults['pvalue']

        if pvalue_old < self.min_pvalue:
            flag_old += 1
        if pvalue_new < self.min_pvalue:
            flag_new += 1

        #  punish fit if pvalue got worse
        if pvalue_new < pvalue_old:
            flag_new += 1

        return flag_old, flag_new

    def get_flags_broad(self, dictResults, index, dct_new_fit=None):
        """Check how the refit affected the number of components flagged as broad.

        This check will only be performed if the 'self.flag_broad=True'.

        Parameters
        ----------
        dictResults : dict
            Dictionary containing the new best fit results after the refit attempt.
        index : int
            Index ('index_fit' keyword) of the spectrum that gets/was refit.
        dct_new_fit : dict
            Only used in phase 2 of the spatially coherent refitting, in case the best fit solution was already updated in a previous iteration.

        Returns
        -------
        flag_old : int
            Flag value before the refit.
        flag_new : int
            Flag value after the refit.

        """
        flag_old, flag_new = (0 for _ in range(2))

        if not self.flag_broad:
            return flag_old, flag_new

        if self.mask_broad_flagged[index]:
            flag_old = 1
            fwhm_max_old = max(self.get_dictionary_value(
                'fwhms_fit', index, dct_new_fit=dct_new_fit))
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
                if (fwhms[-1] > self.fwhm_factor * fwhms[-2]) and\
                        (fwhms[-1] - fwhms[-2]) > self.fwhm_separation:
                    flag_new = 1

        return flag_old, flag_new

    def get_flags_ncomps(self, dictResults, index, dct_new_fit=None):
        """Check how the number of component jumps changed after the refit.

        Parameters
        ----------
        dictResults : dict
            Dictionary containing the new best fit results after the refit attempt.
        index : int
            Index ('index_fit' keyword) of the spectrum that gets/was refit.
        dct_new_fit : dict
            Only used in phase 2 of the spatially coherent refitting, in case the best fit solution was already updated in a previous iteration.

        Returns
        -------
        flag_old : int
            Flag value before the refit.
        flag_new : int
            Flag value after the refit.

        """
        flag_old, flag_new = (0 for _ in range(2))

        if not self.flag_ncomps:
            return flag_old, flag_new

        njumps_old = self.ncomps_jumps[index]

        loc = self.location[index]
        indices = get_neighbors(
            loc, exclude_p=True, shape=self.shape, nNeighbors=1,
            get_indices=True)
        mask_indices = get_neighbors(
            loc, exclude_p=True, shape=self.shape, nNeighbors=1,
            get_mask=True)

        ncomps = np.ones(8) * np.nan
        ncomps[mask_indices] = self.ncomps[indices]
        ncomps_central = self.get_dictionary_value(
             'N_components', index, dct_new_fit=dct_new_fit)
        ncomps = np.insert(ncomps, 4, ncomps_central)
        njumps_new = self.number_of_component_jumps(ncomps)

        ncomps_wmedian = self.ncomps_wmedian[index]
        ndiff_old = abs(ncomps_wmedian - self.ncomps[index])
        ndiff_new = abs(ncomps_wmedian - ncomps_central)

        if (njumps_old > self.n_max_jump_comps) or (
                ndiff_old > self.max_diff_comps):
            flag_old = 1
        if (njumps_new > self.n_max_jump_comps) or (
                ndiff_new > self.max_diff_comps):
            flag_new = 1
        if (njumps_new > njumps_old) or (ndiff_new > ndiff_old):
            flag_new += 1

        return flag_old, flag_new

    def get_flags_centroids(self, dictResults, index, dct_new_fit=None,
                            interval=None, n_centroids=None):
        """Check how the presence of centroid positions changed after the refit.

        This check is only performed in phase 2 of the spatially coherent refitting.

        Parameters
        ----------
        dictResults : dict
            Dictionary containing the new best fit results after the refit attempt.
        index : int
            Index ('index_fit' keyword) of the spectrum that gets/was refit.
        dct_new_fit : dict
            Only used in phase 2 of the spatially coherent refitting, in case the best fit solution was already updated in a previous iteration.
        interval : list
            List specifying the interval of spectral channels where 'n_centroids' number of centroid positions are required.
        n_centroids : int
            Number of centroid positions that should be present in interval.

        Returns
        -------
        flag_old : int
            Flag value before the refit.
        flag_new : int
            Flag value after the refit.

        """
        flag_old, flag_new = (0 for _ in range(2))

        if interval is None:
            return flag_old, flag_new

        means_old = self.get_dictionary_value(
            'means_fit', index, dct_new_fit=dct_new_fit)
        means_new = dictResults['means_fit']

        flag_old, flag_new = (2 for _ in range(2))

        n_centroids_old = self.number_of_values_in_interval(means_old, interval)
        n_centroids_new = self.number_of_values_in_interval(means_new, interval)

        #  reward new fit if it has the required number of centroid positions within 'interval'
        if n_centroids_new == n_centroids:
            flag_new = 0
        #  reward new fit if its number of centroid positions within 'interval' got closer to the required value
        elif abs(n_centroids_new - n_centroids) < abs(
                n_centroids_old - n_centroids):
            flag_new = 1
        #  punish new fit if its number of centroid positions within 'interval' compared to the required value got worse than in the old fit
        elif abs(n_centroids_new - n_centroids) > abs(
                n_centroids_old - n_centroids):
            flag_old = 1

        return flag_old, flag_new

    def choose_new_fit(self, dictResults, index, dct_new_fit=None,
                       interval=None, n_centroids=None):
        """Decide whether to accept the new fit solution as the new best fit.

        Parameters
        ----------
        dictResults : dict
            Dictionary containing the new best fit results after the refit attempt.
        index : int
            Index ('index_fit' keyword) of the spectrum that gets/was refit.
        dct_new_fit : dict
            Only used in phase 2 of the spatially coherent refitting, in case the best fit solution was already updated in a previous iteration.
        interval : list
            List specifying the interval of spectral channels where 'n_centroids' number of centroid positions are required.
        n_centroids : int
            Number of centroid positions that should be present in interval.

        Returns
        -------
        bool
            Decision of whether new fit solution gets accepted as new best fit.

        """
        #  check how values/numbers of flagged features changed after the refit

        flag_blended_old, flag_blended_new = self.get_flags(
            dictResults, index, key='N_blended', flag=self.flag_blended,
            dct_new_fit=dct_new_fit)

        flag_neg_res_peak_old, flag_neg_res_peak_new = self.get_flags(
            dictResults, index, key='N_neg_res_peak',
            flag=self.flag_neg_res_peak, dct_new_fit=dct_new_fit)

        flag_rchi2_old, flag_rchi2_new = self.get_flags_rchi2(
            dictResults, index, dct_new_fit=dct_new_fit)

        flag_residual_old, flag_residual_new = self.get_flags_pvalue(
            dictResults, index, dct_new_fit=dct_new_fit)

        flag_broad_old, flag_broad_new = self.get_flags_broad(
            dictResults, index, dct_new_fit=dct_new_fit)

        flag_ncomps_old, flag_ncomps_new = self.get_flags_ncomps(
            dictResults, index, dct_new_fit=dct_new_fit)

        flag_centroids_old, flag_centroids_new = self.get_flags_centroids(
            dictResults, index, dct_new_fit=dct_new_fit,
            interval=interval, n_centroids=n_centroids)

        #  only for phase 2: do not accept the new fit if there was no improvement for the centroid positions required in 'interval'
        if (n_centroids is not None) and (flag_centroids_new > 1):
            return False

        #  compute total flag values

        n_flags_old = flag_blended_old\
            + flag_neg_res_peak_old\
            + flag_broad_old\
            + flag_rchi2_old\
            + flag_residual_old\
            + flag_ncomps_old\
            + flag_centroids_old

        n_flags_new = flag_blended_new\
            + flag_neg_res_peak_new\
            + flag_broad_new\
            + flag_rchi2_new\
            + flag_residual_new\
            + flag_ncomps_new\
            + flag_centroids_new

        #  do not accept new fit if the total flag value increased
        if n_flags_new > n_flags_old:
            return False

        #  if total flag value is the same or decreased there are two ways for the new best fit to get accepted as the new best fit solution:
        #  - accept the new fit if its AICc value is lower than AICc value of the current best fit solution
        # - if the AICc value of new fit is higher than the AICc value of the current best fit solution, only accept the new fit if the values of the residual are normally distributed, i.e. if it passes the Kolmogorov-Smirnov test

        aicc_old = self.get_dictionary_value(
            'best_fit_aicc', index, dct_new_fit=dct_new_fit)
        aicc_new = dictResults['best_fit_aicc']
        # residual_signal_mask = dictResults['residual_signal_mask']
        pvalue = dictResults['pvalue']

        if (aicc_new > aicc_old) and (pvalue < self.min_pvalue):
            return False

        return True

    def get_initial_values(self, indices_neighbors):
        """Get sorted parameter values (amps, means, fwhms) of neighboring fit components for the grouping.

        Parameters
        ----------
        indices_neighbors : numpy.ndarray
            Array containing the indices of all neighboring fit solutions that should be used for the grouping. in the form [idx1, ..., idxN].

        Returns
        -------
        amps : numpy.ndarray
            Array of amplitude values (sorted according to mean position).
        means : numpy.ndarray
            Array of sorted mean position values.
        fwhms : numpy.ndarray
            Array of FWHM values (sorted according to mean position).

        """
        amps, means, fwhms = (np.array([]) for i in range(3))

        for i in indices_neighbors:
            amps = np.append(amps, self.decomposition['amplitudes_fit'][i])
            means = np.append(means, self.decomposition['means_fit'][i])
            fwhms = np.append(fwhms, self.decomposition['fwhms_fit'][i])

        sorted_indices = np.argsort(means)
        return amps[sorted_indices], means[sorted_indices], fwhms[sorted_indices]

    def grouping(self, amps_tot, means_tot, fwhms_tot, split_fwhm=True):
        """Grouping according to mean position values only or mean position values and FWHM values.

        Parameters
        ----------
        amps_tot : numpy.ndarray
            Array of amplitude values (sorted according to mean position).
        means_tot : numpy.ndarray
            Array of sorted mean position values.
        fwhms_tot : numpy.ndarray
            Array of FWHM values (sorted according to mean position).
        split_fwhm : bool
            Whether to group according to mean position and FWHM values ('True') or only according to mean position values ('False').

        Returns
        -------
        dictCompsOrdered : collections.OrderedDict
            Ordered dictionary containing the results of the grouping.

        """
        #  group with regards to mean positions only
        means_diff = np.append(np.array([0.]), means_tot[1:] - means_tot[:-1])

        split_indices = np.where(means_diff > self.mean_separation)[0]
        split_means_tot = np.split(means_tot, split_indices)
        split_fwhms_tot = np.split(fwhms_tot, split_indices)
        split_amps_tot = np.split(amps_tot, split_indices)

        dictComps = {}

        for amps, fwhms, means in zip(
                split_amps_tot, split_fwhms_tot, split_means_tot):
            if (len(means) == 1) or not split_fwhm:
                key = "{}".format(len(dictComps) + 1)
                dictComps[key] = {
                    "amps": amps, "means": means, "fwhms": fwhms}
                continue

            #  also group with regards to FWHM values

            lst_of_grouped_indices = []
            for i in range(len(means)):
                grouped_indices_means = np.where(
                    (np.abs(means - means[i]) < self.mean_separation))[0]
                grouped_indices_fwhms = np.where(
                    (np.abs(fwhms - fwhms[i]) < self.fwhm_separation))[0]
                ind = np.intersect1d(
                    grouped_indices_means, grouped_indices_fwhms)
                lst_of_grouped_indices.append(list(ind))

            #  merge all sublists from lst_of_grouped_indices that share common indices

            G = to_graph(lst_of_grouped_indices)
            lst = list(connected_components(G))
            lst = [list(l) for l in lst]

            for sublst in lst:
                key = "{}".format(len(dictComps) + 1)
                dictComps[key] = {"amps": amps[sublst],
                                  "means": means[sublst],
                                  "fwhms": fwhms[sublst]}

        dictCompsOrdered = collections.OrderedDict()
        for i, k in enumerate(sorted(dictComps,
                                     key=lambda k: len(dictComps[k]['amps']),
                                     reverse=True)):
            dictCompsOrdered[str(i + 1)] = dictComps[k]

        return dictCompsOrdered

    def get_initial_values_from_neighbor(self, i, spectrum):
        """Get dictionary with information about all fit components from neighboring fit solution.

        Parameters
        ----------
        i : int
            Index of neighboring fit solution.
        spectrum : numpy.ndarray
            Spectrum to refit.

        Returns
        -------
        dictComps : dict
            Dictionary containing information about all fit components from neighboring fit solution.

        """
        dictComps = {}

        for key in range(self.decomposition['N_components'][i]):
            amp = self.decomposition['amplitudes_fit'][i][key]
            mean = self.decomposition['means_fit'][i][key]
            mean_err = self.decomposition['means_fit_err'][i][key]
            fwhm = self.decomposition['fwhms_fit'][i][key]
            stddev = fwhm / 2.354820045

            idx_low = max(0, int(mean - stddev))
            idx_upp = int(mean + stddev) + 2
            amp_max = np.max(spectrum[idx_low:idx_upp])

            mean_min = min(mean - self.mean_separation, mean - mean_err)
            mean_min = max(0, mean_min)  # prevent negative values
            mean_max = max(mean + self.mean_separation, mean + mean_err)

            keyname = str(key + 1)
            dictComps[keyname] = {}
            dictComps[keyname]['amp_ini'] = amp
            dictComps[keyname]['mean_ini'] = mean
            dictComps[keyname]['fwhm_ini'] = fwhm

            dictComps[keyname]['amp_bounds'] = [0., 1.1*amp_max]
            dictComps[keyname]['mean_bounds'] = [mean_min, mean_max]
            dictComps[keyname]['fwhm_bounds'] = [0., None]

        return dictComps

    def determine_average_values(self, spectrum, rms, dictComps):
        """Determine average values for fit components obtained by grouping.

        Parameters
        ----------
        spectrum : numpy.ndarray
            Spectrum to refit.
        rms : float
            Root-mean-square noise value of the spectrum.
        dictComps : collections.OrderedDict
            Ordered dictionary containing results of the grouping.

        Returns
        -------
        dictComps : collections.OrderedDict
            Updated ordered dictionary containing average values for the fit components obtained via the grouping.

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
            stddev_ini = fwhm_ini / 2.354820045

            # TODO: change stddev_ini to fwhm_ini?
            idx_low = max(0, int(mean_ini - stddev_ini))
            idx_upp = int(mean_ini + stddev_ini) + 2

            amp_max = np.max(spectrum[idx_low:idx_upp])
            if amp_max < self.snr*rms:
                dictComps.pop(key)
                continue

            #  determine fitting constraints for mean value
            lower_interval = max(
                abs(mean_ini - np.min(means)), self.mean_separation)
            mean_min = max(0, mean_ini - lower_interval)

            upper_interval = max(
                abs(mean_ini - np.max(means)), self.mean_separation)
            mean_max = mean_ini + upper_interval

            # #  determine fitting constraints for fwhm value
            # lower_interval = max(
            #     abs(fwhm_ini - np.min(fwhms)), self.fwhm_separation)
            # fwhm_min = max(self.min_fwhm, fwhm_ini - lower_interval)
            #
            # upper_interval = max(
            #     abs(fwhm_ini - np.max(fwhms)), self.fwhm_separation)
            # fwhm_max = fwhm_ini + upper_interval

            dictComps[key]['amp_ini'] = amp_ini
            dictComps[key]['mean_ini'] = mean_ini
            dictComps[key]['fwhm_ini'] = fwhm_ini

            dictComps[key]['amp_bounds'] = [0., 1.1*amp_max]
            dictComps[key]['mean_bounds'] = [mean_min, mean_max]
            dictComps[key]['fwhm_bounds'] = [0., None]
            # dictComps[key]['fwhm_bounds'] = [fwhm_min, fwhm_max]
        return dictComps

    def gaussian_fitting(self, spectrum, rms, dictComps, signal_ranges,
                         noise_spike_ranges, signal_mask, params_only=False,
                         channels=None):
        """Perform a new Gaussian decomposition with updated initial guesses.

        Parameters
        ----------
        spectrum : numpy.ndarray
            Spectrum to refit.
        rms : float
            Root-mean-square noise value of the spectrum.
        dictComps : dict
            Dictionary containing information about new initial guesses for fit components.
        signal_ranges : list
            Nested list containing info about ranges of the spectrum that were estimated to contain signal. The goodness-of-fit calculations are only performed for the spectral channels within these ranges.
        noise_spike_ranges : list
            Nested list containing info about ranges of the spectrum that were estimated to contain noise spike features. These will get masked out from goodness-of-fit calculations.
        signal_mask : numpy.ndarray
            Boolean array containing the information of signal_ranges.
        params_only : bool (default: 'False')
            If set to 'True', the returned dictionary of the fit results will only contain information about the amplitudes, FWHM values and mean positions of the fitted components.
        channels : numpy.ndarray
            Array containing the number of spectral channels.

        Returns
        -------
        dictResults : dictionary
            Dictionary containing information about the fit results.

        """
        if channels is None:
            n_channels = self.n_channels
            channels = self.channels
        else:
            n_channels = len(channels)

        if noise_spike_ranges:
            noise_spike_mask = mask_channels(
                n_channels, [[0, n_channels]],
                remove_intervals=noise_spike_ranges)
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
                params.append(dictComps[nr]['{}_ini'.format(key)])
                params_min.append(dictComps[nr]['{}_bounds'.format(key)][0])
                params_max.append(dictComps[nr]['{}_bounds'.format(key)][1])

        #  get new best fit
        best_fit_list = get_best_fit(
            channels, spectrum, errors, params, dct, first=True,
            signal_ranges=signal_ranges, signal_mask=signal_mask,
            params_min=params_min, params_max=params_max,
            noise_spike_mask=noise_spike_mask)

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
                channels, spectrum, errors, best_fit_list, dct,
                fitted_residual_peaks, signal_ranges=signal_ranges,
                signal_mask=signal_mask, noise_spike_mask=noise_spike_mask)
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

        amps, fwhms, means = split_params(params, ncomps)
        amps_errs, fwhms_errs, means_errs = split_params(params_errs, ncomps)

        keys = ['amplitudes_fit', 'fwhms_fit', 'means_fit',
                'amplitudes_fit_err', 'fwhms_fit_err', 'means_fit_err']
        vals = [amps, fwhms, means, amps_errs, fwhms_errs, means_errs]
        dictResults = {key: val for key, val in zip(keys, vals)}

        if params_only:
            return dictResults

        mask = mask_covering_gaussians(
            means, fwhms, n_channels, remove_intervals=noise_spike_ranges)
        rchi2_gauss, aicc_gauss = goodness_of_fit(
            spectrum, best_fit, rms, ncomps, mask=mask, get_aicc=True)

        N_blended = get_fully_blended_gaussians(
            params, get_count=True, separation_factor=self.decomposition[
                'improve_fit_settings']['separation_factor'])
        N_neg_res_peak = check_for_negative_residual(
            channels, spectrum, rms, best_fit_list, dct, get_count=True)

        keys = ["best_fit_rchi2", "best_fit_aicc", "residual_signal_mask",
                "gaussians_rchi2", "gaussians_aicc", "pvalue",
                "N_components", "N_blended", "N_neg_res_peak"]
        values = [rchi2, aicc, residual_signal_mask,
                  rchi2_gauss, aicc_gauss, pvalue,
                  ncomps, N_blended, N_neg_res_peak]
        for key, val in zip(keys, values):
            dictResults[key] = val

        return dictResults

    def save_final_results(self):
        """Save the results of the spatially coherent refitting iterations."""
        pathToFile = os.path.join(
            self.decomp_dirname, '{}.pickle'.format(self.fin_filename))
        pickle.dump(self.decomposition, open(pathToFile, 'wb'), protocol=2)
        say("\033[92mSAVED FILE:\033[0m '{}' in '{}'".format(
            self.fin_filename, self.decomp_dirname), logger=self.logger)

    # def say(self, message):
    #     """Print diagnostic messages to terminal."""
    #     if self.log_output:
    #         self.logger.info(message)
    #     if self.verbose:
    #         print(message)

    #
    #  --- Phase 2: Refitting towards coherence in centroid positions ---
    #

    def get_centroid_interval(self, dct):
        """Calculate the interval spanned by each group of centroids.

        Parameters
        ----------
        dct : dict
            Dictionary containing information of the grouping for phase 2 of the spatially coherent refitting.

        Returns
        -------
        dct : dict
            Updated dictionary containing information about the interval spanned by each group of centroids.

        """
        dct['means_interval'] = {}
        for key in dct['grouping']:
            mean_min = min(dct['grouping'][key]['means'])
            mean_min = max(0, mean_min - self.mean_separation / 2)
            mean_max = max(dct['grouping'][key]['means'])\
                + self.mean_separation / 2
            dct['means_interval'][key] = [mean_min, mean_max]
        return dct

    def components_per_interval(self, dct):
        """Calculate how many components neighboring fits had per grouped centroid interval."""
        dct['ncomps_per_interval'] = {}
        for key in dct['grouping']:
            ncomps_per_interval = []
            means_interval = dct['means_interval'][key]

            for idx in dct['indices_neighbors']:
                means = self.decomposition['means_fit'][idx]
                if means is None:
                    ncomps_per_interval.append(0)
                    continue
                if len(means) == 0:
                    ncomps_per_interval.append(0)
                    continue
                condition_1 = means_interval[0] < np.array(means)
                condition_2 = means_interval[1] > np.array(means)
                mask = np.logical_and(condition_1, condition_2)
                ncomps_per_interval.append(np.count_nonzero(mask))
            dct['ncomps_per_interval'][key] = ncomps_per_interval

        return dct

    def get_n_centroid(self, n_centroids, weights):
        """Calculate expected value for number of centroids per grouped centroid interval."""
        choices = list(set(n_centroids))
        #
        #  first, check only immediate neighboring spectra
        #
        mask_weight = weights >= self.w_min
        n_neighbors = np.count_nonzero(mask_weight)

        counts_choices = []
        for choice in choices:
            if choice == 0:
                counts_choices.append(0)
                continue
            count_choice = np.count_nonzero(n_centroids[mask_weight] == choice)
            counts_choices.append(count_choice)

        if np.max(counts_choices) > 0.5 * n_neighbors:
            idx = np.argmax(counts_choices)
            return choices[idx]
        #
        #  include additional neighbors that are two pixels away
        #
        weights_choices = []
        for choice in choices:
            if choice == 0:
                weights_choices.append(0)
                continue
            mask = n_centroids == choice
            weights_choices.append(sum(weights[mask]))
        idx = np.argmax(weights_choices)
        return choices[idx]

    def compute_weights(self, dct, weights):
        """Calculate weight of required components per centroid interval."""
        dct['factor_required'] = {}
        dct['n_centroids'] = {}
        for key in dct['grouping']:
            array = np.array(dct['ncomps_per_interval'][key])
            dct['n_centroids'][key] = self.get_n_centroid(array, weights)
            array = array.astype('bool')
            dct['factor_required'][key] = sum(array * weights)
        return dct

    def sort_out_keys(self, dct):
        """Keep only centroid intervals that have a certain minimum weight."""
        dct_new = {}
        keys = ['indices_neighbors', 'weights', 'means_interval',
                'n_centroids', 'factor_required']
        dct_new = {key: {} for key in keys}
        dct_new['indices_neighbors'] = dct['indices_neighbors']
        dct_new['weights'] = dct['weights']

        means_interval = []
        for key in dct['factor_required']:
            if dct['factor_required'][key] > self.min_p:
                means_interval.append(dct['means_interval'][key])

        dct_new['means_interval'] = means_interval
        return dct_new

    def add_key_to_dict(self, dct, key='means_interval', val=None):
        """Add a new key number & value to an existing dictionary key."""
        key_new = str(len(dct[key]) + 1)
        dct[key][key_new] = val
        return dct

    def merge_dictionaries(self, dct_1, dct_2):
        """Merge two dictionaries to a single one and calculate new centroid intervals."""
        dct_merged = {key: {} for key in [
            'factor_required', 'n_centroids', 'means_interval']}

        for key in ['indices_neighbors', 'weights']:
            dct_merged[key] = []
            for dct in [dct_1, dct_2]:
                dct_merged[key] = np.append(dct_merged[key], dct[key])

        key = 'means_interval'
        intervals = dct_1[key] + dct_2[key]
        dct_merged[key] = self.merge_intervals(intervals)

        return dct_merged

    def merge_intervals(self, intervals):
        """Merge overlapping intervals.

        Original code by amon: https://codereview.stackexchange.com/questions/69242/merging-overlapping-intervals
        """
        sorted_by_lower_bound = sorted(intervals, key=lambda tup: tup[0])
        merged = []

        for higher in sorted_by_lower_bound:
            if not merged:
                merged.append(higher)
            else:
                lower = merged[-1]
                # test for intersection between lower and higher:
                # we know via sorting that lower[0] <= higher[0]
                if higher[0] <= lower[1]:
                    upper_bound = max(lower[1], higher[1])
                    merged[-1] = (lower[0], upper_bound)  # replace by merged interval
                else:
                    merged.append(higher)
        return merged

    def combine_directions(self, dct):
        """Combine directions and get master dictionary."""
        dct_hv = self.merge_dictionaries(
            dct['horizontal'].copy(), dct['vertical'].copy())

        dct_dd = self.merge_dictionaries(
            dct['diagonal_ul'].copy(), dct['diagonal_ur'].copy())

        dct_total = self.merge_dictionaries(dct_hv, dct_dd)

        intervals = dct_total['means_interval'].copy()
        dct_total['means_interval'] = {}
        for interval in intervals:
            dct_total = self.add_key_to_dict(
                dct_total, key='means_interval', val=interval)

        #  add buffer of half the mean_separation to left and right of means_interval
        for key in dct_total['means_interval']:
            lower, upper = dct_total['means_interval'][key]
            lower = max(0, lower - self.mean_separation / 2)
            upper = upper + self.mean_separation / 2
            dct_total['means_interval'][key] = [lower, upper]

        # for key in ['indices_neighbors', 'weights']: # BUG!
        for key in ['indices_neighbors']:
            dct_total[key] = dct_total[key].astype('int')
        #
        #  Calculate number of centroids per centroid interval of neighbors
        #  and estimate the expected number of centroids for interval
        #
        dct_total['n_comps'] = {}
        dct_total['n_centroids'] = {}
        for key in dct_total['means_interval']:
            dct_total['n_comps'][key] = []
            lower, upper = dct_total['means_interval'][key]
            for idx in dct_total['indices_neighbors']:
                means = self.decomposition['means_fit'][idx]
                ncomps = 0
                for mean in means:
                    if lower < mean < upper:
                        ncomps += 1
                dct_total['n_comps'][key].append(ncomps)
            dct_total['n_centroids'][key] = self.get_n_centroid(
                 np.array(dct_total['n_comps'][key]), dct_total['weights'])

        return dct_total

    def get_weights(self, indices, idx, direction):
        """Allocate weights to neighboring spectra."""
        chosen_weights, indices_neighbors = ([] for _ in range(2))
        possible_indices = np.arange(len(indices))
        index = np.where(indices == idx)[0][0]
        indices = np.delete(indices, index)
        possible_indices -= index
        possible_indices = np.delete(possible_indices, index)

        if direction in ['horizontal', 'vertical']:
            weights = np.array([self.w_2, self.w_1, self.w_1, self.w_2])
        else:
            weights = np.array([self.w_2 / np.sqrt(8), self.w_1 / np.sqrt(2),
                                self.w_1 / np.sqrt(2), self.w_2 / np.sqrt(8)])

        counter = 0
        for i, weight in zip([-2, -1, 1, 2], weights):
            if i in possible_indices:
                idx_neighbor = indices[counter]
                counter += 1
                if self.decomposition['N_components'][idx_neighbor] is not None:
                    if self.decomposition['N_components'][idx_neighbor] != 0:
                        indices_neighbors.append(idx_neighbor)
                        chosen_weights.append(weight)
        return np.array(indices_neighbors), np.array(chosen_weights)

    def check_continuity_centroids(self, idx, loc):
        """Check for coherence of centroid positions of neighboring spectra.

        See Sect. 3.3.2. and Fig. 10 in Riener+ 2019 for more details.

        Parameters
        ----------
        idx : int
            Index ('index_fit' keyword) of the central spectrum.
        loc : tuple
            Location (ypos, xpos) of the central spectrum.

        Returns
        -------
        dct_total : dict
            Dictionary containing results of the spatial coherence check for neighboring centroid positions.

        """
        dct, dct_total = [{} for _ in range(2)]

        for direction in [
                'horizontal', 'vertical', 'diagonal_ul', 'diagonal_ur']:
            indices_neighbors_and_central = get_neighbors(
                loc, exclude_p=False, shape=self.shape, nNeighbors=2,
                direction=direction, get_indices=True)

            indices_neighbors, weights = self.get_weights(
                indices_neighbors_and_central, idx, direction)

            if len(indices_neighbors) == 0:
                dct_total[direction] = {key: {} for key in [
                    'indices_neighbors', 'weights', 'means_interval',
                    'n_centroids', 'factor_required']}
                dct_total[direction]['means_interval'] = []
                dct_total[direction]['indices_neighbors'] = np.array([])
                dct_total[direction]['weights'] = np.array([])
                continue

            dct['indices_neighbors'] = indices_neighbors
            dct['weights'] = weights

            amps, means, fwhms = self.get_initial_values(indices_neighbors)
            dct['grouping'] = self.grouping(
                amps, means, fwhms, split_fwhm=False)
            dct = self.get_centroid_interval(dct)
            dct = self.components_per_interval(dct)
            dct = self.compute_weights(dct, weights)
            dct = self.sort_out_keys(dct)

            #  TODO: check why copy() is needed here
            dct_total[direction] = dct.copy()

        dct_total = self.combine_directions(dct_total)
        return dct_total

    def check_for_required_components(self, idx, dct):
        """Check the presence of the required centroid positions within the determined interval."""
        dct_refit = {key: {} for key in ['n_centroids', 'means_interval']}
        for key in ['indices_neighbors', 'weights']:
            dct_refit[key] = dct[key]
        means = self.decomposition['means_fit'][idx]
        for key in dct['means_interval']:
            ncomps_expected = dct['n_centroids'][key]
            interval = dct['means_interval'][key]
            ncomps = self.number_of_values_in_interval(means, interval)
            if ncomps != ncomps_expected:
                dct_refit = self.add_key_to_dict(
                    dct_refit, key='n_centroids', val=ncomps_expected)
                dct_refit = self.add_key_to_dict(
                    dct_refit, key='means_interval', val=interval)
        return dct_refit

    def number_of_values_in_interval(self, lst, interval):
        """Return number of points in list that are located within the interval."""
        lower, upper = interval
        array = np.array(lst)
        mask = np.logical_and(array > lower, array < upper)
        return np.count_nonzero(mask)

    def select_neighbors_to_use_for_refit(self, dct):
        """Select only neighboring fit solutions for the refit that show the right number of centroid positions within the determined interval."""
        mask = dct['weights'] >= self.w_min
        indices = dct['indices_neighbors'][mask]
        dct['indices_refit'] = {}

        for key in dct['means_interval']:
            interval = dct['means_interval'][key]
            ncomps_expected = dct['n_centroids'][key]

            indices_refit = []
            for idx in indices:
                means = self.decomposition['means_fit'][idx]
                ncomps = self.number_of_values_in_interval(means, interval)
                if ncomps == ncomps_expected:
                    indices_refit.append(idx)

            dct['indices_refit'][key] = indices_refit

        indices_refit_all_individual = list(dct['indices_refit'].values())
        if len(indices_refit_all_individual) > 1:
            indices_refit_all = reduce(
                np.intersect1d, indices_refit_all_individual)
            dct['indices_refit_all'] = indices_refit_all
        else:
            dct['indices_refit_all'] = indices_refit_all_individual[0]

        return dct

    def determine_all_neighbors(self):
        """Determine the indices of all valid neighbors."""
        say("\ndetermine neighbors for all spectra...", logger=self.logger)

        mask_all = np.array(
            [0 if x is None else 1 for x in self.decomposition['N_components']]).astype('bool')
        self.indices_all = np.array(
            self.decomposition['index_fit'])[mask_all]
        self.locations_all = np.take(
            np.array(self.location), self.indices_all, axis=0)

        for i, loc in tqdm(zip(self.indices_all, self.locations_all)):
            indices_neighbors_total = np.array([])
            for direction in [
                    'horizontal', 'vertical', 'diagonal_ul', 'diagonal_ur']:
                indices_neighbors = get_neighbors(
                    loc, exclude_p=True, shape=self.shape, nNeighbors=2,
                    direction=direction, get_indices=True)
                indices_neighbors_total = np.append(
                    indices_neighbors_total, indices_neighbors)
            indices_neighbors_total = indices_neighbors_total.astype('int')
            self.neighbor_indices_all[i] = indices_neighbors_total

    def check_indices_refit(self):
        """Check which spectra show incoherence in their fitted centroid positions and require refitting."""
        say('\ncheck which spectra require refitting...', logger=self.logger)
        if self.refitting_iteration == 1:
            self.determine_all_neighbors()

        if np.count_nonzero(self.mask_refitted) == len(self.mask_refitted):
            self.indices_refit = self.indices_all.copy()
            self.locations_refit = self.locations_all.copy()
            return

        indices_remove = np.array([])

        for i in self.indices_all:
            if np.count_nonzero(
                    self.mask_refitted[self.neighbor_indices_all[i]]) == 0:
                indices_remove = np.append(indices_remove, i).astype('int')

        self.indices_refit = np.delete(self.indices_all.copy(), indices_remove)

        self.locations_refit = np.take(
            np.array(self.location), self.indices_refit, axis=0)

    def check_continuity(self):
        """Check continuity of centroid positions.

        See Fig. 9 and Sect. 3.3.2. in Riener+ 2019 for more details.

        """
        self.refitting_iteration += 1
        say('\nthreshold for required components: {:.3f}'.format(self.min_p),
            logger=self.logger)

        self.determine_spectra_for_flagging()

        self.check_indices_refit()
        self.refitting()

    def refit_spectrum_phase_2(self, index, i):
        """Parallelized function for refitting spectra whose fitted centroid position values show spatial incoherence.

        Parameters
        ----------
        index : int
            Index ('index_fit' keyword) of the spectrum that will be refit.
        i : int
            List index of the entry in the list that is handed over to the
            multiprocessing routine

        Returns
        -------
        list
            A list in the form of [index, dictResults_best, indices_neighbors, refit] in case of a successful refit; otherwise [index, 'None', indices_neighbors, refit] is returned.

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

        dictComps = self.check_continuity_centroids(index, loc)
        dct_refit = self.check_for_required_components(index, dictComps)

        if len(dct_refit['means_interval'].keys()) == 0:
            return [index, None, indices_neighbors, refit]

        dct_refit = self.select_neighbors_to_use_for_refit(dct_refit)

        #  TODO: first try to fit with indices_refit_all if present

        for key in dct_refit['indices_refit']:
            indices_neighbors = np.array(dct_refit['indices_refit'][key]).astype('int')
            interval = dct_refit['means_interval'][key]
            n_centroids = dct_refit['n_centroids'][key]

            dictResults, refit = self.try_refit_with_individual_neighbors(
                index, spectrum, rms, indices_neighbors, signal_ranges,
                noise_spike_ranges, signal_mask, interval=interval,
                n_centroids=n_centroids, dct_new_fit=dictResults)

            if dictResults is not None:
                dictResults_best = dictResults

        return [index, dictResults_best, indices_neighbors, refit]
