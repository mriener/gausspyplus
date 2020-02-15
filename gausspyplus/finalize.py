import os
import pickle

import numpy as np

from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS

from .config_file import get_values_from_config_file
from .gausspy_py3.gp_plus import get_fully_blended_gaussians
from .utils.fit_quality_checks import negative_residuals
from .utils.gaussian_functions import combined_gaussian, area_of_gaussian
from .utils.output import say
from .utils.spectral_cube_functions import correct_header, change_header, save_fits, return_hdu_options
from .spatial_fitting import SpatialFitting


class Finalize(object):
    def __init__(self, path_to_pickle_file=None,
                 path_to_decomp_file=None, fin_filename=None,
                 config_file=''):
        """Class containing methods to finalize the GaussPy+ results.

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
        self.dirpath_table = None
        self.fin_filename = fin_filename

        self.dct_params = None
        self.config_file = config_file
        self.ncomps_max = None

        self.subcube_nr = None
        self.xpos_offset = 0
        self.ypos_offset = 0

        self.main_beam_efficiency = None

        self.initialized_state = False

        if config_file:
            get_values_from_config_file(
                self, config_file, config_key='DEFAULT')

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
            suffix = '_finalized'
            self.fin_filename = self.filename + suffix

        if self.dirpath_gpy is None:
            self.dirpath_gpy = os.path.dirname(self.decomp_dirname)

        if self.dirpath_table is None:
            self.dirpath_table = self.decomp_dirname

    def initialize(self):
        """Read in data files and initialize parameters."""
        with open(self.path_to_pickle_file, "rb") as pickle_file:
            self.pickled_data = pickle.load(pickle_file, encoding='latin1')

        with open(self.path_to_decomp_file, "rb") as pickle_file:
            self.decomposition = pickle.load(pickle_file, encoding='latin1')

        self.length = len(self.pickled_data['index'])

        if 'header' in self.pickled_data.keys():
            self.header = correct_header(self.pickled_data['header'])
            self.wcs = WCS(self.header)
            self.velocity_increment = (
                self.wcs.wcs.cdelt[2] * self.wcs.wcs.cunit[2]).to(
                    self.vel_unit).value
            self.to_unit = (self.wcs.wcs.cunit[2]).to(self.vel_unit)

        self.initialized_state = True

    def finalize_dct(self):
        self.check_settings()
        self.initialize()
        sp = SpatialFitting(config_file=self.config_file)
        if self.dct_params is not None:
            for key, value in self.dct_params.items():
                try:
                    setattr(sp, key, value)
                except ValueError:
                    raise Exception('Could not parse parameter {} from dct_params'.format(key))

        sp.log_output = False
        sp.path_to_pickle_file = self.path_to_pickle_file
        sp.path_to_decomp_file = self.path_to_decomp_file

        results_list = sp.finalize()

        list_means_interval, list_n_centroids = (
            [{} for _ in range(self.length)] for _ in range(2))

        for i, item in enumerate(results_list):
            if not isinstance(item, list):
                say("Error for index {}: {}".format(i, item))
                continue

            index, means_interval, n_centroids = item
            list_means_interval[index] = means_interval
            list_n_centroids[index] = n_centroids

        self.decomposition['broad'] = sp.mask_broad_flagged
        self.decomposition['ncomps_wmedian'] = sp.ncomps_wmedian.astype('int')
        self.decomposition['ncomps_jumps'] = sp.ncomps_jumps.astype('int')
        self.decomposition['means_interval'] = list_means_interval
        self.decomposition['n_centroids'] = list_n_centroids

    def get_flag_blended(self, amps, fwhms, means):
        params_fit = amps + fwhms + means
        indices = get_fully_blended_gaussians(params_fit)
        flags = np.zeros(len(amps))
        flags[indices] = 1
        return flags.astype('int')

    def get_flag_broad(self, fwhms, broad):
        flags = np.zeros(len(fwhms))
        if broad:
            flags[np.argmax(fwhms)] = 1
        return flags.astype('int')

    def get_flag_centroid(self, means, means_interval, n_centroids):
        flag = 0
        for key in means_interval.keys():
            n_wanted = n_centroids[key]
            lower = means_interval[key][0]
            upper = means_interval[key][1]
            n_real = np.count_nonzero(
                np.logical_and(lower <= means, means <= upper))

            flag += abs(n_wanted - n_real)

        return flag

    def get_table_rows(self, idx, j):
        rows = []
        ncomps = self.decomposition['N_components'][idx]

        #  do not continue if spectrum was masked out, was not fitted,
        #  or was fitted by too many components
        if ncomps is None:
            return rows
        elif ncomps == 0:
            return rows
        elif self.ncomps_max is not None:
            if ncomps > self.ncomps_max:
                return rows

        yi, xi = self.pickled_data['location'][idx]
        spectrum = self.pickled_data['data_list'][idx]
        fit_amps = self.decomposition['amplitudes_fit'][idx]
        fit_fwhms = self.decomposition['fwhms_fit'][idx]
        fit_means = self.decomposition['means_fit'][idx]
        fit_e_amps = self.decomposition['amplitudes_fit_err'][idx]
        fit_e_fwhms = self.decomposition['fwhms_fit_err'][idx]
        fit_e_means = self.decomposition['means_fit_err'][idx]
        error = self.pickled_data['error'][idx][0]

        residual = spectrum - combined_gaussian(
            fit_amps, fit_fwhms, fit_means, self.pickled_data['x_values'])

        aicc = self.decomposition['best_fit_aicc'][idx]
        rchi2 = self.decomposition['best_fit_rchi2'][idx]
        pvalue = self.decomposition['pvalue'][idx]

        broad = self.decomposition['broad'][idx]

        ncomp_wmedian = self.decomposition['ncomps_wmedian'][idx]
        ncomp_jumps = self.decomposition['ncomps_jumps'][idx]

        means_interval = self.decomposition['means_interval'][idx]
        n_centroids = self.decomposition['n_centroids'][idx]

        flags_blended = self.get_flag_blended(fit_amps, fit_fwhms, fit_means)
        flags_neg_res_peak = negative_residuals(
            spectrum, residual, error, get_flags=True,
            fwhms=fit_fwhms, means=fit_means)
        flags_broad = self.get_flag_broad(fit_fwhms, broad)
        flag_centroid = self.get_flag_centroid(
            np.array(fit_means), means_interval, n_centroids)

        x_wcs, y_wcs, z_wcs = self.wcs.wcs_pix2world(
            xi, yi, np.array(fit_means), 0)

        velocities = z_wcs * self.to_unit
        e_velocities = np.array(fit_e_means) * self.velocity_increment
        vel_disps = (
            np.array(fit_fwhms) / 2.354820045) * self.velocity_increment
        e_vel_disps = (
            np.array(fit_e_fwhms) / 2.354820045) * self.velocity_increment

        amplitudes = np.array(fit_amps)
        e_amplitudes = np.array(fit_e_amps)

        if self.main_beam_efficiency is not None:
            amplitudes /= self.main_beam_efficiency
            e_amplitudes /= self.main_beam_efficiency
            error /= self.main_beam_efficiency

        integrated_intensity = area_of_gaussian(
            amplitudes, np.array(fit_fwhms) * self.velocity_increment)
        fit_fwhms_plus_error = np.array(fit_fwhms) + np.array(fit_e_fwhms)
        e_integrated_intensity = area_of_gaussian(
            amplitudes + e_amplitudes,
            fit_fwhms_plus_error * self.velocity_increment) -\
            integrated_intensity

        for i in range(ncomps):
            row = [
                xi + self.xpos_offset, yi + self.ypos_offset,
                x_wcs[0], y_wcs[0],
                amplitudes[i], e_amplitudes[i],
                velocities[i], e_velocities[i],
                vel_disps[i], e_vel_disps[i],
                integrated_intensity[i], e_integrated_intensity[i],
                error, pvalue, aicc, rchi2,
                ncomps, ncomp_wmedian, ncomp_jumps,
                flags_blended[i], flags_neg_res_peak[i],
                flags_broad[i], flag_centroid]

            if self.subcube_nr is not None:
                row.append(self.subcube_nr)

            rows.append(row)

        return rows

    def make_table(self):
        import gausspyplus.parallel_processing
        gausspyplus.parallel_processing.init([self.decomposition['index_fit'], [self]])

        results_list = gausspyplus.parallel_processing.func(
            use_ncpus=self.use_ncpus, function='make_table')

        for i, item in enumerate(results_list):
            if not isinstance(item, list):
                say("Error for spectrum with index {}: {}".format(i, item))
                continue

        # results_list = [item for item in results_list if len(item) > 0]
        results_list = np.array([item for sublist in results_list
                                 for item in sublist])

        names = [
            'x_pos', 'y_pos', self.wcs.wcs.lngtyp, self.wcs.wcs.lattyp,
            'amp', 'e_amp', 'VLSR', 'e_VLSR', 'vel_disp', 'e_vel_disp',
            'int_tot', 'e_int_tot', 'rms', 'pvalue', 'aicc', 'rchi2',
            'ncomps', 'ncomp_wmedian', 'ncomp_jumps',
            'flag_blended', 'flag_neg_res_peak', 'flag_broad', 'flag_centroid']

        dtype = ['i4']*2 + ['f4']*14 + ['i4']*7

        if self.subcube_nr is not None:
            names.append('subcube_nr')
            dtype.append('i4')

        table_results = Table(data=results_list, names=names, dtype=dtype)

        for key in names[2:16]:
            table_results[key].format = "{0:.4f}"

        filename = self.fin_filename + '.dat'
        path_to_table = os.path.join(self.dirpath_table, filename)
        table_results.write(path_to_table, format='ascii', overwrite=True)
        say("\033[92mSAVED FILE:\033[0m '{}' in '{}'".format(
            filename, self.decomp_dirname))

    def save_final_results(self):
        """Save the results of the spatially coherent refitting iterations."""
        filename = self.fin_filename + '.pickle'
        pathToFile = os.path.join(self.decomp_dirname, filename)
        pickle.dump(self.decomposition, open(pathToFile, 'wb'), protocol=2)
        say("\033[92mSAVED FILE:\033[0m '{}' in '{}'".format(
            filename, self.decomp_dirname))

    def get_array(self, keyword='error', comments=[], suffix='', save=True,
                  dtype='float32'):
        #  TODO: routine in case pickled_data is missing the header key
        shape = (self.header['NAXIS2'], self.header['NAXIS1'])
        array = np.ones((shape[0], shape[1])) * np.nan

        if keyword in self.pickled_data.keys():
            data = self.pickled_data[keyword]
        elif keyword in self.decomposition.keys():
            data = self.decomposition[keyword]

        for (y, x), value in zip(self.pickled_data['location'], data):
            if value is None:
                continue

            try:
                array[y, x] = value
            except ValueError:
                array[y, x] = value[0]

        header = change_header(self.header.copy(), format='pp',
                               comments=comments)

        array = array.astype(dtype)

        if save:
            if keyword == 'error':
                filename, _ = os.path.splitext(
                    os.path.basename(self.path_to_pickle_file))
            else:
                filename = self.filename

            filename = "{}{}.fits".format(filename, suffix)
            path_to_file = os.path.join(
                self.dirpath_gpy, 'gpy_maps', filename)

            save_fits(array, header, path_to_file,
                      verbose=False)
            say("\n\033[92mSAVED FILE:\033[0m '{}' in '{}'".format(
                filename, os.path.dirname(path_to_file)), logger=self.logger)

        return fits.PrimaryHDU(array, header)

    def produce_noise_map(self, comments=['Noise map.'],
                          suffix='_noise_map', save=True, dtype='float32',
                          get_hdu=False, get_data=False, get_header=False):
        if not self.initialized_state:
            self.check_settings()
            self.initialize()
        hdu = self.get_array(keyword='error', comments=comments,
                             suffix=suffix, save=save, dtype=dtype)

        return return_hdu_options(
            hdu, get_hdu=get_hdu, get_data=get_data, get_header=get_header)

    def produce_rchi2_map(self, suffix='_rchi2_map', save=True, dtype='float32',
                          get_hdu=False, get_data=False, get_header=False,
                          comments=['Reduced chi2 values of GaussPy fits']):
        if not self.initialized_state:
            self.check_settings()
            self.initialize()
        hdu = self.get_array(keyword='best_fit_rchi2', comments=comments,
                             suffix=suffix, save=save, dtype=dtype)

        return return_hdu_options(
            hdu, get_hdu=get_hdu, get_data=get_data, get_header=get_header)

    def produce_component_map(self, suffix='_component_map', save=True,
                              dtype='float32', get_hdu=False, get_data=False, get_header=False, comments=['Number of fitted GaussPy components']):
        if not self.initialized_state:
            self.check_settings()
            self.initialize()
        hdu = self.get_array(keyword='N_components', comments=comments,
                             suffix=suffix, save=save, dtype=dtype)

        return return_hdu_options(
            hdu, get_hdu=get_hdu, get_data=get_data, get_header=get_header)
