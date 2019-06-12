# @Author: riener
# @Date:   2019-02-08T15:40:10+01:00
# @Filename: decompose.py
# @Last modified by:   riener
# @Last modified time: 10-04-2019

import os
import pickle
import warnings

import numpy as np

from astropy import units as u
from astropy.table import Table
from astropy.wcs import WCS
from tqdm import tqdm

from .config_file import get_values_from_config_file
from .utils.gaussian_functions import gaussian, area_of_gaussian, combined_gaussian
from .utils.spectral_cube_functions import change_header, save_fits, correct_header, update_header
from .utils.output import set_up_logger, say


class GaussPyDecompose(object):
    """Decompose spectra with GaussPy+.

    Attributes
    ----------
    path_to_pickle_file : str
        Filepath to the pickled dictionary produced by GaussPyPrepare.
    dirpath_gpy : str
        Directory in which all files produced by GaussPy+ are saved

    two_phase_decomposition : bool
        'True' (default) uses two smoothing parameters (alpha1, alpha2) for the decomposition. 'False' uses only the alpha1 smoothing parameter.
    save_initial_guesses : bool
        Default is 'False'. Set to 'True' if initial GaussPy fitting guesses should be saved.
    alpha1 : float
        First smoothing parameter.
    alpha2 : float
        Second smoothing parameter. Only used if two_phase_decomposition is set to 'True'
    snr_thresh : float
        S/N threshold used for the original spectrum.
    snr2_thresh : float
        S/N threshold used for the second derivate of the smoothed spectrum.

    use_ncpus : int
        Number of CPUs used in the decomposition. By default 75% of all CPUs on the machine are used.
    fitting : dct
        Description of attribute `fitting`.
    separation_factor : float
        The required minimum separation between two Gaussian components (mean1, fwhm1) and (mean2, fwhm2) is determined as separation_factor * min(fwhm1, fwhm2).
    main_beam_efficiency : float
        Default is 'None'. Specify if intensity values should be corrected by the main beam efficiency.
    vel_unit : astropy.units
        Default is 'u.km/u.s'. Unit to which velocity values will be converted.
    testing : bool
        Default is 'False'. Set to 'True' if in testing mode.
    verbose : bool
        Default is 'True'. Set to 'False' if descriptive statements should not be printed in the terminal.
    suffix : str
        Suffix for filename of the decomposition results.
    log_output : bool
        Default is 'True'. Set to 'False' if terminal output should not be logged.

    """

    def __init__(self, path_to_pickle_file=None, config_file=''):
        self.path_to_pickle_file = path_to_pickle_file
        self.dirpath_gpy = None

        # self.gausspy_decomposition = True
        self.two_phase_decomposition = True
        self.save_initial_guesses = False
        self.alpha1 = None
        self.alpha2 = None
        self.snr_thresh = None
        self.snr2_thresh = None

        self.improve_fitting = True
        self.exclude_means_outside_channel_range = True
        self.min_fwhm = 1.
        self.max_fwhm = None
        self.snr = 3.
        self.snr_fit = None
        self.significance = 5.
        self.snr_negative = None
        self.rchi2_limit = None
        self.max_amp_factor = 1.1
        self.refit_neg_res_peak = True
        self.refit_broad = True
        self.refit_blended = True
        self.separation_factor = 0.8493218
        self.fwhm_factor = 2.
        self.min_pvalue = 0.01

        self.main_beam_efficiency = None
        self.vel_unit = u.km / u.s
        self.testing = False
        self.verbose = True
        self.suffix = ''
        self.log_output = True
        self.use_ncpus = None

        self.single_prepared_spectrum = None

        if config_file:
            get_values_from_config_file(
                self, config_file, config_key='decomposition')

    def getting_ready(self):
        string = 'GaussPy decomposition'
        banner = len(string) * '='
        heading = '\n' + banner + '\n' + string + '\n' + banner
        say(heading, logger=self.logger)

    def initialize_data(self):
        self.logger = False
        if self.log_output:
            self.logger = set_up_logger(
                self.dirpath_gpy, self.filename, method='g+_decomposition')

        say("\npickle load '{}'...".format(self.file), logger=self.logger)

        with open(self.path_to_pickle_file, "rb") as pickle_file:
            self.pickled_data = pickle.load(pickle_file, encoding='latin1')

        if 'header' in self.pickled_data.keys():
            self.header = correct_header(self.pickled_data['header'])
            self.wcs = WCS(self.header)
            self.velocity_increment = (
                self.wcs.wcs.cdelt[2] * self.wcs.wcs.cunit[2]).to(
                    self.vel_unit).value
        if 'location' in self.pickled_data.keys():
            self.location = self.pickled_data['location']
        if 'nan_mask' in self.pickled_data.keys():
            self.nan_mask = self.pickled_data['nan_mask']
        if 'testing' in self.pickled_data.keys():
            self.testing = self.pickled_data['testing']
            self.use_ncpus = 1

        self.data = self.pickled_data['data_list']
        self.channels = self.pickled_data['x_values']
        self.errors = self.pickled_data['error']

    def check_settings(self):
        if self.path_to_pickle_file is None:
            raise Exception("Need to specify 'path_to_pickle_file'")

        self.dirname = os.path.dirname(self.path_to_pickle_file)
        self.file = os.path.basename(self.path_to_pickle_file)
        self.filename, self.file_extension = os.path.splitext(self.file)

        if self.dirpath_gpy is None:
            self.dirpath_gpy = os.path.normpath(
                self.dirname + os.sep + os.pardir)

        self.decomp_dirname = os.path.join(self.dirpath_gpy, 'gpy_decomposed')
        if not os.path.exists(self.decomp_dirname):
            os.makedirs(self.decomp_dirname)

        if self.main_beam_efficiency is None:
            warnings.warn('assuming intensities are already corrected for main beam efficiency')

        warnings.warn("converting velocity values to {}".format(self.vel_unit))

    def decompose(self):
        if self.single_prepared_spectrum:
            self.logger = False
            self.testing = True
            self.use_ncpus = 1
            self.log_output = False
            self.getting_ready()
            return self.start_decomposition()
        else:
            self.check_settings()
            self.initialize_data()
            self.getting_ready()
            self.start_decomposition()
            if 'batchdecomp_temp.pickle' in os.listdir(os.getcwd()):
                os.remove('batchdecomp_temp.pickle')

    def decomposition_settings(self):
        if self.snr_negative is None:
            self.snr_negative = self.snr
        if self.snr_fit is None:
            self.snr_fit = self.snr / 2.
        if self.snr_thresh is None:
            self.snr_thresh = self.snr
        if self.snr2_thresh is None:
            self.snr2_thresh = self.snr

        self.fitting = {
            'improve_fitting': self.improve_fitting,
            'min_fwhm': self.min_fwhm,
            'max_fwhm': self.max_fwhm,
            'snr': self.snr,
            'snr_fit': self.snr_fit,
            'significance': self.significance,
            'snr_negative': self.snr_negative,
            'rchi2_limit': self.rchi2_limit,
            'max_amp_factor': self.max_amp_factor,
            'neg_res_peak': self.refit_neg_res_peak,
            'broad': self.refit_broad,
            'blended': self.refit_blended,
            'fwhm_factor': self.fwhm_factor,
            'separation_factor': self.separation_factor,
            'exclude_means_outside_channel_range': self.exclude_means_outside_channel_range,
            'min_pvalue': self.min_pvalue}

        string_gausspy = str(
            '\ndecomposition settings:'
            '\nGaussPy:'
            '\nTwo phase decomposition: {a}'
            '\nalpha1: {b}'
            '\nalpha2: {c}'
            '\nSNR1: {d}'
            '\nSNR2: {e}').format(
                a=self.two_phase_decomposition,
                b=self.alpha1,
                c=self.alpha2,
                d=self.snr_thresh,
                e=self.snr2_thresh)
        say(string_gausspy, logger=self.logger)

        string_gausspy_plus = ''
        if self.fitting['improve_fitting']:
            for key, value in self.fitting.items():
                string_gausspy_plus += str('\n{}: {}').format(key, value)
        else:
            string_gausspy_plus += str(
                '\nimprove_fitting: {}').format(
                    self.fitting['improve_fitting'])
        say(string_gausspy_plus, logger=self.logger)

    def start_decomposition(self):
        if self.alpha1 is None:
            raise Exception("Need to specify 'alpha1' for decomposition.")

        if self.two_phase_decomposition and (self.alpha2 is None):
            raise Exception(
                "Need to specify 'alpha2' for 'two_phase_decomposition'.")

        self.decomposition_settings()
        say('\ndecomposing data...', logger=self.logger)

        from .gausspy_py3 import gp as gp
        g = gp.GaussianDecomposer()  # Load GaussPy
        g.set('use_ncpus', self.use_ncpus)
        g.set('SNR_thresh', self.snr_thresh)
        g.set('SNR2_thresh', self.snr2_thresh)
        g.set('improve_fitting_dict', self.fitting)
        g.set('alpha1', self.alpha1)

        if self.testing:
            g.set('verbose', True)
            g.set('plot', True)

        if self.two_phase_decomposition:
            g.set('phase', 'two')
            g.set('alpha2', self.alpha2)
        else:
            g.set('phase', 'one')

        if self.single_prepared_spectrum:
            return g.batch_decomposition(dct=self.single_prepared_spectrum)

        self.decomposition = g.batch_decomposition(self.path_to_pickle_file)

        self.save_final_results()

        if self.save_initial_guesses:
            self.save_initial_guesses()

    def save_initial_guesses(self):
        say('\npickle dump GaussPy initial guesses...', logger=self.logger)

        filename = '{}{}_fit_ini.pickle'.format(self.filename, self.suffix)
        pathname = os.path.join(self.decomp_dirname, filename)

        dct_initial_guesses = {}

        for key in ["index_initial", "amplitudes_initial",
                    "fwhms_initial", "means_initial"]:
            dct_initial_guesses[key] = self.decomposition[key]

        pickle.dump(dct_initial_guesses, open(pathname, 'wb'), protocol=2)
        say("\033[92mSAVED FILE:\033[0m '{}' in '{}'".format(
            filename, self.decomp_dirname), logger=self.logger)

    def save_final_results(self):
        say('\npickle dump GaussPy final results...', logger=self.logger)

        dct_gausspy_settings = {"two_phase": self.two_phase_decomposition,
                                "alpha1": self.alpha1,
                                "snr1_thresh": self.snr_thresh,
                                "snr2_thresh": self.snr2_thresh}

        if self.two_phase_decomposition:
            dct_gausspy_settings["alpha2"] = self.alpha2

        dct_final_guesses = {}

        for key in ["index_fit", "best_fit_rchi2", "best_fit_aicc", "pvalue",
                    "amplitudes_fit", "amplitudes_fit_err", "fwhms_fit",
                    "fwhms_fit_err", "means_fit", "means_fit_err", "log_gplus",
                    "N_neg_res_peak", "N_blended", "N_components",
                    "quality_control"]:
            dct_final_guesses[key] = self.decomposition[key]

        dct_final_guesses["gausspy_settings"] = dct_gausspy_settings

        dct_final_guesses["improve_fit_settings"] = self.fitting

        filename = '{}{}_fit_fin.pickle'.format(self.filename, self.suffix)
        pathname = os.path.join(self.decomp_dirname, filename)
        pickle.dump(dct_final_guesses, open(pathname, 'wb'), protocol=2)
        say("\033[92mSAVED FILE:\033[0m '{}' in '{}'".format(
            filename, self.decomp_dirname), logger=self.logger)

    def load_final_results(self, pathToDecomp):
        self.check_settings()
        self.initialize_data()
        self.getting_ready()

        say('\npickle load final GaussPy results...', logger=self.logger)

        self.decomp_dirname = os.path.dirname(pathToDecomp)
        with open(pathToDecomp, "rb") as pickle_file:
            self.decomposition = pickle.load(pickle_file, encoding='latin1')

        self.file = os.path.basename(pathToDecomp)
        self.filename, self.file_extension = os.path.splitext(self.file)

        if 'header' in self.decomposition.keys():
            self.header = self.decomposition['header']
        if 'channels' in self.decomposition.keys():
            self.channels = self.decomposition['channels']
        if 'nan_mask' in self.pickled_data.keys():
            self.nan_mask = self.pickled_data['nan_mask']
        if 'location' in self.pickled_data.keys():
            self.location = self.pickled_data['location']

    def make_cube(self, mode='full_decomposition'):
        """Create FITS cube of the decomposition results.

        Parameters
        ----------
        mode : str
            'full_decomposition' recreates the whole FITS cube, 'integrated_intensity' creates a cube with the integrated intensity values of the Gaussian components placed at their mean positions, 'main_component' only retains the fitted component with the largest amplitude value
        """
        say('\ncreate {} cube...'.format(mode), logger=self.logger)

        x = self.header['NAXIS1']
        y = self.header['NAXIS2']
        z = self.header['NAXIS3']

        array = np.zeros([z, y, x], dtype=np.float32)
        nSpectra = len(self.decomposition['N_components'])

        for idx in range(nSpectra):
            ncomps = self.decomposition['N_components'][idx]
            if ncomps is None:
                continue

            yi = self.location[idx][0]
            xi = self.location[idx][1]

            amps = self.decomposition['amplitudes_fit'][idx]
            fwhms = self.decomposition['fwhms_fit'][idx]
            means = self.decomposition['means_fit'][idx]

            if self.main_beam_efficiency is not None:
                amps = [amp / self.main_beam_efficiency for amp in amps]

            if mode == 'main_component' and ncomps > 0:
                j = amps.index(max(amps))
                array[:, yi, xi] = gaussian(
                    amps[j], fwhms[j], means[j], self.channels)
            elif mode == 'integrated_intensity' and ncomps > 0:
                for j in range(ncomps):
                    integrated_intensity = area_of_gaussian(
                        amps[j], fwhms[j] * self.velocity_increment)
                    channel = int(round(means[j]))
                    if self.channels[0] <= channel <= self.channels[-1]:
                        array[channel, yi, xi] += integrated_intensity
            elif mode == 'full_decomposition':
                array[:, yi, xi] = combined_gaussian(
                    amps, fwhms, means, self.channels)

            nans = self.nan_mask[:, yi, xi]
            array[:, yi, xi][nans] = np.NAN

        if mode == 'main_component':
            comment = str('Fitted Gaussians from GaussPy decomposition, '
                          'per spectrum only Gaussian with highest '
                          'amplitude is included')
            filename = "{}{}_main.fits".format(self.filename, self.suffix)
        elif mode == 'integrated_intensity':
            comment = str('integrated intensity of Gaussian components '
                          'from GaussPy decomposition at VLSR positions')
            filename = "{}{}_wco.fits".format(self.filename, self.suffix)
        elif mode == 'full_decomposition':
            comment = 'Fitted Gaussians of GaussPy decomposition'
            filename = "{}{}_decomp.fits".format(self.filename, self.suffix)

        array[self.nan_mask] = np.nan

        comments = [comment]

        self.header = update_header(
            self.header, comments=comments, write_meta=True)

        pathToFile = os.path.join(self.decomp_dirname, 'FITS', filename)
        save_fits(array, self.header, pathToFile, verbose=False)
        say("\033[92mSAVED FILE:\033[0m '{}' in '{}'".format(
            filename, os.path.dirname(pathToFile)), logger=self.logger)

    def create_input_table(self, ncomps_max=None):
        """Create a table of the decomposition results.

        The table contains the following columns:
        {0}: Pixel position in X direction
        {1}: Pixel position in Y direction
        {2}: Pixel position in Z direction
        {3}: Amplitude value of fitted Gaussian component
        {4}: Root-mean-square noise of the spectrum
        {5}: Velocity dispersion value of fitted Gaussian component
        {6}: Integrated intensity value of fitted Gaussian component
        {7}: Coordinate position in X direction
        {8}: Coordinate position in Y direction
        {9}: Mean position (VLSR) of fitted Gaussian component
        {10}: Error of amplitude value
        {11}: Error of velocity dispersion value
        {12}: Error of velocity value
        {13}: Error of integrated intensity value

        Amplitude and RMS values get corrected by the main_beam_efficiency parameter in case it was supplied.

        The table is saved in the 'gpy_tables' directory.

        Parameters
        ----------
        ncomps_max : int
            All spectra whose number of fitted components exceeds this value will be neglected.
        """
        say('\ncreate input table...', logger=self.logger)

        length = len(self.decomposition['amplitudes_fit'])

        x_pos, y_pos, z_pos, amp, rms, vel_disp, int_tot, x_coord, y_coord,\
            velocity, e_amp, e_vel_disp, e_velocity, e_int_tot = (
                [] for i in range(14))

        for idx in tqdm(range(length)):
            ncomps = self.decomposition['N_components'][idx]

            #  do not continue if spectrum was masked out, was not fitted,
            #  or was fitted by too many components
            if ncomps is None:
                continue
            elif ncomps == 0:
                continue
            elif ncomps_max is not None:
                if ncomps > ncomps_max:
                    continue

            yi, xi = self.location[idx]
            fit_amps = self.decomposition['amplitudes_fit'][idx]
            fit_fwhms = self.decomposition['fwhms_fit'][idx]
            fit_means = self.decomposition['means_fit'][idx]
            fit_e_amps = self.decomposition['amplitudes_fit_err'][idx]
            fit_e_fwhms = self.decomposition['fwhms_fit_err'][idx]
            fit_e_means = self.decomposition['means_fit_err'][idx]
            error = self.errors[idx][0]

            if self.main_beam_efficiency is not None:
                fit_amps = [
                    amp / self.main_beam_efficiency for amp in fit_amps]
                fit_e_amps = [
                    e_amp / self.main_beam_efficiency for e_amp in fit_e_amps]
                error /= self.main_beam_efficiency

            for j in range(ncomps):
                amp_value = fit_amps[j]
                e_amp_value = fit_e_amps[j]
                fwhm_value = fit_fwhms[j] * self.velocity_increment
                e_fwhm_value = fit_e_fwhms[j] * self.velocity_increment
                mean_value = fit_means[j]
                e_mean_value = fit_e_means[j]

                channel = int(round(mean_value))
                if channel < self.channels[0] or channel > self.channels[-1]:
                    continue

                x_wcs, y_wcs, z_wcs = self.wcs.wcs_pix2world(
                    xi, yi, mean_value, 0)

                x_pos.append(xi)
                y_pos.append(yi)
                z_pos.append(channel)
                rms.append(error)

                amp.append(amp_value)
                e_amp.append(e_amp_value)

                velocity.append(
                    (z_wcs * self.wcs.wcs.cunit[2]).to(self.vel_unit).value)
                e_velocity.append(
                    e_mean_value * self.velocity_increment)

                vel_disp.append(fwhm_value / 2.354820045)
                e_vel_disp.append(e_fwhm_value / 2.354820045)

                integrated_intensity = area_of_gaussian(amp_value, fwhm_value)
                e_integrated_intensity = area_of_gaussian(
                    amp_value + e_amp_value, fwhm_value + e_fwhm_value) -\
                    integrated_intensity
                int_tot.append(integrated_intensity)
                e_int_tot.append(e_integrated_intensity)
                x_coord.append(x_wcs)
                y_coord.append(y_wcs)

        names = ['x_pos', 'y_pos', 'z_pos', 'amp', 'rms', 'vel_disp',
                 'int_tot', self.wcs.wcs.lngtyp, self.wcs.wcs.lattyp, 'VLSR',
                 'e_amp', 'e_vel_disp', 'e_VLSR', 'e_int_tot']

        dtype = tuple(3*['i4'] + (len(names) - 3)*['f4'])

        table = Table([
            x_pos, y_pos, z_pos, amp, rms, vel_disp, int_tot, x_coord,
            y_coord, velocity, e_amp, e_vel_disp, e_velocity, e_int_tot],
            names=names, dtype=dtype)

        for key in names[3:]:
            table[key].format = "{0:.4f}"

        tableDirname = os.path.join(os.path.dirname(self.dirname), 'gpy_tables')
        if not os.path.exists(tableDirname):
            os.makedirs(tableDirname)

        filename = '{}{}_wco.dat'.format(self.filename, self.suffix)
        pathToTable = os.path.join(tableDirname, filename)
        table.write(pathToTable, format='ascii', overwrite=True)
        say("\n\033[92mSAVED FILE:\033[0m '{}' in '{}'".format(
            filename, tableDirname), logger=self.logger)

    def produce_component_map(self, dtype='float32'):
        """Create FITS map showing the number of fitted components.

        The FITS file in saved in the gpy_maps directory.
        """
        say("\nmaking component map...", logger=self.logger)
        data = np.empty((self.header['NAXIS2'],
                         self.header['NAXIS1']))
        data.fill(np.nan)

        for idx, ((y, x), components) in enumerate(zip(
                self.location, self.decomposition['N_components'])):
            if components is not None:
                data[y, x] = components

        comments = ['Number of fitted GaussPy components']
        header = change_header(self.header.copy(), format='pp',
                               comments=comments)

        filename = "{}{}_component_map.fits".format(
            self.filename, self.suffix)
        pathToFile = os.path.join(
            os.path.dirname(self.dirname), 'gpy_maps', filename)

        save_fits(data.astype(dtype), header, pathToFile, verbose=True)

    def produce_rchi2_map(self, dtype='float32'):
        """Create FITS map showing the reduced chi-square values of the decomposition.

        The FITS file in saved in the gpy_maps directory.
        """
        say("\nmaking reduced chi2 map...", logger=self.logger)

        data = np.empty((self.header['NAXIS2'], self.header['NAXIS1']))
        data.fill(np.nan)

        for idx, ((y, x), components, rchi2) in enumerate(zip(
                self.location, self.decomposition['N_components'],
                self.decomposition['best_fit_rchi2'])):
            if components is not None:
                if rchi2 is None:
                    data[y, x] = 0.
                else:
                    data[y, x] = rchi2

        comments = ['Reduced chi2 values of GaussPy fits']
        header = change_header(self.header.copy(), format='pp',
                               comments=comments)

        filename = "{}{}_rchi2_map.fits".format(
            self.filename, self.suffix)
        pathToFile = os.path.join(
            os.path.dirname(self.dirname), 'gpy_maps', filename)

        save_fits(data.astype(dtype), header, pathToFile, verbose=True)

    def produce_velocity_dispersion_map(self, mode='average', dtype='float32'):
        """Produce map showing the maximum velocity dispersions."""
        say("\nmaking map of maximum velocity dispersions...",
            logger=self.logger)

        data = np.empty((self.header['NAXIS2'], self.header['NAXIS1']))
        data.fill(np.nan)

        # TODO: rewrite this in terms of wcs and CUNIT
        factor_kms = self.header['CDELT3'] / 1e3

        for idx, ((y, x), fwhms) in enumerate(zip(
                self.location, self.decomposition['fwhms_fit'])):
            if fwhms is not None:
                if len(fwhms) > 0:
                    if mode == 'average':
                        data[y, x] = np.mean(fwhms) * factor_kms / 2.354820045
                    elif mode == 'maximum':
                        data[y, x] = max(fwhms) * factor_kms / 2.354820045
                else:
                    data[y, x] = 0

        if mode == 'average':
            comments = ['Average velocity dispersion values of GaussPy fits']
        elif mode == 'maximum':
            comments = ['Maximum velocity dispersion values of GaussPy fits']
        header = change_header(self.header.copy(), format='pp',
                               comments=comments)

        filename = "{}{}_{}_veldisp_map.fits".format(
            self.filename, self.suffix, mode)
        pathToFile = os.path.join(
            os.path.dirname(self.dirname), 'gpy_maps', filename)

        save_fits(data.astype(dtype), header, pathToFile, verbose=False)
        say(">> saved {} velocity dispersion map '{}' in {}".format(
            mode, filename, os.path.dirname(pathToFile)), logger=self.logger)
