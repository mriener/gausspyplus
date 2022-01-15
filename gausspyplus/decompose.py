import functools
import os
import pickle
import warnings
from pathlib import Path

from astropy import units as u
from astropy.wcs import WCS

from gausspyplus.config_file import get_values_from_config_file
from gausspyplus.utils.spectral_cube_functions import correct_header
from gausspyplus.utils.output import set_up_logger, say


class GaussPyDecompose:
    """Decompose spectra with GaussPy+."""

    def __init__(self, path_to_pickle_file=None, config_file=''):
        self.path_to_pickle_file = path_to_pickle_file
        self.dirpath_gpy = None

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
        self.max_ncomps = None

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

    @functools.cached_property
    def dirpath(self):
        # TODO: homogenize attributes self.dirpath_gpy (used here) and self.gpy_dirpath (used in training_set)
        return self.dirpath_gpy if self.dirpath_gpy is not None else Path(self.path_to_pickle_file).parents[1]

    @functools.cached_property
    def decomp_dirname(self):
        (decomp_dirname := Path(self.dirpath, 'gpy_decomposed')).mkdir(parents=True, exist_ok=True)
        return decomp_dirname

    @functools.cached_property
    def filename_in(self):
        return Path(self.path_to_pickle_file).stem

    @functools.cached_property
    def logger(self):
        if self.single_prepared_spectrum:
            return False
        return False if not self.log_output else set_up_logger(parentDirname=self.dirpath,
                                                               filename=self.filename_in,
                                                               method='g+_decomposition')

    @functools.cached_property
    def pickled_data(self):
        say(f"\npickle load '{Path(self.path_to_pickle_file).name}'...", logger=self.logger)
        with open(self.path_to_pickle_file, "rb") as pickle_file:
            return pickle.load(pickle_file, encoding='latin1')

    @functools.cached_property
    def data(self):
        return self.pickled_data['data_list']

    @functools.cached_property
    def channels(self):
        return self.pickled_data['x_values']

    @functools.cached_property
    def errors(self):
        return self.pickled_data['error']

    @functools.cached_property
    def header(self):
        return correct_header(self.pickled_data['header']) if 'header' in self.pickled_data.keys() else None

    @functools.cached_property
    def wcs(self):
        return None if self.header is None else WCS(self.header)

    @functools.cached_property
    def velocity_increment(self):
        return None if self.header is None else (self.wcs.wcs.cdelt[2] * self.wcs.wcs.cunit[2]).to(self.vel_unit).value

    @functools.cached_property
    def location(self):
        return self.pickled_data['location'] if 'location' in self.pickled_data.keys() else None

    @functools.cached_property
    def nan_mask(self):
        return self.pickled_data['nan_mask'] if 'nan_mask' in self.pickled_data.keys() else None

    @functools.cached_property
    def fitting(self):
        return {
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
            'min_pvalue': self.min_pvalue,
            'max_ncomps': self.max_ncomps
        }

    # @functools.cached_property
    # def testing(self):
    #     return self.pickled_data['testing'] if 'testing' in self.pickled_data.keys() else None

    def initialize_data(self):
        if 'testing' in self.pickled_data.keys():
            self.testing = self.pickled_data['testing']
            if self.testing:
                self.use_ncpus = 1

    def check_settings(self):
        if self.path_to_pickle_file is None:
            raise Exception("Need to specify 'path_to_pickle_file'")

        if self.main_beam_efficiency is None:
            warnings.warn('assuming intensities are already corrected for main beam efficiency')

        warnings.warn(f"converting velocity values to {self.vel_unit}")

    def decompose(self):
        if self.single_prepared_spectrum:
            self.testing = True
            self.use_ncpus = 1
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
        self.snr_negative = self.snr if self.snr_negative is None else self.snr_negative
        self.snr_fit = self.snr / 2 if self.snr_fit is None else self.snr_fit
        self.snr_thresh = self.snr if self.snr_thresh is None else self.snr_thresh
        self.snr2_thresh = self.snr if self.snr2_thresh is None else self.snr2_thresh

        string_gausspy = str('\ndecomposition settings:'
                             '\nGaussPy:'
                             f'\nTwo phase decomposition: {self.two_phase_decomposition}'
                             f'\nalpha1: {self.alpha1}'
                             f'\nalpha2: {self.alpha2}'
                             f'\nSNR1: {self.snr_thresh}'
                             f'\nSNR2: {self.snr2_thresh}')
        say(string_gausspy, logger=self.logger)

        if self.fitting['improve_fitting']:
            string_gausspy_plus = '\n' + '\n'.join([f'\n{key}: {value}' for key, value in self.fitting.items()])
        else:
            string_gausspy_plus = f"\nimprove_fitting: {self.fitting['improve_fitting']}"
        say(string_gausspy_plus, logger=self.logger)

    def start_decomposition(self):
        if self.alpha1 is None:
            raise Exception("Need to specify 'alpha1' for decomposition.")

        if self.two_phase_decomposition and (self.alpha2 is None):
            raise Exception(
                "Need to specify 'alpha2' for 'two_phase_decomposition'.")

        self.decomposition_settings()
        say('\ndecomposing data...', logger=self.logger)

        from gausspyplus.gausspy_py3 import gp as gp
        decomposer = gp.GaussianDecomposer()  # Load GaussPy
        decomposer.set('use_ncpus', self.use_ncpus)
        decomposer.set('SNR_thresh', self.snr_thresh)
        decomposer.set('SNR2_thresh', self.snr2_thresh)
        decomposer.set('improve_fitting_dict', self.fitting)
        decomposer.set('alpha1', self.alpha1)

        if self.testing:
            decomposer.set('verbose', True)
            decomposer.set('plot', True)

        if self.two_phase_decomposition:
            decomposer.set('phase', 'two')
            decomposer.set('alpha2', self.alpha2)
        else:
            decomposer.set('phase', 'one')

        if self.single_prepared_spectrum:
            return decomposer.batch_decomposition(dct=self.single_prepared_spectrum)

        self.decomposition = decomposer.batch_decomposition(self.path_to_pickle_file)

        self.save_final_results()

        if self.save_initial_guesses:
            self.save_initial_guesses()

    def save_initial_guesses(self):
        say('\npickle dump GaussPy initial guesses...', logger=self.logger)

        filename = f'{self.filename_in}{self.suffix}_fit_ini.pickle'
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

        dct_final_guesses = {
            key: self.decomposition[key]
            for key in [
                "index_fit",
                "best_fit_rchi2",
                "best_fit_aicc",
                "pvalue",
                "amplitudes_fit",
                "amplitudes_fit_err",
                "fwhms_fit",
                "fwhms_fit_err",
                "means_fit",
                "means_fit_err",
                "log_gplus",
                "N_neg_res_peak",
                "N_blended",
                "N_components",
                "quality_control",
            ]
        }

        dct_final_guesses["gausspy_settings"] = dct_gausspy_settings

        dct_final_guesses["improve_fit_settings"] = self.fitting

        filename = f'{self.filename_in}{self.suffix}_fit_fin.pickle'
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
        say(f'\ncreate {mode} cube...', logger=self.logger)

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
            comment = 'Fit component with highest amplitude per spectrum.'
            filename = f"{self.filename}{self.suffix}_main.fits"
        elif mode == 'integrated_intensity':
            comment = 'Integrated intensity of fit component at VLSR position.'
            filename = f"{self.filename}{self.suffix}_wco.fits"
        elif mode == 'full_decomposition':
            comment = 'Recreated dataset from fit components.'
            filename = f"{self.filename}{self.suffix}_decomp.fits"

        array[self.nan_mask] = np.nan

        comments = ['GaussPy+ decomposition results:']
        comments.append(comment)
        if self.main_beam_efficiency is not None:
            comments.append('Corrected for main beam efficiency of {}.'.format(
                self.main_beam_efficiency))

        header = update_header(
            self.header.copy(), comments=comments, write_meta=True)

        pathToFile = os.path.join(self.decomp_dirname, 'FITS', filename)
        save_fits(array, header, pathToFile, verbose=False)
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

        filename = f'{self.filename}{self.suffix}_wco.dat'
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

        filename = f"{self.filename}{self.suffix}_component_map.fits"
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

        filename = f"{self.filename}{self.suffix}_rchi2_map.fits"
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

        filename = f"{self.filename}{self.suffix}_{mode}_veldisp_map.fits"
        pathToFile = os.path.join(
            os.path.dirname(self.dirname), 'gpy_maps', filename)

        save_fits(data.astype(dtype), header, pathToFile, verbose=False)
        say(">> saved {} velocity dispersion map '{}' in {}".format(
            mode, filename, os.path.dirname(pathToFile)), logger=self.logger)
