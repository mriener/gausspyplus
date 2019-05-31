# @Author: riener
# @Date:   2019-02-18T16:27:12+01:00
# @Filename: training_set.py
# @Last modified by:   riener

# @Last modified time: 2019-04-08T11:59:28+02:00

import itertools
import os
import pickle
import random

import numpy as np

from astropy.io import fits
from astropy.modeling import models, fitting, optimizers
from scipy.signal import argrelextrema

from .config_file import get_values_from_config_file
from .utils.determine_intervals import get_signal_ranges, get_noise_spike_ranges
from .utils.fit_quality_checks import determine_significance, goodness_of_fit,\
    check_residual_for_normality
from .utils.gaussian_functions import gaussian
from .utils.noise_estimation import get_max_consecutive_channels, mask_channels, determine_noise
from .utils.output import check_if_all_values_are_none
from .utils.spectral_cube_functions import remove_additional_axes


class GaussPyTrainingSet(object):
    def __init__(self, config_file=''):
        self.path_to_file = None
        self.filename = None
        self.dirpath_gpy = None
        self.filename_out = None

        self.n_spectra = 5
        self.order = 6
        self.snr = 3
        self.significance = 5
        self.min_fwhm = 1.
        self.max_fwhm = None
        self.p_limit = 0.02
        self.signal_mask = True
        self.pad_channels = 5
        self.min_channels = 100
        self.snr_noise_spike = 5.
        self.min_pvalue = 0.01
        # TODO: also define lower limit for rchi2 to prevent overfitting?
        self.rchi2_limit = 1.5
        self.use_all = False
        self.save_all = False
        self.mask_out_ranges = []

        self.verbose = True
        self.suffix = ''
        self.use_ncpus = None
        self.random_seed = 111

        if config_file:
            get_values_from_config_file(
                self, config_file, config_key='training')

    def check_settings(self):
        check_if_all_values_are_none(self.path_to_file, self.dirpath_gpy,
                                     'path_to_file', 'dirpath_gpy')
        check_if_all_values_are_none(self.path_to_file, self.filename,
                                     'path_to_file', 'filename')

    def initialize(self):
        self.minStddev = None
        if self.min_fwhm is not None:
            self.minStddev = self.min_fwhm/2.355

        self.maxStddev = None
        if self.max_fwhm is not None:
            self.maxStddev = self.max_fwhm/2.355

        if self.path_to_file is not None:
            self.dirname = os.path.dirname(self.path_to_file)
            self.filename = os.path.basename(self.path_to_file)

        if self.dirpath_gpy is not None:
            self.dirname = self.dirpath_gpy

        self.filename, self.file_extension = os.path.splitext(self.filename)

        if self.filename_out is None:
            self.filename_out = '{}-training_set-{}_spectra{}.pickle'.format(
                self.filename, self.n_spectra, self.suffix)
        elif not self.filename_out.endswith('.pickle'):
            self.filename_out = self.filename_out + '.pickle'

        self.header = None

        if self.file_extension == '.fits':
            hdu = fits.open(self.path_to_file)[0]
            self.data = hdu.data
            self.header = hdu.header

            self.data, self.header = remove_additional_axes(
                self.data, self.header)
            self.n_channels = self.data.shape[0]
        else:
            with open(os.path.join(self.path_to_file), "rb") as pickle_file:
                dctData = pickle.load(pickle_file, encoding='latin1')
            self.data = dctData['data_list']
            self.n_channels = len(self.data[0])

        self.channels = np.arange(self.n_channels)

    def say(self, message):
        """Diagnostic messages."""
        # if self.log_output:
        #     self.logger.info(message)
        if self.verbose:
            print(message)

    def decompose_spectra(self):
        self.initialize()
        if self.verbose:
            print("decompose {} spectra ...".format(self.n_spectra))

        if self.random_seed is not None:
            random.seed(self.random_seed)

        data = {}

        self.mask_omit = mask_channels(self.n_channels, self.mask_out_ranges)

        self.max_consecutive_channels = get_max_consecutive_channels(self.n_channels, self.p_limit)

        if self.header:
            yValues = np.arange(self.data.shape[1])
            xValues = np.arange(self.data.shape[2])
            nSpectra = yValues.size * xValues.size
            self.locations = list(itertools.product(yValues, xValues))
            indices = random.sample(list(range(nSpectra)), nSpectra)
        else:
            nSpectra = len(self.data)
            indices = random.sample(list(range(nSpectra)), nSpectra)
            # indices = np.array([4506])  # for testing

        if self.use_all:
            self.n_spectra = nSpectra
            self.filename_out = '{}-training_set-{}_spectra{}.pickle'.format(
                self.filename, self.n_spectra, self.suffix)

        import gausspyplus.parallel_processing
        gausspyplus.parallel_processing.init([indices, [self]])

        results_list = gausspyplus.parallel_processing.func_ts(
            self.n_spectra, use_ncpus=self.use_ncpus)
        print('SUCCESS\n')

        for result in results_list:
            if result is None:
                continue
            fit_values, spectrum, location, signal_ranges, rms, rchi2, pvalue, index, i = result
            # the next four lines are added to deal with the use_all=True feature
            if rchi2 is None:
                continue
            if not self.save_all and (rchi2 > self.rchi2_limit):
            # if not self.save_all and (pvalue < self.min_pvalue):
                continue
            amps, fwhms, means = ([] for i in range(3))
            if fit_values is not None:
                for item in fit_values:
                    amps.append(item[0])
                    means.append(item[1])
                    fwhms.append(item[2]*2.355)

            data['data_list'] = data.get('data_list', []) + [spectrum]
            if self.header:
                data['location'] = data.get('location', []) + [location]
            data['index'] = data.get('index', []) + [index]
            data['error'] = data.get('error', []) + [[rms]]
            data['best_fit_rchi2'] = data.get('best_fit_rchi2', []) + [rchi2]
            data['pvalue'] = data.get('pvalue', []) + [pvalue]
            data['amplitudes'] = data.get('amplitudes', []) + [amps]
            data['fwhms'] = data.get('fwhms', []) + [fwhms]
            data['means'] = data.get('means', []) + [means]
            data['signal_ranges'] = data.get('signal_ranges', []) + [signal_ranges]
        data['x_values'] = self.channels
        if self.header:
            data['header'] = self.header

        dirname = os.path.join(self.dirname, 'gpy_training')
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        path_to_file = os.path.join(dirname, self.filename_out)
        pickle.dump(data, open(path_to_file, 'wb'), protocol=2)
        self.say("\n\033[92mSAVED FILE:\033[0m '{}' in '{}'".format(self.filename_out, dirname))

    def decompose(self, index, i):
        if self.header:
            location = self.locations[index]
            spectrum = self.data[:, location[0], location[1]].copy()
        else:
            location = None
            spectrum = self.data[index].copy()

        if self.mask_out_ranges:
            nan_mask = mask_channels(self.n_channels, self.mask_out_ranges)
            spectrum[nan_mask] = np.nan

        rms = determine_noise(
            spectrum, max_consecutive_channels=self.max_consecutive_channels,
            pad_channels=self.pad_channels, idx=index, average_rms=None)

        if np.isnan(rms):
            return None

        noise_spike_ranges = get_noise_spike_ranges(
            spectrum, rms, snr_noise_spike=self.snr_noise_spike)
        if self.mask_out_ranges:
            noise_spike_ranges += self.mask_out_ranges

        signal_ranges = get_signal_ranges(
            spectrum, rms, snr=self.snr, significance=self.significance,
            pad_channels=self.pad_channels, min_channels=self.min_channels,
            remove_intervals=noise_spike_ranges)

        if signal_ranges:
            mask_signal = mask_channels(self.n_channels, signal_ranges)
        else:
            mask_signal = None

        maxima = self.get_maxima(spectrum, rms)
        fit_values, rchi2, pvalue = self.gaussian_fitting(
            spectrum, maxima, rms, mask_signal=mask_signal)
        # TODO: change the rchi2_limit value??
        # if ((fit_values is not None) and (pvalue > self.min_pvalue)) or self.use_all:
        if ((fit_values is not None) and (rchi2 < self.rchi2_limit)) or self.use_all:
            return [fit_values, spectrum, location, signal_ranges, rms,
                    rchi2, pvalue, index, i]
        else:
            return None

    def get_maxima(self, spectrum, rms):
        array = spectrum.copy()
        #  set all spectral data points below threshold to zero
        low_values = array < self.snr*rms
        array[low_values] = 0
        #  find local maxima (order of x considers x neighboring data points)
        maxima = argrelextrema(array, np.greater, order=self.order)
        return maxima

    def gaussian_fitting(self, spectrum, maxima, rms, mask_signal=None):
        # TODO: don't hardcode the value of stddev_ini
        stddev_ini = 2  # in channels

        gaussians = []
        # loop through spectral channels of the local maxima, fit Gaussians
        sortedAmps = np.argsort(spectrum[maxima])[::-1]

        for idx in sortedAmps:
            mean, amp = maxima[0][idx], spectrum[maxima][idx]
            gauss = models.Gaussian1D(amp, mean, stddev_ini)
            gauss.bounds['amplitude'] = (None, 1.1*amp)
            gaussians.append(gauss)

        improve = True
        while improve is True:
            fit_values = self.determine_gaussian_fit_models(
                gaussians, spectrum)
            if fit_values is not None:
                improve, gaussians = self.check_fit_parameters(
                        fit_values, gaussians, rms)
            else:
                improve = False

        if fit_values is not None:
            comps = len(fit_values)
        else:
            comps = 0

        channels = np.arange(len(spectrum))
        if comps > 0:
            for j in range(len(fit_values)):
                gauss = gaussian(
                    fit_values[j][0], fit_values[j][2]*2.355, fit_values[j][1], channels)
                if j == 0:
                    combined_gauss = gauss
                else:
                    combined_gauss += gauss
        else:
            combined_gauss = np.zeros(len(channels))
        if comps > 0:
            rchi2 = goodness_of_fit(spectrum, combined_gauss, rms, comps, mask=mask_signal)
        else:
            rchi2 = None

        pvalue = check_residual_for_normality(
            spectrum - combined_gauss, rms, mask=mask_signal)

        return fit_values, rchi2, pvalue

    def check_fit_parameters(self, fit_values, gaussians, rms):
        improve = False
        revised_gaussians = gaussians.copy()
        for initial_guess, final_fit in zip(gaussians, fit_values):
            if (final_fit[0] < self.snr*rms):
                revised_gaussians.remove(initial_guess)
                improve = True
                break

            if final_fit[2] <= 0:
                print('negative!')
                # TODO: remove this negative Gaussian
            significance = determine_significance(
                final_fit[0], final_fit[2]*2.35482, rms)
            if significance < self.significance:
                revised_gaussians.remove(initial_guess)
                improve = True
                break

            if self.maxStddev is not None:
                if final_fit[2] > self.maxStddev:
                    revised_gaussians.remove(initial_guess)
                    improve = True
                    break

            if self.minStddev is not None:
                if final_fit[2] < self.minStddev:
                    revised_gaussians.remove(initial_guess)
                    improve = True
                    break

        if improve:
            gaussians = revised_gaussians
        return improve, gaussians

    def determine_gaussian_fit_models(self, gaussians, spectrum):
        fit_values = None
        optimizers.DEFAULT_MAXITER = 1000
        channels = np.arange(self.n_channels)

        # To fit the data create a new superposition with initial
        # guesses for the parameters:
        if len(gaussians) > 0:
            gg_init = gaussians[0]

            if len(gaussians) > 1:
                for i in range(1, len(gaussians)):
                    gg_init += gaussians[i]

            fitter = fitting.SLSQPLSQFitter()
            try:
                gg_fit = fitter(gg_init, channels, spectrum, disp=False)
            except TypeError:
                gg_fit = fitter(gg_init, channels, spectrum, verblevel=False)

            fit_values = []
            if len(gg_fit.param_sets) > 3:
                for i in range(len(gg_fit.submodel_names)):
                    fit_values.append([gg_fit[i].amplitude.value,
                                       gg_fit[i].mean.value,
                                       abs(gg_fit[i].stddev.value)])
            else:
                fit_values.append([gg_fit.amplitude.value,
                                   gg_fit.mean.value,
                                   abs(gg_fit.stddev.value)])
        return fit_values


if __name__ == "__main__":
    ''
