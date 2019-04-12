# @Author: riener
# @Date:   2018-12-19T17:26:54+01:00
# @Filename: training.py
# @Last modified by:   riener
# @Last modified time: 2019-04-08T10:19:11+02:00

import itertools
import os
import pickle
import random
import warnings

import numpy as np

from astropy.io import fits

from .config_file import get_values_from_config_file
from .utils.gaussian_functions import gaussian
from .utils.output import format_warning
warnings.showwarning = format_warning


class GaussPyTraining(object):
    def __init__(self, config_file=''):
        self.path_to_training_set = None
        self.gpy_dirpath = None

        self.two_phase_decomposition = True
        self.snr = 3.
        self.alpha1_initial = None
        self.alpha2_initial = None
        self.snr_thresh = None
        self.snr2_thresh = None

        self.create_training_set = False
        self.params_from_data = True
        self.n_channels = None
        self.n_spectra = None
        self.ncomps_limits = None
        self.amp_limits = None
        self.fwhm_limits = None
        self.mean_limits = None
        self.rms = None
        self.n_spectra_rms = 5000
        self.n_edge_channels = 10

        self.log_output = True
        self.verbose = True
        self.random_seed = 111

        if config_file:
            get_values_from_config_file(
                self, config_file, config_key='training')

    def initialize(self):
        if self.path_to_training_set is None:
            raise Exception("Need to specify 'path_to_training_set'")

        self.dirname = os.path.dirname(self.path_to_training_set)
        if self.gpy_dirpath is None:
            self.gpy_dirpath = os.path.dirname(self.dirname)
        self.file = os.path.basename(self.path_to_training_set)
        self.filename, self.file_extension = os.path.splitext(self.file)

        if self.snr_thresh is None:
            self.snr_thresh = self.snr
        if self.snr2_thresh is None:
            self.snr2_thresh = self.snr
        if self.alpha1_initial is None:
            self.alpha1_initial = 3.
            warnings.warn(
                'No value for {a} supplied. Setting {a} to {b}.'.format(
                    a='alpha1_initial', b=self.alpha1_initial))
        if self.alpha2_initial is None:
            self.alpha2_initial = 6.
            warnings.warn(
                'No value for {a} supplied. Setting {a} to {b}.'.format(
                    a='alpha2_initial', b=self.alpha2_initial))

    def training(self):
        self.initialize()

        if self.create_training_set:
            self.check_settings()
            self.create_training_set()

        self.gausspy_train_alpha()

    def check_settings(self):
        if self.ncomps_limits is None:
            errorMessage = str("specify 'ncomps_limits' as [minComps, maxComps]")
            raise Exception(errorMessage)

        if self.amp_limits is None:
            errorMessage = str("specify 'amp_limits' as [minAmp, maxAmp]")
            raise Exception(errorMessage)

        if self.mean_limits is None:
            errorMessage = str("specify 'mean_limits' in channels as "
                               "[minMean, maxMean]")
            raise Exception(errorMessage)

        if self.rms is None:
            errorMessage = str("specify 'rms'")
            raise Exception(errorMessage)

        if self.fwhm_limits is None:
            errorMessage = str("specify 'fwhm_limits' in channels as "
                               "[minFwhm, maxFwhm]")
            raise Exception(errorMessage)

        if self.n_channels is None:
            errorMessage = str("specify 'n_channels'")
            raise Exception(errorMessage)

        if self.n_spectra is None:
            errorMessage = str("specify 'nSepctra'")
            raise Exception(errorMessage)

    def get_parameters_from_data(self, pathToFile):
        if self.verbose:
            print("determine parameters from data ...")

        if self.random_seed is not None:
            random.seed(self.random_seed)

        hdu = fits.open(pathToFile)[0]
        data = hdu.data

        self.n_channels = data.shape[0]

        yValues = np.arange(data.shape[1])
        xValues = np.arange(data.shape[2])
        locations = list(itertools.product(yValues, xValues))
        if len(locations) > self.n_spectra_rms:
            locations = random.sample(locations, self.n_spectra_rms)
        rmsList, maxAmps = ([] for i in range(2))
        for y, x in locations:
            spectrum = data[:, y, x]
            if not np.isnan(spectrum).any():
                maxAmps.append(max(spectrum))
                rms = np.std(spectrum[spectrum < abs(np.min(spectrum))])
                rmsList.append(rms)

        self.rms = np.median(rmsList)
        self.amp_limits = [3*self.rms, 0.8*max(maxAmps)]
        self.mean_limits = [0 + self.n_edge_channels,
                         data.shape[0] - self.n_edge_channels]

        if self.verbose:
            print("n_channels = {}".format(self.n_channels))
            print("rms = {}".format(self.rms))
            print("amp_limits = {}".format(self.amp_limits))
            print("mean_limits = {}".format(self.mean_limits))

    def create_training_set(self, training_set=True):
        print('create training set ...')

        # Initialize
        data = {}
        channels = np.arange(self.n_channels)
        error = self.rms

        # Begin populating data
        for i in range(self.n_spectra):
            amps, fwhms, means = ([] for i in range(3))
            spectrum = np.random.randn(self.n_channels) * self.rms

            ncomps = np.random.choice(
                np.arange(self.ncomps_limits[0], self.ncomps_limits[1] + 1))

            for comp in range(ncomps):
                # Select random values for components within specified ranges
                amp = np.random.uniform(self.amp_limits[0], self.amp_limits[1])
                fwhm = np.random.uniform(self.fwhm_limits[0], self.fwhm_limits[1])
                mean = np.random.uniform(self.mean_limits[0], self.mean_limits[1])

                # Add Gaussian with random parameters from above to spectrum
                spectrum += gaussian(amp, fwhm, mean, channels)

                # Append the parameters to initialized lists for storing
                amps.append(amp)
                fwhms.append(fwhm)
                means.append(mean)

            # Enter results into AGD dataset
            data['data_list'] = data.get('data_list', []) + [spectrum]
            data['x_values'] = data.get('x_values', []) + [channels]
            data['error'] = data.get('error', []) + [error]

            # If training data, keep answers
            if training_set:
                data['amplitudes'] = data.get('amplitudes', []) + [amps]
                data['fwhms'] = data.get('fwhms', []) + [fwhms]
                data['means'] = data.get('means', []) + [means]

        # Dump synthetic data into specified filename
        if not os.path.exists(self.dirname):
            os.makedirs(self.dirname)
        pickle.dump(data, open(self.path_to_training_set, 'wb'))

    def gausspy_train_alpha(self):
        from .gausspy_py3 import gp as gp

        g = gp.GaussianDecomposer()

        g.load_training_data(self.path_to_training_set)
        g.set('SNR_thresh', self.snr_thresh)
        g.set('SNR2_thresh', self.snr2_thresh)

        if self.log_output:
            log_output = {'dirname': self.gpy_dirpath, 'filename': self.filename}

        if self.two_phase_decomposition:
            g.set('phase', 'two')  # Set GaussPy parameters
            # Train AGD starting with initial guess for alpha
            g.train(alpha1_initial=self.alpha1_initial, alpha2_initial=self.alpha2_initial,
                    log_output=log_output)
        else:
            g.set('phase', 'one')
            g.train(alpha1_initial=self.alpha1_initial, log_output=log_output)
