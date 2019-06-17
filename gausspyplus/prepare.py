# @Author: riener
# @Date:   2019-02-26T16:38:04+01:00
# @Filename: prepare.py
# @Last modified by:   riener
# @Last modified time: 10-04-2019

import os
import pickle
import itertools

import numpy as np

from astropy.io import fits
from tqdm import tqdm

from .config_file import get_values_from_config_file
from .utils.determine_intervals import get_signal_ranges, get_noise_spike_ranges
from .utils.noise_estimation import get_max_consecutive_channels, mask_channels, determine_noise, calculate_average_rms_noise
from .utils.output import set_up_logger, check_if_all_values_are_none, check_if_value_is_none, say
from .utils.spectral_cube_functions import remove_additional_axes, add_noise, change_header, save_fits


class GaussPyPrepare(object):
    def __init__(self, path_to_file=None, hdu=None, filename=None,
                 gpy_dirname=None, config_file=''):
        self.path_to_file = path_to_file
        self.path_to_noise_map = None
        self.hdu = hdu
        self.filename = filename
        self.dirpath_gpy = gpy_dirname

        self.gausspy_pickle = True
        self.testing = False
        self.data_location = None
        self.simulation = False

        self.rms_from_data = True
        self.average_rms = None
        self.n_spectra_rms = 1000
        self.p_limit = 0.02
        self.pad_channels = 5
        self.signal_mask = True
        self.min_channels = 100
        self.mask_out_ranges = []

        self.snr = 3.
        self.significance = 5.
        self.snr_noise_spike = 5.

        self.suffix = ''
        self.use_ncpus = None
        self.log_output = True
        self.verbose = True
        self.overwrite = True
        self.random_seed = 111

        if config_file:
            get_values_from_config_file(
                self, config_file, config_key='preparation')

    def check_settings(self):
        text = "specify 'data_location' as (y, x) for 'testing'"
        check_if_value_is_none(self.testing, self.data_location,
                               'testing', 'data_location',
                               additional_text=text)
        check_if_value_is_none(self.simulation, self.average_rms,
                               'simulation', 'average_rms')
        check_if_all_values_are_none(self.path_to_file, self.hdu,
                                     'path_to_file', 'hdu')
        check_if_all_values_are_none(self.path_to_file, self.dirpath_gpy,
                                     'path_to_file', 'dirpath_gpy')
        check_if_all_values_are_none(self.path_to_file, self.filename,
                                     'path_to_file', 'filename')

    def initialize(self):
        if self.path_to_file is not None:
            self.dirpath = os.path.dirname(self.path_to_file)
            self.filename = os.path.basename(self.path_to_file)
            self.hdu = fits.open(self.path_to_file)[0]

        self.filename, self.file_extension = os.path.splitext(self.filename)

        if self.dirpath_gpy is not None:
            self.dirpath = self.dirpath_gpy

        if not self.testing:
            self.dirpath_pickle = os.path.join(self.dirpath, 'gpy_prepared')
            if not os.path.exists(self.dirpath_pickle):
                os.makedirs(self.dirpath_pickle)

        self.logger = False
        if self.log_output:
            self.logger = set_up_logger(
                self.dirpath, self.filename, method='g+_preparation')

        if self.simulation:
            self.rms_from_data = False
            self.hdu = add_noise(self.average_rms, hdu=self.hdu, get_hdu=True,
                                 random_seed=self.random_seed)

        self.data = self.hdu.data
        self.header = self.hdu.header

        self.data, self.header = remove_additional_axes(self.data, self.header)

        self.errors = np.empty((self.data.shape[1], self.data.shape[2]))

        if self.average_rms is None:
            self.rms_from_data = True

        self.noise_map = None
        if self.path_to_noise_map is not None:
            self.rms_from_data = False
            self.noise_map = fits.getdata(self.path_to_noise_map)

        if self.testing:
            ypos = self.data_location[0]
            xpos = self.data_location[1]
            say('\nTesting: using only pixel at location ({}, {})'.format(
                ypos, xpos), logger=self.logger)
            self.data = self.data[:, ypos, xpos]
            self.data = self.data[:, np.newaxis, np.newaxis]
            self.rms_from_data = False
            self.use_ncpus = 1

        self.n_channels = self.data.shape[0]
        if self.n_channels < self.min_channels:
            self.signal_mask = False

        self.max_consecutive_channels = get_max_consecutive_channels(
            self.n_channels, self.p_limit)

        if self.rms_from_data:
            self.calculate_average_rms_from_data()

    def getting_ready(self):
        string = 'GaussPy preparation'
        banner = len(string) * '='
        heading = '\n' + banner + '\n' + string + '\n' + banner
        say(heading, logger=self.logger)

    def return_single_prepared_spectrum(self, data_location=None):
        if data_location:
            self.data_location = data_location
        self.testing = True
        self.log_output = False
        self.check_settings()
        self.initialize()

        location = (0, 0)
        result = self.calculate_rms_noise(location, 0)
        idx, spectrum, location, rms, signal_ranges, noise_spike_ranges = result

        data = {}
        data['header'] = self.header
        data['nan_mask'] = np.isnan(self.data)
        data['x_values'] = np.arange(self.data.shape[0])
        data['data_list'] = [spectrum]
        data['error'] = [[rms]]
        data['index'] = [idx]
        data['location'] = [location]
        data['signal_ranges'] = [signal_ranges]
        data['noise_spike_ranges'] = [noise_spike_ranges]

        return data

    def prepare_cube(self):
        self.check_settings()
        self.initialize()
        self.getting_ready()

        self.prepare_gausspy_pickle()

    def calculate_average_rms_from_data(self):
        say('\ncalculating average rms from data...', logger=self.logger)

        self.average_rms = calculate_average_rms_noise(
            self.data.copy(), self.n_spectra_rms,
            pad_channels=self.pad_channels, random_seed=self.random_seed,
            max_consecutive_channels=self.max_consecutive_channels)

        say('>> calculated rms value of {:.3f} from data'.format(
            self.average_rms), logger=self.logger)

    def prepare_gausspy_pickle(self):
        say('\npreparing GaussPy cube...', logger=self.logger)

        data = {}
        channels = np.arange(self.data.shape[0])

        if self.testing:
            locations = [(0, 0)]
        else:
            yMax = self.data.shape[1]
            xMax = self.data.shape[2]
            locations = list(itertools.product(range(yMax), range(xMax)))

        data['header'] = self.header
        data['nan_mask'] = np.isnan(self.data)
        data['x_values'] = channels
        data['data_list'], data['error'], data['index'], data['location'] = (
            [] for _ in range(4))

        # if self.signal_mask:
        data['signal_ranges'], data['noise_spike_ranges'] = ([] for _ in range(2))

        import gausspyplus.parallel_processing
        gausspyplus.parallel_processing.init([locations, [self]])

        results_list = gausspyplus.parallel_processing.func(use_ncpus=self.use_ncpus, function='gpy_noise')

        print('SUCCESS\n')

        for i, item in tqdm(enumerate(results_list)):
            if not isinstance(item, list):
                print(i, item)
                continue
            idx, spectrum, (ypos, xpos), error, signal_ranges, noise_spike_ranges =\
                item
            self.errors[ypos, xpos] = error
            data['index'].append(idx)
            data['location'].append((ypos, xpos))

            if not np.isnan(error):
                # TODO: return spectrum = None if spectrum wasn't part nans
                # and changes with randomly rms sampled values
                # then add condition if spectrum is None for next line
                # data['data_list'].append(self.data[:, ypos, xpos])
                data['data_list'].append(spectrum)
                data['error'].append([error])
                data['signal_ranges'].append(signal_ranges)
                data['noise_spike_ranges'].append(noise_spike_ranges)
            else:
                # TODO: rework that so that list is initialized with None values
                # and this condition is obsolete?
                data['data_list'].append(None)
                data['error'].append([None])
                # if self.signal_mask:
                data['signal_ranges'].append(None)
                data['noise_spike_ranges'].append(None)

        if self.testing:
            suffix = '_test_Y{}X{}'.format(
                self.data_location[0], self.data_location[1])
            data['testing'] = self.testing
        else:
            suffix = self.suffix

        say("\npickle dump dictionary...", logger=self.logger)

        if self.gausspy_pickle:
            path_to_file = os.path.join(
                self.dirpath_pickle, '{}{}.pickle'.format(
                    self.filename, suffix))
            pickle.dump(data, open(path_to_file, 'wb'), protocol=2)
            print("\033[92mFor GaussPyDecompose:\033[0m 'path_to_pickle_file' = '{}'".format(path_to_file))

    def calculate_rms_noise(self, location, idx):
        ypos, xpos = location
        spectrum = self.data[:, ypos, xpos].copy()

        signal_ranges, noise_spike_ranges = (None for _ in range(2))

        if self.mask_out_ranges:
            nan_mask = mask_channels(self.n_channels, self.mask_out_ranges)
            spectrum[nan_mask] = np.nan

        #  if spectrum contains nans they will be replaced by noise values
        #  randomly sampled from the calculated rms value
        if self.noise_map is not None:
            rms = self.noise_map[ypos, xpos]
        else:
            rms = determine_noise(
                spectrum, max_consecutive_channels=self.max_consecutive_channels,
                pad_channels=self.pad_channels, idx=idx, average_rms=self.average_rms)

        if not np.isnan(rms):
            noise_spike_ranges = get_noise_spike_ranges(
                spectrum, rms, snr_noise_spike=self.snr_noise_spike)
            if self.mask_out_ranges:
                noise_spike_ranges += self.mask_out_ranges
            if self.signal_mask:
                signal_ranges = get_signal_ranges(
                    spectrum, rms, snr=self.snr, significance=self.significance,
                    pad_channels=self.pad_channels, min_channels=self.min_channels,
                    remove_intervals=noise_spike_ranges)

        return [idx, spectrum, location, rms, signal_ranges, noise_spike_ranges]

    def produce_noise_map(self, dtype='float32'):
        comments = ['noise map']
        header = change_header(self.header.copy(), format='pp',
                               comments=comments)

        filename = "{}{}_noise_map.fits".format(self.filename, self.suffix)
        path_to_file = os.path.join(
            os.path.dirname(self.dirpath_pickle), 'gpy_maps', filename)

        save_fits(self.errors.astype(dtype), header, path_to_file,
                  verbose=False)
        say("\n\033[92mSAVED FILE:\033[0m '{}' in '{}'".format(
            filename, os.path.dirname(path_to_file)), logger=self.logger)
