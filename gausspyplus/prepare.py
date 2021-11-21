import functools
import os
import pickle
import itertools
from pathlib import Path

import numpy as np

from astropy.io import fits
from tqdm import tqdm

from gausspyplus.config_file import get_values_from_config_file
from gausspyplus.utils.determine_intervals import get_signal_ranges, get_noise_spike_ranges
from gausspyplus.utils.noise_estimation import determine_maximum_consecutive_channels, mask_channels, determine_noise, calculate_average_rms_noise
from gausspyplus.utils.output import set_up_logger, check_if_all_values_are_none, check_if_value_is_none, say
from gausspyplus.utils.spectral_cube_functions import remove_additional_axes, add_noise, change_header, save_fits


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

        if self.testing:
            say(f'\nTesting: using only pixel at location ({self.data_location[0]}, {self.data_location[1]})',
                logger=self.logger)
            self.use_ncpus = 1

        if self.n_channels < self.min_channels:
            self.signal_mask = False

    @functools.cached_property
    def filename_in(self):
        return self.filename if self.filename is not None else Path(self.path_to_file).stem

    @functools.cached_property
    def dirpath(self):
        # TODO: homogenize attributes self.dirpath_gpy (used here) and self.gpy_dirpath (used in training_set)
        return self.dirpath_gpy if self.dirpath_gpy is not None else Path(self.path_to_file).parent

    @functools.cached_property
    def logger(self):
        return False if not self.log_output else set_up_logger(parentDirname=self.dirpath,
                                                               filename=self.filename_in,
                                                               method='g+_preparation')

    @functools.cached_property
    def noise_map(self):
        return None if self.path_to_noise_map is None else fits.getdata(self.path_to_noise_map)

    @functools.cached_property
    def input_object(self):
        hdu = fits.open(self.path_to_file)[0]
        if self.simulation:
            hdu = add_noise(self.average_rms, hdu=hdu, get_hdu=True, random_seed=self.random_seed)
        # TODO: additional axes should be removed before adding noise for self.simulation
        data, header = remove_additional_axes(hdu.data, hdu.header)
        return fits.PrimaryHDU(data=data, header=header)

    @functools.cached_property
    # TODO: what is the problem here?
    def data(self):
        return (np.expand_dims(self.input_object.data[:, self.data_location[0], self.data_location[1]], axis=(1, 2))
                if self.testing else self.input_object.data)

    @functools.cached_property
    def header(self):
        return self.input_object.header

    @functools.cached_property
    def n_channels(self):
        return self.data.shape[0]

    @functools.cached_property
    def max_consecutive_channels(self):
        return determine_maximum_consecutive_channels(self.n_channels, self.p_limit)

    @functools.cached_property
    def dirpath_pickle(self):
        # TODO: is the self.testing condition really necessary?
        if self.testing:
            return None
        (dirpath_pickle := Path(self.dirpath, 'gpy_prepared')).mkdir(parents=True, exist_ok=True)
        return dirpath_pickle

    def getting_ready(self):
        string = 'GaussPy preparation'
        banner = len(string) * '='
        heading = '\n' + banner + '\n' + string + '\n' + banner
        say(heading, logger=self.logger)

    def return_single_prepared_spectrum(self, data_location=None):
        self.data_location = data_location if data_location is not None else self.data_location
        self.testing = True
        self.log_output = False
        self.check_settings()

        result = self.calculate_rms_noise(location=[0, 0], idx=0)
        idx, spectrum, location, rms, signal_ranges, noise_spike_ranges = result

        return {
            'header': self.header,
            'nan_mask': np.isnan(self.data),
            'x_values': np.arange(self.data.shape[0]),
            'data_list': [spectrum],
            'error': [[rms]],
            'index': [idx],
            'location': [location],
            'signal_ranges': [signal_ranges],
            'noise_spike_ranges': [noise_spike_ranges],
        }

    def prepare_cube(self):
        self.check_settings()
        self.getting_ready()
        self.prepare_gausspy_pickle()

    def calculate_average_rms_from_data(self):
        say('\ncalculating average rms from data...', logger=self.logger)
        # TODO: change calculate_average_rms_noise so that it does not change data and no copy of data has to be created
        self.average_rms = calculate_average_rms_noise(data=self.data.copy(),
                                                       number_rms_spectra=self.n_spectra_rms,
                                                       pad_channels=self.pad_channels,
                                                       random_seed=self.random_seed,
                                                       max_consecutive_channels=self.max_consecutive_channels)
        say(f'>> calculated rms value of {self.average_rms:.3f} from data', logger=self.logger)

    def prepare_gausspy_pickle(self):
        say('\npreparing GaussPy cube...', logger=self.logger)

        if not self.simulation and not self.testing and self.path_to_noise_map is None and self.average_rms is None:
            self.calculate_average_rms_from_data()

        data = {}
        channels = np.arange(self.data.shape[0])

        if self.testing:
            locations = [(0, 0)]
        else:
            yMax = self.data.shape[1]
            xMax = self.data.shape[2]
            locations = list(itertools.product(range(yMax), range(xMax)))

        data['header'] = self.header
        # TODO: Info about NaNs can be safed more efficient -> but where is this used? Make sure to change it everywhere
        data['nan_mask'] = np.isnan(self.data)
        data['x_values'] = channels
        data['data_list'], data['error'], data['index'], data['location'] = (
            [] for _ in range(4))

        # if self.signal_mask:
        data['signal_ranges'], data['noise_spike_ranges'] = ([] for _ in range(2))

        import gausspyplus.parallel_processing
        gausspyplus.parallel_processing.init([locations, [self]])

        results_list = gausspyplus.parallel_processing.func(
            use_ncpus=self.use_ncpus,
            function='gpy_noise')

        print('SUCCESS\n')

        # TODO: this is just needed for the noise map -> create this later?
        self.errors = np.empty((self.data.shape[1], self.data.shape[2]))

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
            suffix = f'_test_Y{self.data_location[0]}X{self.data_location[1]}'
            data['testing'] = self.testing
        else:
            suffix = self.suffix

        say("\npickle dump dictionary...", logger=self.logger)

        if self.gausspy_pickle:
            path_to_file = os.path.join(
                self.dirpath_pickle, f'{self.filename_in}{suffix}.pickle')
            pickle.dump(data, open(path_to_file, 'wb'), protocol=2)
            print("\033[92mFor GaussPyDecompose:\033[0m 'path_to_pickle_file' = '{}'".format(path_to_file))

    def _get_rms_noise(self, index, spectrum, location):
        if self.noise_map is not None:
            # TODO: change location to self.locations[index] as in training_set
            y_position, x_position = location
            return self.noise_map[y_position, x_position]
        return determine_noise(spectrum=spectrum,
                               max_consecutive_channels=self.max_consecutive_channels,
                               pad_channels=self.pad_channels,
                               idx=index,
                               average_rms=self.average_rms)

    def _get_noise_spike_ranges(self, spectrum, rms):
        if np.isnan(rms):
            return None
        mask_out_ranges = [] if self.mask_out_ranges is None else self.mask_out_ranges
        return mask_out_ranges + get_noise_spike_ranges(spectrum,
                                                        rms,
                                                        snr_noise_spike=self.snr_noise_spike)

    def _get_signal_ranges(self, spectrum, rms, noise_spike_ranges):
        if np.isnan(rms) or not self.signal_mask:
            return None
        return get_signal_ranges(spectrum,
                                 rms,
                                 snr=self.snr,
                                 significance=self.significance,
                                 pad_channels=self.pad_channels,
                                 min_channels=self.min_channels,
                                 remove_intervals=noise_spike_ranges)

    def calculate_rms_noise(self, location, idx):
        ypos, xpos = location
        spectrum = self.data[:, ypos, xpos].copy()

        if self.mask_out_ranges:
            nan_mask = mask_channels(self.n_channels, self.mask_out_ranges)
            spectrum[nan_mask] = np.nan

        rms = self._get_rms_noise(idx, spectrum, location)

        # TODO: does this make sense here?
        #  if spectrum contains nans they will be replaced by noise values randomly sampled from the calculated rms
        if not np.isnan(rms):
            nans = np.isnan(spectrum)
            spectrum[nans] = np.random.randn(len(spectrum[nans])) * rms

        noise_spike_ranges = self._get_noise_spike_ranges(spectrum, rms)
        signal_ranges = self._get_signal_ranges(spectrum, rms, noise_spike_ranges)

        return [idx, spectrum, location, rms, signal_ranges, noise_spike_ranges]

    def produce_noise_map(self, dtype='float32'):
        comments = ['noise map']
        header = change_header(self.header.copy(), format='pp',
                               comments=comments)

        filename = f"{self.filename_in}{self.suffix}_noise_map.fits"
        path_to_file = os.path.join(
            os.path.dirname(self.dirpath_pickle), 'gpy_maps', filename)

        save_fits(self.errors.astype(dtype), header, path_to_file,
                  verbose=False)
        say("\n\033[92mSAVED FILE:\033[0m '{}' in '{}'".format(
            filename, os.path.dirname(path_to_file)), logger=self.logger)


if __name__ == '__main__':
    pass
