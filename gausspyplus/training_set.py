import itertools
import os
import pickle
import random
from collections import namedtuple
from pathlib import Path
from typing import Optional, List

import numpy as np

from astropy.io import fits
from astropy.modeling import models, fitting, optimizers
from scipy.signal import argrelextrema

from gausspyplus.config_file import get_values_from_config_file
from gausspyplus.utils.determine_intervals import get_signal_ranges, get_noise_spike_ranges
from gausspyplus.utils.fit_quality_checks import determine_significance, goodness_of_fit
from gausspyplus.utils.gaussian_functions import combined_gaussian
from gausspyplus.utils.noise_estimation import determine_maximum_consecutive_channels, mask_channels, determine_noise
from gausspyplus.utils.output import check_if_all_values_are_none
from gausspyplus.utils.spectral_cube_functions import remove_additional_axes
from gausspyplus.definitions import FitResults


optimizers.DEFAULT_MAXITER = 1000  # set maximum iterations for SLSQPLSQFitter


class GaussPyTrainingSet(object):
    def __init__(self, config_file=''):
        self.path_to_file = None
        self.path_to_noise_map = None
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
        # TODO: also define lower limit for rchi2 to prevent overfitting?
        self.rchi2_limit = 1.5
        self.use_all = False
        self.save_all = False
        self.mask_out_ranges = []

        self.amp_threshold = None

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

        self.noise_map = None
        if self.path_to_noise_map is not None:
            self.noise_map = fits.getdata(self.path_to_noise_map)

    def say(self, message):
        """Diagnostic messages."""
        # if self.log_output:
        #     self.logger.info(message)
        if self.verbose:
            print(message)

    def _save_result(self, data):
        (dirpath_out := Path(self.dirname, 'gpy_training')).mkdir(exist_ok=True, parents=True)
        filename = (self.filename_out if self.filename_out is not None
                    else f'{self.filename}-training_set-{self.n_spectra}_spectra{self.suffix}.pickle')
        if not filename.endswith('.pickle'):
            filename += '.pickle'

        with open(dirpath_out / filename, 'wb') as file:
            pickle.dump(data, file, protocol=2)
        self.say(f"\n\033[92mSAVED FILE:\033[0m '{filename}' in '{str(dirpath_out)}'")


    def decompose_spectra(self):
        self.initialize()
        if self.verbose:
            print(f"decompose {self.n_spectra} spectra ...")

        if self.random_seed is not None:
            random.seed(self.random_seed)

        data = {}

        self.mask_omit = mask_channels(self.n_channels, self.mask_out_ranges)

        self.max_consecutive_channels = determine_maximum_consecutive_channels(self.n_channels, self.p_limit)

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

        import gausspyplus.parallel_processing
        gausspyplus.parallel_processing.init([indices, [self]])

        results = gausspyplus.parallel_processing.func_ts(
            self.n_spectra, use_ncpus=self.use_ncpus)
        print('SUCCESS\n')

        fit_results: Optional[namedtuple]

        for fit_results in results:
            if fit_results is None:
                continue
            # the next four lines are added to deal with the use_all=True feature
            if fit_results.reduced_chi2_value is None:
                continue
            if not self.save_all and (fit_results.reduced_chi2_value > self.rchi2_limit):
                continue

            if self.amp_threshold is not None:
                if max(fit_results.amplitude_values) < self.amp_threshold:
                    continue

            data['data_list'] = data.get('data_list', []) + [fit_results.intensity_values]
            if self.header:
                data['location'] = data.get('location', []) + [fit_results.position_yx]
            data['index'] = data.get('index', []) + [fit_results.index]
            # TODO: Change rms from list of list to single value
            data['error'] = data.get('error', []) + [[fit_results.rms_noise]]
            data['best_fit_rchi2'] = data.get('best_fit_rchi2', []) + [fit_results.reduced_chi2_value]
            data['amplitudes'] = data.get('amplitudes', []) + [fit_results.amplitude_values]
            data['fwhms'] = data.get('fwhms', []) + [fit_results.fwhm_values]
            data['means'] = data.get('means', []) + [fit_results.mean_values]
            data['signal_ranges'] = data.get('signal_ranges', []) + [fit_results.signal_intervals]
        data['x_values'] = self.channels
        if self.header:
            data['header'] = self.header

        self._save_result(data)

    def decompose(self, index, i):
        # TODO: is the variable i needed here?
        if self.header:
            location = self.locations[index]
            spectrum = self.data[:, location[0], location[1]].copy()
        else:
            location = None
            spectrum = self.data[index].copy()

        if self.mask_out_ranges:
            nan_mask = mask_channels(self.n_channels, self.mask_out_ranges)
            spectrum[nan_mask] = np.nan

        if self.noise_map is not None:
            rms = self.noise_map[location[0], location[1]]
            nans = np.isnan(spectrum)
            spectrum[nans] = np.random.randn(len(spectrum[nans])) * rms
        else:
            rms = determine_noise(
                spectrum,
                max_consecutive_channels=self.max_consecutive_channels,
                pad_channels=self.pad_channels,
                idx=index,
                average_rms=None)

        if np.isnan(rms):
            return None

        noise_spike_ranges = get_noise_spike_ranges(spectrum, rms, snr_noise_spike=self.snr_noise_spike)
        if self.mask_out_ranges:
            noise_spike_ranges += self.mask_out_ranges

        signal_ranges = get_signal_ranges(spectrum,
                                          rms,
                                          snr=self.snr,
                                          significance=self.significance,
                                          pad_channels=self.pad_channels,
                                          min_channels=self.min_channels,
                                          remove_intervals=noise_spike_ranges)

        fit_values = self.gaussian_fitting(spectrum, rms)

        n_comps = len(fit_values)
        amplitude_values = [fit_params[0] for fit_params in fit_values]
        fwhm_values = [fit_params[2] * 2.355 for fit_params in fit_values]
        mean_values = [fit_params[1] for fit_params in fit_values]
        modelled_spectrum = combined_gaussian(amps=amplitude_values,
                                              fwhms=fwhm_values,
                                              means=mean_values,
                                              x=np.arange(len(spectrum)))
        mask_signal = mask_channels(self.n_channels, signal_ranges) if signal_ranges else None
        rchi2 = None if n_comps == 0 else goodness_of_fit(data=spectrum,
                                                          best_fit_final=modelled_spectrum,
                                                          errors=rms,
                                                          ncomps_fit=n_comps,
                                                          mask=mask_signal)
        # TODO: change the rchi2_limit value??
        # TODO: if self.use_all is True then fit_values needs to be None instead of []
        if (fit_values and (rchi2 < self.rchi2_limit)) or self.use_all:
            return FitResults(amplitude_values=amplitude_values,
                              mean_values=mean_values,
                              fwhm_values=fwhm_values,
                              intensity_values=spectrum,
                              position_yx=location,
                              signal_intervals=signal_ranges,
                              rms_noise=rms,
                              reduced_chi2_value=rchi2,
                              index=index)
        else:
            return None

    def _get_maxima(self, spectrum: np.ndarray, rms: float) -> np.ndarray:
        """Set intensity values below threshold to zero and find local maxima.

        The value of order defines how many neighboring spectral channels are considered for the comparison.
        """
        return argrelextrema(data=np.where(spectrum < self.snr*rms, 0, spectrum),
                             comparator=np.greater,
                             order=self.order)[0]

    def gaussian_fitting(self, spectrum, rms):
        initial_gaussian_models = []
        for idx in self._get_maxima(spectrum, rms):
            initial_gaussian_model = models.Gaussian1D(amplitude=spectrum[idx], mean=idx, stddev=2)
            initial_gaussian_model.bounds['amplitude'] = (None, 1.1 * spectrum[idx])
            initial_gaussian_models.append(initial_gaussian_model)

        improve = True
        while improve:
            fit_values = self.determine_gaussian_fit_models(initial_gaussian_models, spectrum)
            if fit_values:
                improve, initial_gaussian_models = self.check_fit_parameters(fit_values, initial_gaussian_models, rms)
            else:
                improve = False
        return fit_values

    def check_fit_parameters(self, fit_values, gaussians, rms):
        improve = False
        revised_gaussians = gaussians.copy()
        for initial_guess, (amp, mean, stddev) in zip(gaussians, fit_values):
            if amp < self.snr * rms:
                revised_gaussians.remove(initial_guess)
                improve = True
                break
            significance = determine_significance(amp=amp, fwhm=stddev * 2.35482, rms=rms)
            if significance < self.significance:
                revised_gaussians.remove(initial_guess)
                improve = True
                break

            if self.maxStddev is not None and stddev > self.maxStddev:
                revised_gaussians.remove(initial_guess)
                improve = True
                break

            if self.minStddev is not None and stddev < self.minStddev:
                revised_gaussians.remove(initial_guess)
                improve = True
                break

        if improve:
            gaussians = revised_gaussians
        return improve, gaussians

    @staticmethod
    def _get_initial_fit_model(gaussians):
        """Return a superposition of the initial guesses for the Gaussian fits."""
        initial_fit_model = gaussians[0]
        for gaussian_model in gaussians[1:]:
            initial_fit_model += gaussian_model
        return initial_fit_model

    @staticmethod
    def _perform_fit(initial_fit_model, channels, spectrum):
        fitter = fitting.SLSQPLSQFitter()
        try:
            return fitter(initial_fit_model, channels, spectrum, disp=False)
        except TypeError:
            return fitter(initial_fit_model, channels, spectrum, verblevel=False)

    @staticmethod
    def _get_fit_parameters(final_fit_model):
        fit_values = []
        if len(final_fit_model.param_sets) > 3:
            for i in range(len(final_fit_model.submodel_names)):
                fit_values.append([final_fit_model[i].amplitude.value,
                                   final_fit_model[i].mean.value,
                                   abs(final_fit_model[i].stddev.value)])
        else:
            fit_values.append([final_fit_model.amplitude.value,
                               final_fit_model.mean.value,
                               abs(final_fit_model.stddev.value)])
        return fit_values

    def determine_gaussian_fit_models(self, gaussians, spectrum: np.ndarray) -> List[Optional[List]]:
        """Return list of fit parameters [[amp_1, mean_1, stddev_1], ... [amp_N, mean_N, stddev_N]]."""
        if len(gaussians) == 0:
            return []
        initial_fit_model = GaussPyTrainingSet._get_initial_fit_model(gaussians)
        final_fit_model = GaussPyTrainingSet._perform_fit(initial_fit_model=initial_fit_model,
                                                          channels=np.arange(self.n_channels),
                                                          spectrum=spectrum)
        return GaussPyTrainingSet._get_fit_parameters(final_fit_model=final_fit_model)


if __name__ == '__main__':
    ROOT = Path(os.path.realpath(__file__)).parents[0]
    data = fits.getdata(ROOT / 'data' / 'grs-test_field.fits')
    # spectrum = data[:, 26, 8]
    # results = determine_peaks(spectrum, amp_threshold=0.4)
    spectrum = data[:, 31, 40]
    training_set = GaussPyTrainingSet()
    rms = 0.10634302494716603
    training_set.n_channels = spectrum.size
    training_set.maxStddev = training_set.max_fwhm / 2.355 if training_set.max_fwhm is not None else None
    training_set.minStddev = training_set.min_fwhm / 2.355 if training_set.min_fwhm is not None else None
    maxima = training_set._get_maxima(spectrum, rms)
    fit_values = training_set.gaussian_fitting(spectrum, rms)
    print(fit_values)

