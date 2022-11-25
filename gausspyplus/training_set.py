import functools
import itertools
import os
import pickle
import random
from pathlib import Path
from typing import Optional, List

import numpy as np

from astropy.io import fits
from astropy.modeling import models, fitting, optimizers
from scipy.signal import argrelextrema

from gausspyplus.definitions.config_file import get_values_from_config_file
from gausspyplus.definitions.checks import BaseChecks
from gausspyplus.utils.determine_intervals import (
    get_signal_ranges,
    get_noise_spike_ranges,
)
from gausspyplus.decomposition.fit_quality_checks import (
    determine_significance,
    goodness_of_fit,
)
from gausspyplus.decomposition.gaussian_functions import (
    multi_component_gaussian_model,
    CONVERSION_STD_TO_FWHM,
)
from gausspyplus.utils.noise_estimation import (
    determine_maximum_consecutive_channels,
    mask_channels,
    determine_noise,
)
from gausspyplus.utils.spectral_cube_functions import remove_additional_axes
from gausspyplus.definitions.definitions import (
    FitResults,
    SettingsDefault,
    SettingsTraining,
)


optimizers.DEFAULT_MAXITER = 1000  # set maximum iterations for SLSQPLSQFitter


class GaussPyTrainingSet(SettingsDefault, SettingsTraining, BaseChecks):
    def __init__(self, config_file=""):
        self.path_to_file = None
        self.path_to_noise_map = None
        self.filename = None
        self.dirpath_gpy = None
        self.filename_out = None

        # TODO: also define lower limit for rchi2 to prevent overfitting?
        self.save_all = False

        # TODO: Can the amp_threshold attribute be replaced or covered by another attribute?
        self.amp_threshold = None

        if config_file:
            get_values_from_config_file(self, config_file, config_key="training")

    @functools.cached_property
    def min_stddev(self):
        return None if self.min_fwhm is None else self.min_fwhm / CONVERSION_STD_TO_FWHM

    @functools.cached_property
    def max_stddev(self):
        return None if self.max_fwhm is None else self.max_fwhm / CONVERSION_STD_TO_FWHM

    @functools.cached_property
    def noise_map(self):
        return (
            None
            if self.path_to_noise_map is None
            else fits.getdata(self.path_to_noise_map)
        )

    @functools.cached_property
    def dirpath(self):
        return (
            self.dirpath_gpy
            if self.dirpath_gpy is not None
            else Path(self.path_to_file).parent
        )

    @functools.cached_property
    def filename_in(self):
        return Path(self.filename or self.path_to_file).stem

    @functools.cached_property
    def input_file_type(self):
        return Path(self.filename or self.path_to_file).suffix

    @functools.cached_property
    def n_channels(self):
        return (
            self.data.shape[0] if self.input_file_type == ".fits" else len(self.data[0])
        )

    @functools.cached_property
    def channels(self):
        return np.arange(self.n_channels)

    @functools.cached_property
    def input_object(self):
        if self.input_file_type == ".fits":
            hdu = fits.open(self.path_to_file)[0]
            data, header = remove_additional_axes(hdu.data, hdu.header)
            return fits.PrimaryHDU(data=data, header=header)
        elif self.input_file_type == ".pickle":
            with open(self.path_to_file, "rb") as pickle_file:
                return pickle.load(pickle_file, encoding="latin1")

    @functools.cached_property
    def data(self):
        return (
            self.input_object.data
            if self.input_file_type == ".fits"
            else self.input_object["data_list"]
        )

    @functools.cached_property
    def header(self):
        return self.input_object.header if self.input_file_type == ".fits" else None

    @functools.cached_property
    def max_consecutive_channels(self):
        return determine_maximum_consecutive_channels(self.n_channels, self.p_limit)

    @functools.cached_property
    def n_available_spectra(self):
        return (
            self.data.shape[1] * self.data.shape[2]
            if self.input_file_type == ".fits"
            else len(self.data)
        )

    @functools.cached_property
    def locations(self):
        return (
            list(
                itertools.product(range(self.data.shape[1]), range(self.data.shape[2]))
            )
            if self.input_file_type == ".fits"
            else None
        )

    @functools.cached_property
    def threshold_amplitude(self):
        return self.amp_threshold or 0

    def _prepare_training_set(self, results):
        return {
            "data_list": [fit.intensity_values for fit in results],
            "location": [fit.position_yx for fit in results],
            "index": [fit.index for fit in results],
            # TODO: Change rms from list of list to single value
            "error": [[fit.rms_noise] for fit in results],
            "best_fit_rchi2": [fit.reduced_chi2_value for fit in results],
            "amplitudes": [fit.amplitude_values for fit in results],
            "fwhms": [fit.fwhm_values for fit in results],
            "means": [fit.mean_values for fit in results],
            "signal_ranges": [fit.signal_intervals for fit in results],
            "x_values": self.channels,
            "header": self.header,
        }

    def _save_as_pickled_file(self, data):
        (dirpath_out := Path(self.dirpath, "gpy_training")).mkdir(
            exist_ok=True, parents=True
        )
        filename = (
            self.filename_out
            if self.filename_out is not None
            else f"{self.filename_in}-training_set-{self.n_spectra}_spectra"
            f'{"" if self.suffix is None else self.suffix}.pickle'
        )
        if not filename.endswith(".pickle"):
            filename += ".pickle"

        with open(dirpath_out / filename, "wb") as file:
            pickle.dump(data, file, protocol=2)
        print(f"\n\033[92mSAVED FILE:\033[0m '{filename}' in '{str(dirpath_out)}'")

    def decompose_spectra(self):
        self.raise_exception_if_attribute_is_none("path_to_file")
        self.raise_exception_if_attribute_is_none("rchi2_limit")
        if self.verbose:
            print(f"decompose {self.n_spectra} spectra ...")
        if self.random_seed is not None:
            random.seed(self.random_seed)

        indices = random.sample(
            range(self.n_available_spectra), self.n_available_spectra
        )
        # indices = np.array([4506])  # for testing

        if self.use_all:
            self.n_spectra = self.n_available_spectra

        import gausspyplus.parallel_processing

        gausspyplus.parallel_processing.parallel_processing.init([indices, [self]])

        results = gausspyplus.parallel_processing.parallel_processing.func_ts(
            self.n_spectra, use_ncpus=self.use_ncpus
        )
        print("SUCCESS\n")

        training_set = self._prepare_training_set(
            [result for result in results if result is not None]
        )
        self._save_as_pickled_file(training_set)

    def _get_spectrum(self, index):
        if self.input_file_type == ".fits":
            y_position, x_position = self.locations[index]
            spectrum = self.data[:, y_position, x_position].copy()
        else:
            spectrum = self.data[index].copy()
        if self.mask_out_ranges is not None:
            nan_mask = mask_channels(self.n_channels, self.mask_out_ranges)
            spectrum[nan_mask] = np.nan
        return spectrum

    def _get_rms_noise(self, index, spectrum):
        if self.noise_map is not None:
            y_position, x_position = self.locations[index]
            return self.noise_map[y_position, x_position]
        return determine_noise(
            spectrum=spectrum,
            max_consecutive_channels=self.max_consecutive_channels,
            pad_channels=self.pad_channels,
            idx=index,
            average_rms=None,
        )

    def decompose(self, index, i):
        # TODO: is the variable i needed here?
        spectrum = self._get_spectrum(index)
        rms_noise = self._get_rms_noise(index, spectrum)
        # If no noise can be determined from the spectrum, we cannot fit the spectrum and skip the remaining steps
        if np.isnan(rms_noise):
            return None

        # We cannot deal with NaN values in the spectrum, so we replace them with random values sampled from the noise
        nans = np.isnan(spectrum)
        spectrum[nans] = np.random.randn(len(spectrum[nans])) * rms_noise

        noise_spike_ranges = get_noise_spike_ranges(
            spectrum, rms_noise, snr_noise_spike=self.snr_noise_spike
        )
        if self.mask_out_ranges is not None:
            noise_spike_ranges += self.mask_out_ranges

        signal_ranges = get_signal_ranges(
            spectrum,
            rms_noise,
            snr=self.snr,
            significance=self.significance,
            pad_channels=self.pad_channels,
            min_channels=self.min_channels,
            remove_intervals=noise_spike_ranges,
        )

        fit_values = self.gaussian_fitting(spectrum, rms_noise)

        n_comps = len(fit_values)
        amplitude_values = [fit_params[0] for fit_params in fit_values]
        fwhm_values = [
            fit_params[2] * CONVERSION_STD_TO_FWHM for fit_params in fit_values
        ]
        mean_values = [fit_params[1] for fit_params in fit_values]
        modelled_spectrum = multi_component_gaussian_model(
            amps=amplitude_values, fwhms=fwhm_values, means=mean_values, x=self.channels
        )
        mask_signal = (
            mask_channels(self.n_channels, signal_ranges) if signal_ranges else None
        )
        rchi2 = (
            None
            if n_comps == 0
            else goodness_of_fit(
                data=spectrum,
                best_fit_final=modelled_spectrum,
                errors=rms_noise,
                ncomps_fit=n_comps,
                mask=mask_signal,
            )
        )
        # TODO: change the rchi2_limit value??
        # TODO: if self.use_all is True then fit_values needs to be None instead of []
        if self.use_all or (
            fit_values
            and rchi2 < self.rchi2_limit
            and max(amplitude_values) > self.threshold_amplitude
        ):
            return FitResults(
                amplitude_values=amplitude_values,
                mean_values=mean_values,
                fwhm_values=fwhm_values,
                intensity_values=spectrum,
                position_yx=self.locations[index],
                signal_intervals=signal_ranges,
                rms_noise=rms_noise,
                reduced_chi2_value=rchi2,
                index=index,
            )
        else:
            return None

    def _get_maxima(self, spectrum: np.ndarray, rms: float) -> np.ndarray:
        """Set intensity values below threshold to zero and find local maxima.

        The value of order defines how many neighboring spectral channels are considered for the comparison.
        """
        return argrelextrema(
            data=np.where(spectrum < self.snr * rms, 0, spectrum),
            comparator=np.greater,
            order=self.order,
        )[0]

    def gaussian_fitting(self, spectrum, rms):
        initial_gaussian_models = []
        for idx in self._get_maxima(spectrum, rms):
            initial_gaussian_model = models.Gaussian1D(
                amplitude=spectrum[idx], mean=idx, stddev=2
            )
            initial_gaussian_model.bounds["amplitude"] = (None, 1.1 * spectrum[idx])
            initial_gaussian_models.append(initial_gaussian_model)

        improve = True
        while improve:
            fit_values = self.determine_gaussian_fit_models(
                initial_gaussian_models, spectrum
            )
            if fit_values:
                improve, initial_gaussian_models = self.check_fit_parameters(
                    fit_values, initial_gaussian_models, rms
                )
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
            significance = determine_significance(
                amp=amp, fwhm=stddev * CONVERSION_STD_TO_FWHM, rms=rms
            )
            if significance < self.significance:
                revised_gaussians.remove(initial_guess)
                improve = True
                break

            if self.max_stddev is not None and stddev > self.max_stddev:
                revised_gaussians.remove(initial_guess)
                improve = True
                break

            if self.min_stddev is not None and stddev < self.min_stddev:
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
                fit_values.append(
                    [
                        final_fit_model[i].amplitude.value,
                        final_fit_model[i].mean.value,
                        abs(final_fit_model[i].stddev.value),
                    ]
                )
        else:
            fit_values.append(
                [
                    final_fit_model.amplitude.value,
                    final_fit_model.mean.value,
                    abs(final_fit_model.stddev.value),
                ]
            )
        return fit_values

    def determine_gaussian_fit_models(
        self, gaussians, spectrum: np.ndarray
    ) -> List[Optional[List]]:
        """Return list of fit parameters [[amp_1, mean_1, stddev_1], ... [amp_N, mean_N, stddev_N]]."""
        if len(gaussians) == 0:
            return []
        initial_fit_model = GaussPyTrainingSet._get_initial_fit_model(gaussians)
        final_fit_model = GaussPyTrainingSet._perform_fit(
            initial_fit_model=initial_fit_model,
            channels=np.arange(self.n_channels),
            spectrum=spectrum,
        )
        return GaussPyTrainingSet._get_fit_parameters(final_fit_model=final_fit_model)


if __name__ == "__main__":
    ROOT = Path(os.path.realpath(__file__)).parents[0]
    data = fits.getdata(ROOT / "data" / "grs-test_field.fits")
    # spectrum = data[:, 26, 8]
    # results = determine_peaks(spectrum, amp_threshold=0.4)
    spectrum = data[:, 31, 40]
    training_set = GaussPyTrainingSet()
    rms = 0.10634302494716603
    training_set.n_channels = spectrum.size
    # training_set.maxStddev = training_set.max_fwhm / CONVERSION_STD_TO_FWHM if training_set.max_fwhm is not None else None
    # training_set.minStddev = training_set.min_fwhm / CONVERSION_STD_TO_FWHM if training_set.min_fwhm is not None else None
    maxima = training_set._get_maxima(spectrum, rms)
    fit_values = training_set.gaussian_fitting(spectrum, rms)
    print(fit_values)
