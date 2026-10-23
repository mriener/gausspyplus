"""Tests for the Spectrum and Model dataclasses in definitions/spectrum.py and definitions/model.py."""
import numpy as np
import pytest

from gausspyplus.definitions.model import Model
from gausspyplus.definitions.spectrum import Spectrum

N_CHANNELS = 100
CHANNELS = np.arange(N_CHANNELS)
RMS_NOISE = 1.0
# Gaussian component (amp=1, fwhm=12, mean=50) evaluated on CHANNELS
GAUSSIAN = np.exp(-4.0 * np.log(2) * (CHANNELS - 50) ** 2 / 12**2)
# deterministic fake noise: alternating +0.05/-0.05 pattern
RESIDUAL_NOISE = 0.05 * np.where(CHANNELS % 2 == 0, 1.0, -1.0)


def make_spectrum(intensity_values=GAUSSIAN + RESIDUAL_NOISE, channels=CHANNELS, **kwargs):
    return Spectrum(intensity_values=intensity_values, channels=channels, rms_noise=RMS_NOISE, **kwargs)


def make_model(intensity_values=GAUSSIAN + RESIDUAL_NOISE, parameters=None, **kwargs):
    model = Model(spectrum=make_spectrum(intensity_values, **kwargs))
    if parameters is not None:
        model.parameters = parameters
    return model


def test_spectrum_construction():
    spectrum = make_spectrum()
    assert spectrum.n_channels == N_CHANNELS
    np.testing.assert_array_equal(spectrum.noise_values, np.full(N_CHANNELS, RMS_NOISE))


def test_spectrum_noise_values_is_cached():
    spectrum = make_spectrum()
    noise_values = spectrum.noise_values
    # cached_property: later changes to rms_noise do not affect the cached noise values
    spectrum.rms_noise = 99.0
    np.testing.assert_array_equal(spectrum.noise_values, noise_values)
    # a freshly created spectrum reflects the new rms_noise value
    np.testing.assert_array_equal(make_spectrum().noise_values, np.full(N_CHANNELS, RMS_NOISE))


def test_spectrum_signal_mask():
    spectrum = make_spectrum(intensity_values=np.arange(10.0), channels=np.arange(10), signal_intervals=[[2, 5]])
    np.testing.assert_array_equal(spectrum.signal_mask, [False, False, True, True, True] + [False] * 5)
    # noise spike intervals are removed from the signal mask
    spectrum = Spectrum(
        intensity_values=np.arange(10.0),
        channels=np.arange(10),
        rms_noise=1.0,
        signal_intervals=[[0, 10]],
        noise_spike_intervals=[[7, 9]],
    )
    np.testing.assert_array_equal(spectrum.signal_mask, [True] * 7 + [False] * 2 + [True])


def test_spectrum_signal_mask_is_none_without_signal_intervals():
    # without signal_intervals the signal mask is None, even if noise_spike_intervals are given
    spectrum = Spectrum(
        intensity_values=np.arange(10.0),
        channels=np.arange(10),
        rms_noise=1.0,
        noise_spike_intervals=[[7, 9]],
    )
    assert spectrum.signal_mask is None


def test_spectrum_noise_spike_mask():
    # the noise spike mask covers the full spectrum except for the noise spike intervals
    spectrum = Spectrum(
        intensity_values=np.arange(10.0),
        channels=np.arange(10),
        rms_noise=1.0,
        noise_spike_intervals=[[7, 9]],
    )
    np.testing.assert_array_equal(spectrum.noise_spike_mask, [True] * 7 + [False] * 2 + [True])
    # without noise_spike_intervals the mask is None
    assert make_spectrum(intensity_values=np.arange(10.0), channels=np.arange(10)).noise_spike_mask is None


def test_model_defaults():
    model = Model(spectrum=make_spectrum())
    assert model.parameters is None
    assert model.n_components == 0
    assert model.amps is None
    assert model.fwhms is None
    assert model.means is None
    assert model.parameter_uncertainties == []
    assert model.modelled_intensity_values is None
    assert model.residual is None
    assert model.rchi2 is None
    assert model.aicc is None
    assert model.pvalue is None
    assert model.quality_control == []
    assert model.log_of_successful_refits == []
    assert model.new_best_fit is False


def test_model_parameters_setter_splits_parameters():
    model = make_model(parameters=[1, 2, 3, 4, 5, 6])
    assert model.n_components == 2
    assert model.amps == [1, 2]
    assert model.fwhms == [3, 4]
    assert model.means == [5, 6]


def test_model_parameters_setter_invalid_length_raises():
    model = make_model()
    with pytest.raises(Exception, match="One or more fit parameters are missing"):
        model.parameters = [1, 2, 3, 4]


def test_model_parameter_uncertainties():
    model = make_model()
    assert model.parameter_uncertainties == []
    model.parameter_uncertainties = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    assert model.amps_uncertainties == [0.1, 0.2]
    assert model.fwhms_uncertainties == [0.3, 0.4]
    assert model.means_uncertainties == [0.5, 0.6]
    with pytest.raises(Exception, match="One or more fit uncertainty parameters are missing"):
        model.parameter_uncertainties = [0.1, 0.2, 0.3, 0.4]


def test_model_computed_fit_properties():
    model = make_model(parameters=[1, 12, 50])
    # the modelled intensity values correspond to the Gaussian component (amp=1, fwhm=12, mean=50)
    np.testing.assert_allclose(model.modelled_intensity_values, GAUSSIAN)
    # the residual corresponds to the alternating +/-0.05 fake noise pattern
    np.testing.assert_allclose(model.residual, RESIDUAL_NOISE)
    # chi2 = sum((+/-0.05)^2 / 1^2) over 100 channels = 0.25; rchi2 = 0.25 / (100 - 3)
    assert np.isclose(model.rchi2, 0.25 / 97)
    # ssr = 0.25 -> log-likelihood = -50 * ln(0.0025) = 299.5732...
    # aicc = 2 * (3 - log-likelihood) + 2 * 3 * 4 / (100 - 3 - 1)
    assert np.isclose(model.aicc, -592.8964547107981)
    # characterization test (deterministic function; any change here indicates a bug, not dependency drift):
    # pvalue from the normality checks of the residual, recorded from the
    # current implementation on the reference stack
    assert np.isclose(model.pvalue, 8.893431635866619e-175, rtol=1e-3, atol=0.0)


def test_model_blended_components():
    # components 0 (amp=1, fwhm=10, mean=50) and 1 (amp=5, fwhm=10, mean=55) are blended:
    # their separation (5) is smaller than 0.8493... * min(fwhm) (~8.49); component 2 is well separated
    model = make_model(parameters=[1, 5, 2, 10, 10, 30, 50, 55, 80])
    np.testing.assert_array_equal(model.indices_of_blended_components(), [0, 1])
    assert model.number_of_blended_components() == 1


def test_model_blended_components_sorted_by_amplitude():
    # indices of blended components are sorted by ascending amplitude value
    model = make_model(parameters=[5, 1, 10, 10, 50, 55])
    np.testing.assert_array_equal(model.indices_of_blended_components(), [1, 0])


def test_model_no_blended_components():
    model = make_model(parameters=[1, 5, 10, 10, 50, 80])
    assert model.indices_of_blended_components().size == 0
    assert model.number_of_blended_components() == 0


def test_model_number_of_negative_residual_peaks():
    # the fitted component (amp=5) overshoots the spectrum (amp=1) by 4 at channel 50,
    # which is counted as a negative residual peak for the default neg_res_snr=3.0
    model = make_model(intensity_values=GAUSSIAN, parameters=[5, 12, 50])
    assert model.number_of_negative_residual_peaks() == 1
    # an overshoot of 2.5 is below the default threshold of 3.0 * rms_noise
    model = make_model(intensity_values=GAUSSIAN, parameters=[3.5, 12, 50])
    assert model.number_of_negative_residual_peaks() == 0
    # ... but it is picked up for a smaller required signal-to-noise ratio
    assert model.number_of_negative_residual_peaks(neg_res_snr=2.0) == 1


def test_model_log_in_case_of_successful_refit():
    model = make_model()
    # without new_best_fit set, nothing is logged
    model.log_in_case_of_successful_refit(mode="blended")
    assert model.log_of_successful_refits == []
    model.new_best_fit = True
    model.log_in_case_of_successful_refit(mode="positive_residual_peak")
    model.log_in_case_of_successful_refit(mode="negative_residual_peak")
    model.log_in_case_of_successful_refit(mode="broad")
    model.log_in_case_of_successful_refit(mode="blended")
    assert model.log_of_successful_refits == [1, 2, 3, 4]
    # the log can be replaced via the setter
    model.log_of_successful_refits = [4]
    assert model.log_of_successful_refits == [4]


def test_model_quality_control():
    model = make_model()
    assert model.quality_control == []
    model.quality_control = [True, False]
    assert model.quality_control == [True, False]
