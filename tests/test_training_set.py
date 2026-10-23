"""Unit tests for methods of the GaussPyTrainingSet class."""
import numpy as np

from gausspyplus.training.training_set import GaussPyTrainingSet

RMS = 0.10634302494716603  # noise level of the test spectrum at position (31, 40); recorded as the asserted output of determine_noise in test_noise_estimation.py


def test_get_maxima(grs_test_field):
    # characterization test (deterministic function; any change here indicates a bug, not dependency drift)
    spectrum = grs_test_field[:, 31, 40]
    training_set = GaussPyTrainingSet()
    maxima = training_set._get_maxima(spectrum, RMS)
    assert np.array_equal(maxima, np.array([155, 187, 232]))


def test_gaussian_fitting(grs_test_field):
    spectrum = grs_test_field[:, 31, 40]
    training_set = GaussPyTrainingSet()
    training_set.n_channels = spectrum.size
    training_set.maxStddev = training_set.max_fwhm / 2.355 if training_set.max_fwhm is not None else None
    training_set.minStddev = training_set.min_fwhm / 2.355 if training_set.min_fwhm is not None else None
    fit_values = training_set.gaussian_fitting(spectrum, RMS)
    #  characterization test: fit values recorded from the current
    #  implementation on the reference dependency stack; compared with a
    #  tolerance since fitting results drift in the low digits between
    #  dependency versions (see tests/test_pipeline.py for details)
    assert np.allclose(
        fit_values,
        [
            [0.49323248704016304, 182.4598447452132, 16.670207796744403],
            [0.6966674945382034, 232.65252631359184, 5.672850108091679],
        ],
        rtol=1e-3,
    )
