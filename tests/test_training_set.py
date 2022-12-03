import os
from pathlib import Path

import numpy as np
from astropy.io import fits

from gausspyplus.training.training_set import GaussPyTrainingSet

ROOT = Path(os.path.realpath(__file__)).parents[1]
DATA = fits.getdata(ROOT / "gausspyplus" / "data" / "grs-test_field.fits")


def test_get_maxima():
    spectrum = DATA[:, 31, 40]
    training_set = GaussPyTrainingSet()
    rms = 0.10634302494716603
    # %timeit training_set._get_maxima(spectrum, rms)
    # before refactoring: 147 µs ± 12.1 µs per loop (mean ± std. dev. of 7 runs, 10000 loops each)
    # after refactoring: 106 µs ± 315 ns per loop (mean ± std. dev. of 7 runs, 10000 loops each)
    maxima = training_set._get_maxima(spectrum, rms)
    assert np.all(np.equal(maxima, np.array([155, 187, 232])))


def test_gaussian_fitting():
    # %timeit training_set.gaussian_fitting(spectrum, maxima, rms)
    # before refactoring: 260 ms ± 11 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
    spectrum = DATA[:, 31, 40]
    training_set = GaussPyTrainingSet()
    rms = 0.10634302494716603
    training_set.n_channels = spectrum.size
    training_set.maxStddev = (
        training_set.max_fwhm / 2.355 if training_set.max_fwhm is not None else None
    )
    training_set.minStddev = (
        training_set.min_fwhm / 2.355 if training_set.min_fwhm is not None else None
    )
    fit_values = training_set.gaussian_fitting(spectrum, rms)
    assert fit_values == [
        [0.49323248704016304, 182.4598447452132, 16.670207796744403],
        [0.6966674945382034, 232.65252631359184, 5.672850108091679],
    ]
    # assert rchi2 == 1.1918769968287917
    # assert pvalue == 0.2815043462467674
