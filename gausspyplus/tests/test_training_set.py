import os
from pathlib import Path

import numpy as np
from astropy.io import fits

from gausspyplus.training_set import GaussPyTrainingSet

ROOT = Path(os.path.realpath(__file__)).parents[1]
DATA = fits.getdata(ROOT / 'data' / 'grs-test_field.fits')


def test_get_signal_ranges():
    spectrum = DATA[:, 31, 40]
    training_set = GaussPyTrainingSet()
    rms = 0.10634302494716603
    # %timeit training_set._get_maxima(spectrum, rms)
    # before refactoring: 147 µs ± 12.1 µs per loop (mean ± std. dev. of 7 runs, 10000 loops each)
    # after refactoring: 106 µs ± 315 ns per loop (mean ± std. dev. of 7 runs, 10000 loops each)
    maxima = training_set._get_maxima(spectrum, rms)
    assert np.all(np.equal(maxima, np.array([155, 187, 232])))
