"""pytest tests for module noise_estimation.py"""
import os
from pathlib import Path

from astropy.io import fits

import numpy as np

ROOT = Path(os.path.realpath(__file__)).parents[1]
DATA = fits.getdata(ROOT / 'data' / 'grs-test_field.fits')


def test_determine_maximum_consecutive_channels():
    from ..utils.noise_estimation import determine_maximum_consecutive_channels
    assert determine_maximum_consecutive_channels(100, 0.05) == 11
    assert determine_maximum_consecutive_channels(200, 0.02) == 14
    assert determine_maximum_consecutive_channels(1000, 0.01) == 17


def test_get_rms_noise():
    from ..utils.noise_estimation import get_rms_noise
    spectrum = DATA[:, 31, 40]
    rms_noise = get_rms_noise(spectrum)
    assert np.allclose(rms_noise, 0.10634302494716603)
