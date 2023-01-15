"""pytest tests for module noise_estimation.py"""
import os
from pathlib import Path

from astropy.io import fits

ROOT = Path(os.path.realpath(__file__)).parents[1]
DATA = fits.getdata(ROOT / "data" / "grs-test_field.fits")


def test_get_signal_ranges():
    from gausspyplus.preparation.determine_intervals import get_signal_ranges

    spectrum = DATA[:, 31, 40]
    signal_ranges = get_signal_ranges(spectrum=spectrum, rms=0.1)
    assert signal_ranges == [[142, 207], [212, 252]]


if __name__ == "__main__":
    test_get_signal_ranges()
