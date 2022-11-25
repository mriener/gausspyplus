"""pytest tests for module spatial_fitting.py (Phase 1 methods)"""
import os
import pickle
from pathlib import Path

import pytest
from astropy.io import fits

import numpy as np

ROOT = Path(os.path.realpath(__file__)).parents[1]
filepath = Path(ROOT / "data" / "grs-test_field_10x10.fits")
DATA = fits.getdata(filepath)


def test_spatial_fitting_phase_1():
    from ..spatial_fitting.spatial_fitting import SpatialFitting

    sp = SpatialFitting()
    sp.length = 16
    sp.shape = (4, 4)
    sp.fwhm_factor = 2.0
    sp.fwhm_separation = 4.0
    sp.broad_neighbor_fraction = 0.5
    sp.nanMask = np.zeros(16, dtype=bool)
    sp.nanMask[0] = True
    sp.decomposition = {
        "fwhms_fit": [
            None,
            [30.5, 40.5, 50.5],
            [10.0],
            [20.0, 40.0],
            [100.0],
            [30.0, 40.0],
            [100.0],
            [30.0, 40.0],
            [100.0, 49.5],
            [30.0, 100.0],
            [100.0],
            [7.0, 2.0],
            [100.0, 50.5],
            [100.0],
            [100.0],
            [6.0, 2.1],
        ]
    }
    mask_broad = sp._define_mask_broad()
    assert np.all(np.equal(np.flatnonzero(mask_broad), np.array([6, 8, 9, 11])))


if __name__ == "__main__":
    test_spatial_fitting_phase_1()
