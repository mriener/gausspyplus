"""Shared fixtures for the GaussPy+ test suite."""
import os
from pathlib import Path

import pytest
from astropy.io import fits

REPO_ROOT = Path(os.path.realpath(__file__)).parents[1]


@pytest.fixture(scope="session")
def repo_root():
    """Path to the repository root."""
    return REPO_ROOT


@pytest.fixture(scope="session")
def data_path(repo_root):
    """Path to the directory containing the test data files."""
    return repo_root / "data"


@pytest.fixture(scope="session")
def grs_test_field(data_path):
    """GRS test field (26x25 spectra) used by unit tests."""
    return fits.getdata(data_path / "grs-test_field.fits")


@pytest.fixture(scope="session")
def grs_test_field_5x5(data_path):
    """Small 5x5 cutout of the GRS test field (25 spectra)."""
    return fits.getdata(data_path / "grs-test_field_5x5.fits")


@pytest.fixture(scope="session")
def grs_test_field_10x10(data_path):
    """10x10 cutout of the GRS test field (100 spectra)."""
    return fits.getdata(data_path / "grs-test_field_10x10.fits")


@pytest.fixture(scope="session")
def results_dir(tmp_path_factory):
    """Isolated working directory for pipeline tests.

    Replaces the former behaviour of writing intermediate pickles into
    ``tests/test_grs`` inside the repository.
    """
    return tmp_path_factory.mktemp("gpy_results")
