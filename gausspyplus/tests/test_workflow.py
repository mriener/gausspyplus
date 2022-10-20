"""pytest tests for module prepare.py"""
import os
import pickle
from pathlib import Path

from astropy.io import fits

import numpy as np

ROOT = Path(os.path.realpath(__file__)).parents[1]
DATA = fits.getdata(ROOT / "data" / "grs-test_field_5x5.fits")


def is_not_none(x):
    return x is not None


def test_prepare_cube():
    from ..prepare import GaussPyPrepare

    prepare = GaussPyPrepare()
    prepare.path_to_file = ROOT / "data" / "grs-test_field_5x5.fits"
    prepare.dirpath_gpy = "test_grs"
    prepare.use_ncpus = 1
    prepare.log_output = False
    prepare.verbose = False
    prepare.prepare_cube()
    with open(
        ROOT / "tests" / "test_grs/gpy_prepared/grs-test_field_5x5.pickle", "rb"
    ) as pfile:
        data_prepared = pickle.load(pfile)
    expected_values = [
        0.10368931207074261,
        2.5922328017685654,
    ]
    actual_values = [
        prepare.average_rms,
        np.array(data_prepared["error"]).sum(),
    ]
    assert np.allclose(expected_values, actual_values)


def test_decompose_cube_gausspy():
    from ..decompose import GaussPyDecompose

    decompose = GaussPyDecompose()
    decompose.path_to_pickle_file = (
        ROOT / "tests" / "test_grs/gpy_prepared/grs-test_field_5x5.pickle"
    )
    decompose.alpha1 = 2.58
    decompose.alpha2 = 5.14
    decompose.suffix = "_g"
    decompose.use_ncpus = 1
    decompose.log_output = False
    decompose.verbose = False
    decompose.improve_fitting = False
    decompose.decompose()
    with open(
        ROOT / "tests" / "test_grs/gpy_decomposed/grs-test_field_5x5_g_fit_fin.pickle",
        "rb",
    ) as pfile:
        data_decomposed = pickle.load(pfile)

    expected_values = [
        40,
        698.5091534299819,
        86.91178374709153,
    ]
    actual_values = [
        np.array(data_decomposed["N_components"]).sum(),
        sum(map(sum, filter(is_not_none, data_decomposed["fwhms_fit"]))),
        sum(map(sum, filter(is_not_none, data_decomposed["fwhms_fit_err"]))),
    ]
    assert np.allclose(expected_values, actual_values)

    decompose.improve_fitting = True
    decompose.suffix = "_g+"
    decompose.decompose()
    with open(
        ROOT / "tests" / "test_grs/gpy_decomposed/grs-test_field_5x5_g+_fit_fin.pickle",
        "rb",
    ) as pfile:
        data_decomposed_gplus = pickle.load(pfile)
    expected_values = [
        63,
        1078.5970429518438,
        147.2004731702768,
        4.3322788117455175,
        30.872647664381116,
        -12029.69958608633,
        24,
        25,
    ]
    actual_values = [
        np.array(data_decomposed_gplus["N_components"]).sum(),
        sum(map(sum, filter(is_not_none, data_decomposed_gplus["fwhms_fit"]))),
        sum(map(sum, filter(is_not_none, data_decomposed_gplus["fwhms_fit_err"]))),
        sum(data_decomposed_gplus["pvalue"]),
        sum(data_decomposed_gplus["best_fit_rchi2"]),
        sum(data_decomposed_gplus["best_fit_aicc"]),
        sum(map(sum, filter(is_not_none, data_decomposed_gplus["log_gplus"]))),
        len(data_decomposed_gplus["log_gplus"]),
    ]
    assert np.allclose(expected_values, actual_values)

    # TODO: test a new decomposition round with n_max_comps


def test_spatial_fitting_phase_1():
    from ..spatial_fitting import SpatialFitting

    sp = SpatialFitting()
    sp.path_to_pickle_file = (
        ROOT / "tests" / "test_grs/gpy_prepared/grs-test_field_5x5.pickle"
    )
    #  Filepath to the pickled dictionary of the decomposition results
    sp.path_to_decomp_file = (
        ROOT / "tests" / "test_grs/gpy_decomposed/grs-test_field_5x5_g+_fit_fin.pickle"
    )
    sp.refit_blended = True
    sp.refit_neg_res_peak = True
    sp.refit_broad = True
    sp.flag_residual = True
    sp.refit_residual = True
    sp.refit_ncomps = True
    sp.use_ncpus = 1
    sp.log_output = True
    sp.verbose = True
    sp.spatial_fitting()

    with open(
        ROOT
        / "tests"
        / "test_grs/gpy_decomposed/grs-test_field_5x5_g+_fit_fin_sf-p1.pickle",
        "rb",
    ) as pfile:
        data_spatial_fitted_phase_1 = pickle.load(pfile)

    # TODO: The spatial refitting seems to refit the spectra, because the values are slightly changed; this makes no
    #  difference to the results but might prolong the whole spatial refitting unnecessarily, because refit_iteration
    #  is still increased -> it's better in such cases to compare whether the number of components or fit values have
    #  changed substantially with np.allclose -> if not, the values from the previous iteration should be kept
    # TODO: check whether refit_iteration tracks the number of how often a spectrum has been refit
    expected_values = [
        1078.597046432567,
        147.200472355394,
        0,
    ]
    actual_values = [
        sum(map(sum, filter(is_not_none, data_spatial_fitted_phase_1["fwhms_fit"]))),
        sum(
            map(sum, filter(is_not_none, data_spatial_fitted_phase_1["fwhms_fit_err"]))
        ),
        sum(data_spatial_fitted_phase_1["refit_iteration"]),
    ]
    assert np.allclose(expected_values, actual_values)


def test_spatial_fitting_phase_2():
    from ..spatial_fitting import SpatialFitting

    sp = SpatialFitting()
    sp.path_to_pickle_file = (
        ROOT / "tests" / "test_grs/gpy_prepared/grs-test_field_5x5.pickle"
    )
    #  Filepath to the pickled dictionary of the decomposition results
    sp.path_to_decomp_file = (
        ROOT
        / "tests"
        / "test_grs/gpy_decomposed/grs-test_field_5x5_g+_fit_fin_sf-p1.pickle"
    )
    sp.refit_blended = False
    sp.refit_neg_res_peak = False
    sp.refit_broad = False
    sp.refit_residual = False
    sp.refit_ncomps = True
    sp.use_ncpus = 1
    sp.log_output = True
    sp.verbose = True
    sp.spatial_fitting(continuity=True)

    with open(
        ROOT
        / "tests"
        / "test_grs/gpy_decomposed/grs-test_field_5x5_g+_fit_fin_sf-p2.pickle",
        "rb",
    ) as pfile:
        data_spatial_fitted_phase_2 = pickle.load(pfile)

    expected_values = [
        1078.597046432567,
        147.200472355394,
        0,  # TODO: Check if this value is correct
    ]
    actual_values = [
        sum(map(sum, filter(is_not_none, data_spatial_fitted_phase_2["fwhms_fit"]))),
        sum(
            map(sum, filter(is_not_none, data_spatial_fitted_phase_2["fwhms_fit_err"]))
        ),
        sum(data_spatial_fitted_phase_2["refit_iteration"]),
    ]
    assert np.allclose(expected_values, actual_values)


if __name__ == "__main__":
    # test_prepare_cube()
    test_decompose_cube_gausspy()
    # test_spatial_fitting_phase_1()
    # test_spatial_fitting_phase_2()
