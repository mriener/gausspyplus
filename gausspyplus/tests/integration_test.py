import os
import pickle
from pathlib import Path

import pytest
from astropy.io import fits

import numpy as np

ROOT = Path(os.path.realpath(__file__)).parents[1]
filepath = Path(ROOT / "data" / "grs-test_field_10x10.fits")
DATA = fits.getdata(filepath)

# TODO: Delete created pickle files after running the tests; otherwise intermediate-stage pickle files can still be
#  used if an intermediate stage fails and later tests seem to pass even though they should not


# @pytest.mark.skip(reason="Temporarily disabled to make tests run quicker")
def test_prepare_cube():
    from ..prepare import GaussPyPrepare

    prepare = GaussPyPrepare()
    prepare.path_to_file = str(filepath)
    prepare.dirpath_gpy = "test_grs"
    prepare.use_ncpus = 1
    prepare.log_output = False
    prepare.verbose = False
    prepare.prepare_cube()
    with open(
        ROOT / "tests" / f"test_grs/gpy_prepared/{filepath.stem}.pickle", "rb"
    ) as pfile:
        data_prepared = pickle.load(pfile)
    # print(
    #     "0",
    #     prepare.average_rms,
    #     sum(rms[0] for rms in data_prepared["error"] if rms[0] is not None)
    # )
    assert np.allclose(prepare.average_rms, 0.10315973929242594)
    assert np.allclose(
        sum(rms[0] for rms in data_prepared["error"] if rms[0] is not None),
        10.212814189950166,
    )


# @pytest.mark.skip(reason="Temporarily disabled to make tests run quicker")
def test_decompose_cube_gausspy():
    from ..decompose import GaussPyDecompose

    decompose = GaussPyDecompose()
    decompose.path_to_pickle_file = str(
        ROOT / "tests" / f"test_grs/gpy_prepared/{filepath.stem}.pickle"
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
        ROOT / "tests" / f"test_grs/gpy_decomposed/{filepath.stem}_g_fit_fin.pickle",
        "rb",
    ) as pfile:
        data_decomposed = pickle.load(pfile)
    # print(
    #     "1",
    #     sum(ncomps for ncomps in data_decomposed["N_components"] if ncomps is not None),
    #     sum(sum(lst) for lst in data_decomposed["fwhms_fit"] if lst is not None),
    #     sum(sum(lst) for lst in data_decomposed["fwhms_fit_err"] if lst is not None)
    # )
    assert (
        sum(ncomps for ncomps in data_decomposed["N_components"] if ncomps is not None)
        == 181
    )
    assert np.allclose(
        sum(sum(lst) for lst in data_decomposed["fwhms_fit"] if lst is not None),
        2821.8647475612247,
    )
    assert np.allclose(
        sum(sum(lst) for lst in data_decomposed["fwhms_fit_err"] if lst is not None),
        302.183306677499,
    )

    decompose.improve_fitting = True
    decompose.suffix = "_g+"
    decompose.decompose()
    with open(
        ROOT / "tests" / f"test_grs/gpy_decomposed/{filepath.stem}_g+_fit_fin.pickle",
        "rb",
    ) as pfile:
        data_decomposed_gplus = pickle.load(pfile)
    # print(
    #     "2",
    #     sum(ncomps for ncomps in data_decomposed_gplus["N_components"] if ncomps is not None),
    #     sum(sum(lst) for lst in data_decomposed_gplus["fwhms_fit"] if lst is not None),
    #     sum(sum(lst) for lst in data_decomposed_gplus["fwhms_fit_err"] if lst is not None),
    #     sum(x for x in data_decomposed_gplus["pvalue"] if x is not None),
    #     sum(x for x in data_decomposed_gplus["best_fit_rchi2"] if x is not None),
    #     sum(x for x in data_decomposed_gplus["best_fit_aicc"] if x is not None),
    #     sum(sum(lst) for lst in data_decomposed_gplus["log_gplus"] if lst is not None),
    #     sum(len(x) for x in data_decomposed_gplus["log_gplus"] if x is not None and bool(x)),
    #     sum(sum(lst) for lst in data_decomposed_gplus["quality_control"] if lst is not None),
    # )
    assert (
        sum(
            ncomps
            for ncomps in data_decomposed_gplus["N_components"]
            if ncomps is not None
        )
        == 272
    )
    assert np.allclose(
        sum(sum(lst) for lst in data_decomposed_gplus["fwhms_fit"] if lst is not None),
        4210.279681986788,
    )
    assert np.allclose(
        sum(
            sum(lst)
            for lst in data_decomposed_gplus["fwhms_fit_err"]
            if lst is not None
        ),
        572.5082893844038,
    )
    assert np.allclose(
        sum(x for x in data_decomposed_gplus["pvalue"] if x is not None),
        14.174536904795435,
    )
    assert np.allclose(
        sum(x for x in data_decomposed_gplus["best_fit_rchi2"] if x is not None),
        117.25885765551081,
    )
    assert np.allclose(
        sum(x for x in data_decomposed_gplus["best_fit_aicc"] if x is not None),
        -46522.959923033035,
    )
    assert (
        sum(sum(lst) for lst in data_decomposed_gplus["log_gplus"] if lst is not None)
        == 107
    )
    assert (
        sum(
            len(x)
            for x in data_decomposed_gplus["log_gplus"]
            if x is not None and bool(x)
        )
        == 70
    )
    # assert sum(sum(lst) for lst in data_decomposed_gplus["quality_control"] if lst is not None) == 159

    # TODO: test a new decomposition round with n_max_comps


# @pytest.mark.skip(reason="Temporarily disabled to make tests run quicker")
def test_spatial_fitting_phase_1():
    from ..spatial_fitting import SpatialFitting

    sp = SpatialFitting()
    sp.path_to_pickle_file = str(
        ROOT / "tests" / f"test_grs/gpy_prepared/{filepath.stem}.pickle"
    )
    #  Filepath to the pickled dictionary of the decomposition results
    sp.path_to_decomp_file = str(
        ROOT / "tests" / f"test_grs/gpy_decomposed/{filepath.stem}_g+_fit_fin.pickle"
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
        / f"test_grs/gpy_decomposed/{filepath.stem}_g+_fit_fin_sf-p1.pickle",
        "rb",
    ) as pfile:
        data_spatial_fitted_phase_1 = pickle.load(pfile)

    # TODO: The spatial refitting seems to refit the spectra, because the values are slightly changed; this makes no
    #  difference to the results but might prolong the whole spatial refitting unnecessarily, because refit_iteration
    #  is still increased -> it's better in such cases to compare whether the number of components or fit values have
    #  changed substantially with np.allclose -> if not, the values from the previous iteration should be kept
    # TODO: check whether refit_iteration tracks the number of how often a spectrum has been refit
    # print(
    #     "3",
    #     sp.refitting_iteration,
    #     sum(ncomps for ncomps in data_spatial_fitted_phase_1["N_components"] if ncomps is not None),
    #     sum(sum(lst) for lst in data_spatial_fitted_phase_1["fwhms_fit"] if lst is not None),
    #     sum(sum(lst) for lst in data_spatial_fitted_phase_1["fwhms_fit_err"] if lst is not None),
    #     sum(data_spatial_fitted_phase_1["refit_iteration"])
    # )
    assert sp.refitting_iteration == 2
    assert (
        sum(
            ncomps
            for ncomps in data_spatial_fitted_phase_1["N_components"]
            if ncomps is not None
        )
        == 274
    )
    assert np.allclose(
        sum(
            sum(lst)
            for lst in data_spatial_fitted_phase_1["fwhms_fit"]
            if lst is not None
        ),
        4120.289339690797,
    )
    assert np.allclose(
        sum(
            sum(lst)
            for lst in data_spatial_fitted_phase_1["fwhms_fit_err"]
            if lst is not None
        ),
        603.782311784271,
    )
    assert sum(data_spatial_fitted_phase_1["refit_iteration"]) == 42


# @pytest.mark.skip(reason="Temporarily disabled to make tests run quicker")
def test_spatial_fitting_phase_2():
    from ..spatial_fitting import SpatialFitting

    sp = SpatialFitting()
    sp.path_to_pickle_file = str(
        ROOT / "tests" / f"test_grs/gpy_prepared/{filepath.stem}.pickle"
    )
    #  Filepath to the pickled dictionary of the decomposition results
    sp.path_to_decomp_file = str(
        ROOT
        / "tests"
        / f"test_grs/gpy_decomposed/{filepath.stem}_g+_fit_fin_sf-p1.pickle"
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
        / f"test_grs/gpy_decomposed/{filepath.stem}_g+_fit_fin_sf-p2.pickle",
        "rb",
    ) as pfile:
        data_spatial_fitted_phase_2 = pickle.load(pfile)

    # print(
    #     "4",
    #     sp.refitting_iteration,
    #     sum(ncomps for ncomps in data_spatial_fitted_phase_2["N_components"] if ncomps is not None),
    #     sum(sum(lst) for lst in data_spatial_fitted_phase_2["fwhms_fit"] if lst is not None),
    #     sum(sum(lst) for lst in data_spatial_fitted_phase_2["fwhms_fit_err"] if lst is not None),
    #     sum(data_spatial_fitted_phase_2["refit_iteration"])
    # )
    assert sp.refitting_iteration == 8
    assert (
        sum(
            ncomps
            for ncomps in data_spatial_fitted_phase_2["N_components"]
            if ncomps is not None
        )
        == 273
    )
    assert np.allclose(
        sum(
            sum(lst)
            for lst in data_spatial_fitted_phase_2["fwhms_fit"]
            if lst is not None
        ),
        4121.091273601258,
    )
    assert np.allclose(
        sum(
            sum(lst)
            for lst in data_spatial_fitted_phase_2["fwhms_fit_err"]
            if lst is not None
        ),
        592.8518512079702,
    )
    assert sum(data_spatial_fitted_phase_2["refit_iteration"]) == 2


if __name__ == "__main__":
    test_prepare_cube()
    test_decompose_cube_gausspy()
    test_spatial_fitting_phase_1()
    test_spatial_fitting_phase_2()
