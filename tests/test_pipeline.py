"""End-to-end pipeline regression tests (prepare → decompose → spatial refitting).

Runs the full GaussPy+ chain on the 5x5 and 10x10 test fields within a single
test function per field, so the stages' intermediate on-disk results cannot go
stale between tests.

Golden values are regression baselines recorded on the reference stack
(Python 3.14.9, numpy 2.5.4, scipy 1.18.1, astropy 8.0.2, lmfit 1.3.4,
matplotlib 3.11.2, networkx 3.6, tqdm 4.70.1):

* Float aggregates are compared with ``rtol=1e-3`` — fitting results
  legitimately drift in the low digits between dependency versions.
* Sums of fitted FWHM errors in the spatial-fitting stages are compared
  with ``rtol=0.05``: for marginally degenerate refits, whether lmfit's
  covariance estimation succeeds (large finite errors) or fails (0 via the
  None→0 stderr sentinel in ``gaussian_functions.errs_vec_from_lmfit``)
  depends on platform-level floating-point differences, so the summed
  errors legitimately shift by one spectrum's contribution between CPU
  architectures.
* Integer counts are compared with a small tolerance — marginal fits can
  flip decision thresholds between dependency versions.

When the dependency stack is updated, re-run this test: if values drift
beyond the tolerances, verify the shift is legitimate version drift (not a
bug) and re-record the baselines.

The 10x10 field is marked ``slow``; exclude it with ``pytest -m "not slow"``.
"""

from pathlib import Path

import numpy as np
import pytest

from tests.helpers import assert_close, assert_count, load_pickle, sum_flat, sum_nested

FIELD_FILES = {
    "5x5": "grs-test_field_5x5.fits",
    "10x10": "grs-test_field_10x10.fits",
}

#  Baselines recorded on the reference stack (see module docstring).
EXPECTED = {
    "5x5": {
        "prepare": [0.10368932, 2.5922327],
        # [sum N_components, sum fwhms_fit, sum fwhms_fit_err]
        "decompose_g": [40, 698.6367721321028, 86.91550132053058],
        # [sum N_components, sum fwhms_fit, sum fwhms_fit_err, sum pvalue,
        #  sum best_fit_rchi2, sum best_fit_aicc, sum log_gplus,
        #  number of non-empty log_gplus entries]
        "decompose_gplus": [
            63,
            1078.9933965125817,
            147.28349208980956,
            4.334427936336032,
            30.873467291678917,
            -12029.640413173274,
            24,
            20,
        ],
        # [refitting_iteration, sum N_components, sum fwhms_fit,
        #  sum fwhms_fit_err, sum refit_iteration]
        "sf1": [2, 63, 1064.3349631531858, 145.23931605029236, 7],
        "sf2": [6, 63, 1064.3349631531858, 145.23931605029236, 0],
    },
    "10x10": {
        "prepare": [0.10315975, 10.212815],
        "decompose_g": [181, 2822.005436183146, 302.18645260103784],
        "decompose_gplus": [
            272,
            4243.113147680418,
            577.3745564059784,
            14.17298973825546,
            116.98788282255391,
            -46549.87738520191,
            107,
            70,
        ],
        "sf1": [2, 267, 4133.950305204568, 548.7186530786249, 26],
        "sf2": [8, 267, 4127.480046893161, 550.4096106611357, 2],
    },
}


@pytest.mark.parametrize("field", ["5x5", pytest.param("10x10", marks=pytest.mark.slow)])
def test_pipeline(field, data_path, results_dir):
    from gausspyplus.decomposition.decompose import GaussPyDecompose
    from gausspyplus.preparation.prepare import GaussPyPrepare
    from gausspyplus.spatial_fitting.spatial_fitting import SpatialFitting

    fname = FIELD_FILES[field]
    stem = Path(fname).stem
    n_spectra = 25 if field == "5x5" else 100
    expected = EXPECTED[field]
    workdir = results_dir / field

    # Stage 1: prepare
    prepare = GaussPyPrepare()
    prepare.path_to_file = str(data_path / fname)
    prepare.dirpath_gpy = str(workdir)
    prepare.use_ncpus = 1
    prepare.log_output = False
    prepare.verbose = False
    prepare.prepare_cube()
    prepared = load_pickle(workdir / "gpy_prepared" / f"{stem}.pickle")

    assert len(prepared["error"]) == n_spectra
    rms_values = [rms[0] for rms in prepared["error"] if rms is not None and rms[0] is not None]
    assert all(rms > 0 for rms in rms_values)
    assert_close([prepare.average_rms, sum(rms_values)], expected["prepare"], msg=f"{field}: prepare")

    # Stage 2: decompose with plain GaussPy
    decompose = GaussPyDecompose()
    decompose.path_to_pickle_file = str(workdir / "gpy_prepared" / f"{stem}.pickle")
    decompose.alpha1 = 2.58
    decompose.alpha2 = 5.14
    decompose.use_ncpus = 1
    decompose.log_output = False
    decompose.verbose = False

    decompose.suffix = "_g"
    decompose.improve_fitting = False
    decompose.decompose()
    decomposed_g = load_pickle(workdir / "gpy_decomposed" / f"{stem}_g_fit_fin.pickle")

    exp = expected["decompose_g"]
    assert_count(sum_flat(decomposed_g["N_components"]), exp[0], tol=2, msg=f"{field}: decompose_g components")
    assert_close(
        [sum_nested(decomposed_g["fwhms_fit"]), sum_nested(decomposed_g["fwhms_fit_err"])],
        exp[1:],
        msg=f"{field}: decompose_g",
    )
    assert len(decomposed_g["N_components"]) == n_spectra
    assert all(n is None or n >= 0 for n in decomposed_g["N_components"])
    assert all(np.all(np.asarray(fwhms) > 0) for fwhms in decomposed_g["fwhms_fit"] if fwhms is not None)

    # Stage 3: decompose with GaussPy+ (improved fitting)
    decompose.suffix = "_g+"
    decompose.improve_fitting = True
    decompose.decompose()
    decomposed_gplus = load_pickle(workdir / "gpy_decomposed" / f"{stem}_g+_fit_fin.pickle")

    exp = expected["decompose_gplus"]
    assert_count(sum_flat(decomposed_gplus["N_components"]), exp[0], tol=2, msg=f"{field}: decompose_gplus components")
    assert_close(
        [
            sum_nested(decomposed_gplus["fwhms_fit"]),
            sum_nested(decomposed_gplus["fwhms_fit_err"]),
            sum_flat(decomposed_gplus["pvalue"]),
            sum_flat(decomposed_gplus["best_fit_rchi2"]),
            sum_flat(decomposed_gplus["best_fit_aicc"]),
        ],
        exp[1:6],
        msg=f"{field}: decompose_gplus",
    )
    assert_count(sum_nested(decomposed_gplus["log_gplus"]), exp[6], tol=2, msg=f"{field}: decompose_gplus log entries")
    # number of non-empty per-spectrum refit log entries
    assert_count(
        sum(len(entry) for entry in decomposed_gplus["log_gplus"] if entry is not None and bool(entry)),
        exp[7],
        tol=2,
        msg=f"{field}: decompose_gplus non-empty log entries",
    )
    # structural invariant: one log entry per spectrum (exact)
    assert len(decomposed_gplus["log_gplus"]) == n_spectra
    # GaussPy+ improvement should not lose components relative to plain GaussPy
    assert sum_flat(decomposed_gplus["N_components"]) >= sum_flat(decomposed_g["N_components"])
    assert all(p is None or 0 <= p <= 1 for p in decomposed_gplus["pvalue"])
    assert all(np.all(np.asarray(fwhms) > 0) for fwhms in decomposed_gplus["fwhms_fit"] if fwhms is not None)

    # Stage 4: spatial fitting phase 1 (flag-driven refits)
    sp = SpatialFitting()
    sp.path_to_pickle_file = str(workdir / "gpy_prepared" / f"{stem}.pickle")
    sp.path_to_decomp_file = str(workdir / "gpy_decomposed" / f"{stem}_g+_fit_fin.pickle")
    sp.refit_blended = True
    sp.refit_neg_res_peak = True
    sp.refit_broad = True
    sp.flag_residual = True
    sp.refit_residual = True
    sp.refit_ncomps = True
    sp.use_ncpus = 1
    sp.log_output = False
    sp.verbose = False
    sp.spatial_fitting()
    fitted_p1 = load_pickle(workdir / "gpy_decomposed" / f"{stem}_g+_fit_fin_sf-p1.pickle")

    exp = expected["sf1"]
    assert_count(sp.refitting_iteration, exp[0], tol=2, msg=f"{field}: sf1 refitting_iteration")
    assert_count(sum_flat(fitted_p1["N_components"]), exp[1], tol=2, msg=f"{field}: sf1 components")
    assert_close(
        [sum_nested(fitted_p1["fwhms_fit"])],
        exp[2:3],
        msg=f"{field}: sf1",
    )
    #  FWHM error sums are discontinuous for marginally degenerate refits: the
    #  None→0 stderr sentinel (gaussian_functions.errs_vec_from_lmfit) zeroes
    #  a spectrum's errors when lmfit's covariance estimation fails, and that
    #  failure depends on platform-level floating-point differences. See the
    #  module docstring; the fit sums above stay tight across architectures.
    assert_close(
        [sum_nested(fitted_p1["fwhms_fit_err"])],
        exp[3:4],
        rtol=0.05,
        msg=f"{field}: sf1",
    )
    assert_count(sum(fitted_p1["refit_iteration"]), exp[4], tol=2, msg=f"{field}: sf1 refit_iteration")
    assert all(np.all(np.asarray(fwhms) > 0) for fwhms in fitted_p1["fwhms_fit"] if fwhms is not None)

    # Stage 5: spatial fitting phase 2 (continuity)
    sp = SpatialFitting()
    sp.path_to_pickle_file = str(workdir / "gpy_prepared" / f"{stem}.pickle")
    sp.path_to_decomp_file = str(workdir / "gpy_decomposed" / f"{stem}_g+_fit_fin_sf-p1.pickle")
    sp.refit_blended = False
    sp.refit_neg_res_peak = False
    sp.refit_broad = False
    sp.refit_residual = False
    sp.refit_ncomps = True
    sp.use_ncpus = 1
    sp.log_output = False
    sp.verbose = False
    sp.spatial_fitting(continuity=True)
    fitted_p2 = load_pickle(workdir / "gpy_decomposed" / f"{stem}_g+_fit_fin_sf-p2.pickle")

    exp = expected["sf2"]
    assert_count(sp.refitting_iteration, exp[0], tol=2, msg=f"{field}: sf2 refitting_iteration")
    assert_count(sum_flat(fitted_p2["N_components"]), exp[1], tol=2, msg=f"{field}: sf2 components")
    assert_close(
        [sum_nested(fitted_p2["fwhms_fit"])],
        exp[2:3],
        msg=f"{field}: sf2",
    )
    #  Same discontinuity as in phase 1: phase 2 inherits the phase 1 errors,
    #  including any zeroed sentinel values (see module docstring).
    assert_close(
        [sum_nested(fitted_p2["fwhms_fit_err"])],
        exp[3:4],
        rtol=0.05,
        msg=f"{field}: sf2",
    )
    assert_count(sum(fitted_p2["refit_iteration"]), exp[4], tol=2, msg=f"{field}: sf2 refit_iteration")
    assert all(np.all(np.asarray(fwhms) > 0) for fwhms in fitted_p2["fwhms_fit"] if fwhms is not None)
