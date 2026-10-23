"""Shared helper functions for the GaussPy+ test suite."""
import pickle

import numpy as np

# Conventions used in this test suite:
# - "characterization test" comments mark assertions whose expected values were
#   recorded from the current implementation on the reference dependency stack
#   (real-data results have no independently computable ground truth). They
#   detect regressions; they do not verify correctness.
#   The comment includes the drift-risk class:
#     * deterministic functions -> any change indicates a bug, not dependency drift
#     * fitting-dependent values -> may legitimately drift between dependency versions
# - All other golden values are either analytically derived or follow from
#   synthetic inputs whose expected output is determined by the code contract.


def is_not_none(x):
    return x is not None


def sum_nested(values):
    """Sum over per-spectrum lists/arrays, skipping None entries."""
    return sum(map(sum, filter(is_not_none, values)))


def sum_flat(values):
    """Sum a flat list, skipping None entries."""
    return sum(filter(is_not_none, values))


def load_pickle(path):
    """Load a pickled file (with the file handle properly closed)."""
    with open(path, "rb") as pfile:
        return pickle.load(pfile)


def assert_close(actual, expected, rtol=1e-3, msg=""):
    """Compare float regression values with a tolerance for dependency-version drift.

    Golden values in these tests are regression baselines recorded on a
    reference dependency stack; fitting results legitimately shift in the
    last digits between dependency versions, so exact comparisons are
    intentionally avoided.
    """
    assert np.allclose(actual, expected, rtol=rtol), f"{msg}: {actual} differs from expected {expected} (rtol={rtol})"


def assert_count(actual, expected, tol, msg=""):
    """Compare integer counts allowing small drift.

    Discrete values (component counts, refit iterations, ...) can legitimately
    shift by a small amount between dependency versions when marginal fits flip
    a decision threshold. ``tol=0`` enforces an exact match.
    """
    assert abs(actual - expected) <= tol, f"{msg}: {actual} differs from expected {expected} (tol={tol})"
