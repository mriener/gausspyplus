"""Shared helper functions for the GaussPy+ test suite."""
import pickle

import numpy as np


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
