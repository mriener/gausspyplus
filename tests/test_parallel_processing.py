"""Tests for the parallel processing helper."""

import pytest

from gausspyplus.parallel_processing.parallel_processing import func, parallel_process


def _square(x):
    return x * x


def test_parallel_process_front_num_zero():
    # n_jobs=1 runs serially; front_num=0 must yield an empty serial prefix
    assert parallel_process([1, 2, 3, 4], _square, n_jobs=1, front_num=0) == [1, 4, 9, 16]


def test_parallel_process_front_num_larger_than_array():
    # a serial prefix larger than the array consumes the whole array; the
    # parallel remainder is empty
    assert parallel_process([1, 2, 3, 4], _square, n_jobs=1, front_num=10) == [1, 4, 9, 16]


def test_parallel_process_front_num_subset():
    assert parallel_process([1, 2, 3, 4], _square, n_jobs=1, front_num=2) == [1, 4, 9, 16]


def test_func_unknown_function_name_raises():
    with pytest.raises(ValueError, match="Unknown function name"):
        func(use_ncpus=1, function="does_not_exist")
