"""Unit tests for the functions in spatial_fitting/flags.py."""
import pytest

from gausspyplus.spatial_fitting.flags import (
    get_flags,
    get_flags_broad,
    get_flags_centroids,
    get_flags_ncomps,
    get_flags_pvalue,
    get_flags_rchi2,
)


def test_get_flags():
    #  no flagged features before or after the refit
    assert get_flags(0, 0) == (0, 0)
    #  old fit contained flagged features, new fit does not
    assert get_flags(2, 0) == (1, 0)
    #  old fit did not contain flagged features, new fit does
    assert get_flags(0, 2) == (0, 1)
    #  new fit contains more flagged features than the old fit
    assert get_flags(2, 3) == (1, 2)
    #  new and old fit contain the same number of flagged features
    assert get_flags(3, 3) == (1, 1)


def test_get_flags_rchi2():
    #  old and new fit are both above the limit
    assert get_flags_rchi2(3.0, 4.0, 2.0) == (1, 1)
    #  old fit is above the limit, new fit below
    assert get_flags_rchi2(3.0, 1.0, 2.0) == (1, 0)
    #  old fit is below the limit, new fit above
    assert get_flags_rchi2(1.0, 3.0, 2.0) == (0, 1)
    #  both fits below the limit: new fit is closer to rchi2 = 1 and gets rewarded
    assert get_flags_rchi2(0.5, 0.9, 2.0) == (1, 0)
    #  both fits below the limit: old fit is closer to rchi2 = 1; no reward
    assert get_flags_rchi2(0.9, 0.5, 2.0) == (0, 0)
    #  both fits below the limit and equidistant from rchi2 = 1: no reward
    assert get_flags_rchi2(0.8, 1.2, 2.0) == (0, 0)
    #  rchi2 values exactly at the limit are not flagged (strict inequality)
    assert get_flags_rchi2(2.0, 1.0, 2.0) == (0, 0)


def test_get_flags_pvalue():
    #  old fit below the minimum p-value, new fit above
    assert get_flags_pvalue(0.01, 0.5, 0.05) == (1, 0)
    #  both fits below the minimum p-value; new fit got worse and is punished
    assert get_flags_pvalue(0.01, 0.001, 0.05) == (1, 2)
    #  both fits below the minimum p-value; new fit improved but is still below
    assert get_flags_pvalue(0.01, 0.04, 0.05) == (1, 1)
    #  both fits above the minimum p-value; new fit got worse and is punished
    assert get_flags_pvalue(0.5, 0.1, 0.05) == (0, 1)
    #  both fits above the minimum p-value; new fit improved
    assert get_flags_pvalue(0.1, 0.5, 0.05) == (0, 0)
    #  p-value exactly at the minimum is not flagged (strict inequality)
    assert get_flags_pvalue(0.05, 0.5, 0.05) == (0, 0)


def test_get_flags_broad():
    #  broad component present in old fit; maximum FWHM unchanged in new fit
    assert get_flags_broad([10.0, 20.0], [20.0, 5.0], True) == (1, 1)
    #  broad component present in old fit; new fit got even broader and is punished
    assert get_flags_broad([10.0], [20.0], True) == (1, 2)
    #  broad component present in old fit; new fit got narrower
    assert get_flags_broad([10.0, 20.0], [15.0], True) == (1, 0)
    #  no broad component in old fit; new fit introduced a broad component
    assert get_flags_broad([5.0], [5.0, 20.0], False) == (0, 1)
    #  ... but not if the FWHM separation criterion is not fulfilled
    assert get_flags_broad([5.0], [5.0, 20.0], False, fwhm_separation=20.0) == (0, 0)
    #  ... or the FWHM factor criterion is not fulfilled
    assert get_flags_broad([5.0], [5.0, 9.0], False) == (0, 0)
    #  a new fit with a single component never gets flagged as broad here
    assert get_flags_broad([5.0], [50.0], False) == (0, 0)
    #  custom FWHM factor: 19.0 > 1.5 * 10.0 and separation 9.0 > 4.0
    assert get_flags_broad([10.0], [10.0, 19.0], False, fwhm_factor=1.5) == (0, 1)


@pytest.mark.xfail(
    reason="known bug: get_flags_broad compares the maximum FWHM values with exact float equality "
    "(see TODO in source); mathematically equal values that differ by float round-off must not be "
    "punished as 'got even broader'",
    strict=True,
)
def test_get_flags_broad_unchanged_max_float_roundoff():
    #  0.1 + 0.2 == 0.3 mathematically, so the maximum FWHM value is unchanged
    #  and the new fit must not be flagged as broader
    assert get_flags_broad([0.3], [0.1 + 0.2], True) == (1, 1)


def test_get_flags_ncomps():
    #  no jumps or differences above the limits before or after the refit
    assert get_flags_ncomps(1, 1, 0, 0, 1, 0) == (0, 0)
    #  old fit flagged due to too many component jumps; new fit is fine
    assert get_flags_ncomps(0, 0, 1, 0, 5, 0) == (1, 0)
    #  old fit flagged and new fit has more component jumps: punished
    assert get_flags_ncomps(0, 0, 1, 2, 5, 0) == (1, 2)
    #  new fit exceeds the limit for the difference of components and got worse
    assert get_flags_ncomps(0, 2, 0, 0, 1, 0) == (0, 2)
    #  difference of components increased but stayed within the limit: still punished
    assert get_flags_ncomps(0, 1, 0, 0, 1, 0) == (0, 1)
    #  old fit flagged due to difference of components exceeding the limit
    assert get_flags_ncomps(2, 0, 0, 0, 1, 0) == (1, 0)


def test_get_flags_centroids():
    interval = [5, 15]
    #  new fit has the required number of centroids within the interval
    assert get_flags_centroids([30], [10], interval, 1) == (2, 0)
    #  required: 3 centroids; new fit got closer to the required number and is rewarded
    assert get_flags_centroids([10], [10, 12], interval, 3) == (2, 1)
    #  required: 3 centroids; new fit got farther from the required number; old fit is favored
    assert get_flags_centroids([10, 12], [10], interval, 3) == (1, 2)
    #  required: 2 centroids; old and new fit are equidistant from the required number
    assert get_flags_centroids([10], [10, 11, 12], interval, 2) == (2, 2)
    #  interval bounds are exclusive: a centroid exactly at the bound does not count
    assert get_flags_centroids([5], [10], interval, 1) == (2, 0)
    #  ... a centroid exactly at the upper bound does not count either; losing
    #  the previously required centroid is punished
    assert get_flags_centroids([10], [15], interval, 1) == (1, 2)
