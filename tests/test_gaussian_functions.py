"""Tests for the pure helper functions in decomposition/gaussian_functions.py."""
import numpy as np
from lmfit import Parameters

from gausspyplus.decomposition.gaussian_functions import (
    CONVERSION_STD_TO_FWHM,
    area_of_gaussian,
    errs_vec_from_lmfit,
    multi_component_gaussian_model,
    number_of_gaussian_components,
    paramvec_to_lmfit,
    single_component_gaussian_model,
    sort_parameters,
    split_params,
    upper_limit_for_amplitude,
    vals_vec_from_lmfit,
)


def test_conversion_std_to_fwhm_constant():
    """FWHM of a Gaussian with standard deviation 1 is 2*sqrt(2*ln(2))."""
    np.testing.assert_allclose(CONVERSION_STD_TO_FWHM, 2.0 * np.sqrt(2.0 * np.log(2.0)))


def test_area_of_gaussian():
    """Integrated area of a Gaussian is amp * sigma * sqrt(2*pi), with fwhm = sigma * CONVERSION_STD_TO_FWHM."""
    amp, fwhm = 3.0, 4.7096  # sigma = 2.0
    expected = amp * 2.0 * np.sqrt(2.0 * np.pi)
    np.testing.assert_allclose(area_of_gaussian(amp, fwhm), expected, rtol=1e-4)
    # unit amplitude and sigma=1 (fwhm = CONVERSION_STD_TO_FWHM) gives area sqrt(2*pi)
    np.testing.assert_allclose(area_of_gaussian(1.0, CONVERSION_STD_TO_FWHM), np.sqrt(2.0 * np.pi), rtol=1e-12)
    # area scales linearly with the amplitude
    np.testing.assert_allclose(area_of_gaussian(2.0, 4.0), 2.0 * area_of_gaussian(1.0, 4.0), rtol=1e-12)


def test_area_of_gaussian_matches_numerical_integral():
    """The analytic area coincides with the numerically integrated single-component model."""
    amp, fwhm = 2.5, 3.0
    sigma = fwhm / CONVERSION_STD_TO_FWHM
    x = np.linspace(-50.0 * sigma, 50.0 * sigma, 200001)
    model = single_component_gaussian_model(amp, fwhm, 0.0, x)
    np.testing.assert_allclose(np.trapz(model, x), area_of_gaussian(amp, fwhm), rtol=1e-6)


def test_single_component_gaussian_model():
    """Peak value is amp at the mean, half-max at mean +/- fwhm/2, model is symmetric and tends to zero."""
    amp, fwhm, mean = 5.0, 8.0, 20.0
    x = np.arange(100, dtype=float)
    model = single_component_gaussian_model(amp, fwhm, mean, x)
    assert model.shape == x.shape
    np.testing.assert_allclose(model[x == mean], amp)
    np.testing.assert_allclose(model[x == mean - fwhm / 2.0], amp / 2.0, rtol=1e-12)
    np.testing.assert_allclose(model[x == mean + fwhm / 2.0], amp / 2.0, rtol=1e-12)
    # symmetry around the mean
    np.testing.assert_allclose(
        single_component_gaussian_model(amp, fwhm, mean, x),
        single_component_gaussian_model(amp, fwhm, mean, 2.0 * mean - x),
        rtol=1e-12,
    )
    # essentially zero several FWHM away from the mean
    tail = single_component_gaussian_model(amp, fwhm, mean, np.array([mean + 10.0 * fwhm]))
    assert tail[0] < 1e-10


def test_multi_component_gaussian_model_equals_sum_of_singles():
    """The multi-component model is the sum of the individual single-component models."""
    amps, fwhms, means = [1.0, 2.0, 0.5], [3.0, 5.0, 2.0], [10.0, 40.0, 70.0]
    x = np.arange(100, dtype=float)
    expected = np.zeros(x.size)
    for amp, fwhm, mean in zip(amps, fwhms, means):
        expected += single_component_gaussian_model(amp, fwhm, mean, x)
    np.testing.assert_allclose(multi_component_gaussian_model(amps, fwhms, means, x), expected, rtol=1e-12)
    # also accepts numpy arrays instead of lists
    np.testing.assert_allclose(
        multi_component_gaussian_model(np.array(amps), np.array(fwhms), np.array(means), x),
        expected,
        rtol=1e-12,
    )


def test_multi_component_gaussian_model_without_components():
    """An empty component list yields a zero model."""
    x = np.arange(10, dtype=float)
    np.testing.assert_array_equal(multi_component_gaussian_model([], [], [], x), np.zeros(10))


def test_split_params():
    """Parameter vector is split into amps, fwhms and means."""
    params = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    amps, fwhms, means = split_params(params, ncomps=2)
    assert amps == [1.0, 2.0]
    assert fwhms == [3.0, 4.0]
    assert means == [5.0, 6.0]
    # ncomps=0 returns three empty lists
    assert split_params(params, ncomps=0) == ([], [], [])


def test_number_of_gaussian_components():
    """Number of components is the parameter vector length divided by 3 (floor division)."""
    assert number_of_gaussian_components([1.0, 2.0, 3.0]) == 1
    assert number_of_gaussian_components([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]) == 2
    assert number_of_gaussian_components([]) == 0
    # incomplete trailing triplets are discarded
    assert number_of_gaussian_components([1.0, 2.0, 3.0, 4.0]) == 1


def test_vals_vec_from_lmfit():
    """Parameter values are returned as a list in insertion order."""
    params = paramvec_to_lmfit([1.0, 2.0, 3.0])
    assert vals_vec_from_lmfit(params) == [1.0, 2.0, 3.0]


def test_errs_vec_from_lmfit():
    """Uncertainties are returned in insertion order; None stderrs are converted to 0."""
    params = Parameters()
    params.add("p1", value=1.0)
    params.add("p2", value=2.0)
    params.add("p3", value=3.0)
    params["p1"].stderr = 0.25
    assert params["p2"].stderr is None
    assert params["p3"].stderr is None
    np.testing.assert_array_equal(errs_vec_from_lmfit(params), [0.25, 0.0, 0.0])


def test_paramvec_to_lmfit_names_and_defaults():
    """Default call creates p1..p(3N) with value/min/max set to paramvec value/0/inf."""
    paramvec = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    params = paramvec_to_lmfit(paramvec)
    assert list(params.keys()) == ["p1", "p2", "p3", "p4", "p5", "p6"]
    for i, value in enumerate(paramvec):
        param = params[f"p{i + 1}"]
        assert param.value == value
        assert param.min == 0.0
        assert param.max in (None, np.inf)  # lmfit stores an unset max as inf


def test_paramvec_to_lmfit_max_amp_and_max_fwhm():
    """max_amp/max_fwhm bound the amplitude and FWHM parameters; means stay unbounded."""
    params = paramvec_to_lmfit([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], max_amp=10.0, max_fwhm=20.0)
    assert params["p1"].max == 10.0
    assert params["p2"].max == 10.0
    assert params["p3"].max == 20.0
    assert params["p4"].max == 20.0
    assert params["p5"].max in (None, np.inf)
    assert params["p6"].max in (None, np.inf)


def test_paramvec_to_lmfit_params_min_only():
    """params_min is respected; max_amp still applies to the amplitude parameters when params_max is not given."""
    params = paramvec_to_lmfit([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], params_min=[0.5, 0.6, 1.5, 1.6, 2.5, 2.6], max_amp=10.0)
    assert (params["p1"].value, params["p1"].min, params["p1"].max) == (1.0, 0.5, 10.0)
    assert (params["p2"].value, params["p2"].min, params["p2"].max) == (2.0, 0.6, 10.0)
    assert (params["p3"].value, params["p3"].min, params["p3"].max) == (3.0, 1.5, np.inf)
    assert (params["p4"].value, params["p4"].min, params["p4"].max) == (4.0, 1.6, np.inf)
    assert (params["p5"].value, params["p5"].min, params["p5"].max) == (5.0, 2.5, np.inf)
    assert (params["p6"].value, params["p6"].min, params["p6"].max) == (6.0, 2.6, np.inf)


def test_paramvec_to_lmfit_explicit_bounds():
    """Explicitly given params_min and params_max take precedence over the defaults."""
    paramvec = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    params_min = [0.5, 0.6, 1.5, 1.6, 2.5, 2.6]
    params_max = [10.5, 10.6, 20.5, 20.6, 30.5, 30.6]
    params = paramvec_to_lmfit(paramvec, params_min=params_min, params_max=params_max)
    for i in range(6):
        assert params[f"p{i + 1}"].min == params_min[i]
        assert params[f"p{i + 1}"].max == params_max[i]


def test_paramvec_to_lmfit_params_max_overrides_max_amp():
    """If params_max is given, max_amp is silently ignored (bounds come from params_max only)."""
    params = paramvec_to_lmfit([1.0, 2.0, 3.0], max_amp=99.0, params_max=[9.0, 9.0, 9.0])
    assert params["p1"].max == 9.0


def test_sort_parameters_descending():
    """Default (descending=True) sorts components by decreasing amplitude, keeping triples consistent."""
    amps, fwhms, means = [1.0, 3.0, 2.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]
    np.testing.assert_array_equal(sort_parameters(amps, fwhms, means), [3.0, 2.0, 1.0, 5.0, 6.0, 4.0, 8.0, 9.0, 7.0])


def test_sort_parameters_ascending():
    """descending=False sorts components by increasing amplitude, keeping triples consistent."""
    amps, fwhms, means = [1.0, 3.0, 2.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]
    np.testing.assert_array_equal(
        sort_parameters(amps, fwhms, means, descending=False), [1.0, 2.0, 3.0, 4.0, 6.0, 5.0, 7.0, 9.0, 8.0]
    )


def test_sort_parameters_single_component():
    """A single component is returned unchanged."""
    np.testing.assert_array_equal(sort_parameters([1.0], [2.0], [3.0]), [1.0, 2.0, 3.0])


def test_sort_parameters_empty():
    """Empty input is handled gracefully and returns an empty array."""
    result = sort_parameters([], [], [])
    assert isinstance(result, np.ndarray)
    assert result.size == 0


def test_sort_parameters_tied_amplitudes():
    """Tied amplitudes: descending=True reverses the tied group relative to ascending order.

    Note: np.argsort(...)[::-1] is not a stable sort, so with equal amplitude values
    the fwhms/means of the tied components come back in reversed order compared to
    descending=False. The same set of (amp, fwhm, mean) triples is preserved either
    way; only their order within the tied group differs. This documents current
    behaviour rather than asserting it is the only sensible one.
    """
    amps, fwhms, means = [1.0, 1.0], [5.0, 6.0], [7.0, 8.0]
    np.testing.assert_array_equal(sort_parameters(amps, fwhms, means), [1.0, 1.0, 6.0, 5.0, 8.0, 7.0])
    np.testing.assert_array_equal(
        sort_parameters(amps, fwhms, means, descending=False), [1.0, 1.0, 5.0, 6.0, 7.0, 8.0]
    )


def test_upper_limit_for_amplitude():
    """Upper limit is the maximum intensity within the interval mean +/- 1 sigma, times the buffer factor.

    The interval half width is fwhm / CONVERSION_STD_TO_FWHM (i.e. one standard
    deviation); the upper slice index additionally extends 2 channels beyond
    mean + sigma (see get_slice_indices_for_interval).
    """
    intensity_values = np.arange(100, dtype=float)
    mean, fwhm = 50.0, 10.0
    sigma = fwhm / CONVERSION_STD_TO_FWHM  # ~4.2466
    idx_low = max(0, int(mean - sigma))  # 45
    idx_upp = int(mean + sigma) + 2  # 56
    expected = np.max(intensity_values[idx_low:idx_upp])
    np.testing.assert_allclose(upper_limit_for_amplitude(intensity_values, mean, fwhm), expected)
    # buffer_factor scales the result
    np.testing.assert_allclose(
        upper_limit_for_amplitude(intensity_values, mean, fwhm, buffer_factor=2.0), 2.0 * expected
    )


def test_upper_limit_for_amplitude_lower_bound_clamped():
    """A mean less than one sigma clamps the lower slice index at 0 instead of going negative."""
    intensity_values = np.arange(100, dtype=float)
    mean, fwhm = 2.0, 10.0
    sigma = fwhm / CONVERSION_STD_TO_FWHM
    idx_low = max(0, int(mean - sigma))
    idx_upp = int(mean + sigma) + 2
    np.testing.assert_allclose(
        upper_limit_for_amplitude(intensity_values, mean, fwhm), np.max(intensity_values[idx_low:idx_upp])
    )
