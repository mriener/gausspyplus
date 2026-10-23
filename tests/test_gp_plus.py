"""Tests for the fit-improvement logic in decomposition/gp_plus.py.

get_best_fit_model discards fit components that violate the quality control
criteria and refits the remaining components. These refits must keep the
parameter bounds of the surviving components: the bounds are pruned alongside
the removed components and propagated to the refit and to the returned model.
The pipeline golden values in test_pipeline.py cannot pin this behaviour
tightly, because their fit-sum tolerances are deliberately loose enough to
absorb cross-platform drift of marginally degenerate refits (see the module
docstring there).
"""

import numpy as np
import pytest

from gausspyplus.definitions.definitions import SettingsImproveFit
from gausspyplus.definitions.model import Model
from gausspyplus.definitions.spectrum import Spectrum
from gausspyplus.decomposition.gaussian_functions import multi_component_gaussian_model
from gausspyplus.decomposition.gp_plus import get_best_fit_model


def _settings_improve_fit() -> SettingsImproveFit:
    return SettingsImproveFit(
        improve_fitting=True,
        min_fwhm=1.0,
        max_fwhm=None,
        snr=3.0,
        snr_fit=1.5,
        significance=5.0,
        snr_negative=3.0,
        rchi2_limit=1.5,
        max_amp_factor=2.0,
        refit_neg_res_peak=False,
        refit_broad=False,
        refit_blended=False,
        fwhm_factor=2.0,
        separation_factor=0.8493218,
        exclude_means_outside_channel_range=True,
        min_pvalue=0.01,
        max_ncomps=None,
    )


def test_quality_control_refit_keeps_component_bounds():
    """A refit triggered by quality control keeps and prunes the bounds.

    The fit for the second component fails the signal-to-noise criterion, so
    the component is discarded and the first component is refit. The refit and
    the returned model must carry the first component's bounds.
    """
    channels = np.arange(100, dtype=float)
    #  A single Gaussian component with a small deterministic perturbation:
    #  the fit converges to it deterministically on every platform.
    intensity_values = multi_component_gaussian_model([1.0], [6.0], [50.0], channels)
    intensity_values = intensity_values + 0.001 * np.sin(channels / 5.0)
    spectrum = Spectrum(intensity_values=intensity_values, channels=channels, rms_noise=0.1)

    #  The second component sits on a signal-free region of the spectrum; its
    #  mean is confined there via the bounds, so the fit drives its amplitude
    #  to zero and it fails the snr_fit criterion.
    params_fit = [1.0, 0.01, 6.0, 4.0, 50.0, 10.0]
    params_min = [0.0, 0.0, 2.0, 0.5, 45.0, 5.0]
    params_max = [2.0, 2.0, 8.0, 5.0, 55.0, 15.0]

    model = get_best_fit_model(
        model=Model(spectrum=spectrum),
        params_fit=params_fit,
        settings_improve_fit=_settings_improve_fit(),
        params_min=params_min,
        params_max=params_max,
    )

    #  The signal-to-noise criterion (quality control flag 2) discarded the
    #  second component and triggered a refit of the first one.
    assert model.n_components == 1
    assert 2 in model.quality_control

    #  The bounds were pruned to the surviving component and propagated to the
    #  refit and to the returned model.
    assert model.parameters_min_values == pytest.approx([0.0, 2.0, 45.0])
    assert model.parameters_max_values == pytest.approx([2.0, 8.0, 55.0])

    #  The refit converged to the underlying component and stayed within its
    #  bounds.
    assert model.amps == pytest.approx([1.0], rel=1e-2)
    assert model.fwhms == pytest.approx([6.0], rel=1e-2)
    assert model.means == pytest.approx([50.0], rel=1e-2)
    assert 2.0 <= model.fwhms[0] <= 8.0
