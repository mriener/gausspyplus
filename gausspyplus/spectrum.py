import functools
from dataclasses import dataclass
from typing import Optional, List

import numpy as np

from gausspyplus.utils.fit_quality_checks import check_residual_for_normality
from gausspyplus.utils.noise_estimation import mask_channels


@dataclass
class Spectrum:
    intensity_values: np.ndarray
    channels: np.ndarray
    rms_noise: float
    signal_intervals: Optional[List] = None
    noise_spike_intervals: Optional[List] = None

    @functools.cached_property
    def n_channels(self):
        return len(self.intensity_values)

    @functools.cached_property
    def noise_values(self):
        return np.ones(self.n_channels) * self.rms_noise

    @functools.cached_property
    def signal_mask(self) -> Optional[np.ndarray]:
        return None if not self.signal_intervals else mask_channels(
            n_channels=self.n_channels,
            ranges=self.signal_intervals,
            pad_channels=None,
            remove_intervals=self.noise_spike_intervals
        )

    @functools.cached_property
    def noise_spike_mask(self) -> Optional[np.ndarray]:
        return None if not self.noise_spike_intervals else mask_channels(
            n_channels=self.n_channels,
            ranges=[[0, self.n_channels]],
            pad_channels=None,
            remove_intervals=self.noise_spike_intervals
        )
