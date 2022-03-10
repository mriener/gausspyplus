from collections import namedtuple


FitResults = namedtuple('FitResults', ["amplitude_values",
                                       "mean_values",
                                       "fwhm_values",
                                       "intensity_values",
                                       "position_yx",
                                       "signal_intervals",
                                       "rms_noise",
                                       "reduced_chi2_value",
                                       "index"])

PreparedSpectrum = namedtuple('PreparedSpectrum', ["intensity_values",
                                                   "position_yx",
                                                   "rms_noise",
                                                   "signal_intervals",
                                                   "noise_spike_intervals",
                                                   "index"])

fields = [
    "index",
    "intensity_values",
    "position_yx",
    "rms_noise",
    "signal_intervals",
    "noise_spike_intervals",
    "n_fit_components",
    "amplitude_values",
    "mean_values",
    "fwhm_values",
    "reduced_chi2_value",
]
defaults = [None for field in fields]

Spectrum = namedtuple('Spectrum', fields, defaults=defaults)
