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
    "channels",
    "n_channels",
    "position_yx",
    "rms_noise",
    "noise_values",
    "signal_intervals",
    "signal_mask",
    "noise_spike_intervals",
    "noise_spike_mask",
    "n_fit_components",
    "amplitude_values",
    "mean_values",
    "fwhm_values",
    "reduced_chi2_value",
]
defaults = [None for field in fields]

# TODO: Spectrum is used in gp_plus, spatial_fitting and plotting but not all fields are used everywhere. Maybe define
#  multiple different Spectrum namedtuples?
Spectrum = namedtuple('Spectrum', fields, defaults=defaults)
