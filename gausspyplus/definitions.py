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
