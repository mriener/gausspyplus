from collections import namedtuple
from dataclasses import dataclass, field
from typing import Optional, Tuple, List

import astropy
from astropy import units as u

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


@dataclass
class SettingsDefault:
    log_output: bool = field(
        default=True,
        metadata={
            "description": "log messages printed to the terminal in 'gpy_log' directory [True/False]",
            "simple": False
        }
    )
    verbose: bool = field(
        default=True,
        metadata={
            "description": "Print messages to the terminal [True/False]",
            "simple": False
        }
    )
    overwrite: bool = field(
        default=True,
        metadata={
            "description": "Overwrite files [True/False]",
            "simple": False
        }
    )
    suffix: Optional[str] = field(
        default=None,
        metadata={
            "description": "Suffix added to filename [str]",
            "simple": False
        }
    )
    use_ncpus: Optional[int] = field(
        default=None,
        metadata={
            "description": "Number of CPUs used for parallel processing. "
                           "By default 75% of all CPUs on the machine are used. [int]",
            "simple": True
        }
    )
    snr: float = field(
        default=3.,
        metadata={
            "description": "Required minimum signal-to-noise ratio for data peak. [float]",
            "simple": True
        }
    )
    significance: float = field(
        default=5.,
        metadata={
            "description": "Required minimum value for significance criterion. [float]",
            "simple": True
        }
    )
    snr_noise_spike: float = field(
        default=5.,
        metadata={
            "description": "Required signal-to-noise ratio for negative data values to be counted as noise spikes. "
                           "[float]",
            "simple": False
        }
    )
    min_fwhm: float = field(
        default=1.,
        metadata={
            "description": "Required minimum value for FWHM values of fitted Gaussian components specified in "
                           "fractions of channels. [float]",
            "simple": False
        }
    )
    max_fwhm: Optional[float] = field(
        default=None,
        metadata={
            "description": "Enforced maximum value for FWHM parameter specified in fractions of channels. "
                           "Use with caution! Can lead to artifacts in the fitting. [float]",
            "simple": False
        }
    )
    separation_factor: float = field(
        default=0.8493218,
        metadata={
            "description": "The required minimum separation between two Gaussian components (mean1, fwhm1) and "
                           "(mean2, fwhm2) is determined as separation_factor * min(fwhm1, fwhm2). [float]",
            "simple": False
        }
    )
    fwhm_factor: float = field(
        default=2.,
        metadata={
            "description": "Factor by which the FWHM value of a fit component has to exceed all other (neighboring) "
                           "fit components to get flagged [float]",
            "simple": False
        }
    )
    min_pvalue: float = field(
        default=0.01,
        metadata={
            "description": "p-value for the null hypothesis that the normalised residual resembles a normal "
                           "distribution. [float]",
            "simple": False
        }
    )
    max_ncomps: Optional[int] = field(
        default=None,
        metadata={
            "description": "Maximum number of allowed fit components per spectrum. Use with caution. [int]",
            "simple": False
        }
    )
    two_phase_decomposition: bool = field(
        default=True,
        metadata={
            "description": "Whether to use one or two smoothing parameters for the decomposition. [True/False]",
            "simple": False
        }
    )
    refit_blended: bool = field(
        default=True,
        metadata={
            "description": "Refit blended components. [True/False]",
            "simple": True
        }
    )
    refit_broad: bool = field(
        default=True,
        metadata={
            "description": "Refit broad components. [True/False]",
            "simple": True
        }
    )
    refit_neg_res_peak: bool = field(
        default=True,
        metadata={
            "description": "Refit negative residual features. [True/False]",
            "simple": True
        }
    )
    refit_rchi2: bool = field(
        default=False,
        metadata={
            "description": "Refit spectra with high reduced chi-square value. [True/False]",
            "simple": False
        }
    )
    refit_residual: bool = field(
        default=True,
        metadata={
            "description": "Refit spectra with non-Gaussian distributed residuals. [True/False]",
            "simple": True
        }
    )
    refit_ncomps: bool = field(
        default=True,
        metadata={
            "description": "Refit if number of fitted components is not compatible with neighbors. [True/False]",
            "simple": True
        }
    )
    p_limit: float = field(
        default=0.02,
        metadata={
            "description": "Probability threshold given in percent for features of consecutive positive or negative "
                           "channels to be counted as more likely to be a noise feature [float]",
            "simple": False
        }
    )
    signal_mask: bool = field(
        default=True,
        metadata={
            "description": "Constrict goodness-of-fit calculations to spectral regions estimated to contain signal "
                           "[True/False]",
            "simple": False
        }
    )
    pad_channels: int = field(
        default=5,
        metadata={
            "description": "Number of channels by which an interval (low, upp) gets extended on both sides, resulting "
                           "in (low - pad_channels, upp + pad_channels). [int]",
            "simple": False
        }
    )
    min_channels: int = field(
        default=100,
        metadata={
            "description": "Required minimum number of spectral channels that the signal ranges should contain. "
                           "[int]",
            "simple": False
        }
    )
    # TODO: Change default argument from mutable to immutable
    mask_out_ranges: Optional[List] = field(
        default=None,
        metadata={
            "description": "Mask out ranges in the spectrum; specified as a list of tuples [(low1, upp1), ..., "
                           "(lowN, uppN)]",
            "simple": False
        }
    )
    random_seed: int = field(
        default=111,
        metadata={
            "description": "Seed for random processes [int]",
            "simple": True
        }
    )
    main_beam_efficiency: Optional[float] = field(
        default=None,
        metadata={
            "description": "Specify if intensity values should be corrected by the main beam efficiency given in "
                           "percent. [float]",
            "simple": False
        }
    )
    vel_unit: str = field(
        default="u.km / u.s",
        metadata={
            "description": "Unit to which velocity values will be converted. [astropy.units]",
            "simple": True
        }
    )
    testing: bool = field(
        default=False,
        metadata={
            "description": "Testing mode; only decomposes a single spectrum. [True/False]",
            "simple": False
        }
    )


@dataclass
class SettingsTraining:
    n_spectra: int = field(
        default=100,
        metadata={
            "description": "Number of spectra contained in the training set [int]",
            "simple": True
        }
    )
    order: int = field(
        default=6,
        metadata={
            "description": "Minimum number of spectral channels a peak has to contain on either side [int]",
            "simple": True
        }
    )
    rchi2_limit: float = field(
        default=1.5,
        metadata={
            "description": "Maximium value of reduced chi-squared for decomposition result [float]",
            "simple": True
        }
    )
    use_all: bool = field(
        default=False,
        metadata={
            "description": "Use all spectra in FITS cube as training set [True/False]",
            "simple": False
        }
    )
    params_from_data: bool = field(
        default=True,
        metadata={
            "description": " [True/False]",
            "simple": False
        }
    )
    alpha1_initial: Optional[float] = field(
        default=None,
        metadata={
            "description": " [float]",
            "simple": False
        }
    )
    alpha2_initial: Optional[float] = field(
        default=None,
        metadata={
            "description": " [float]",
            "simple": False
        }
    )
    snr_thresh: Optional[float] = field(
        default=None,
        metadata={
            "description": " [float]",
            "simple": False
        }
    )
    snr2_thresh: Optional[float] = field(
        default=None,
        metadata={
            "description": " [float]",
            "simple": False
        }
    )


@dataclass
class SettingsPreparation:
    n_spectra_rms: int = field(
        default=1000,
        metadata={
            "description": "Number of spectra used to estimate average root-mean-square noise [int]",
            "simple": False
        }
    )
    gausspy_pickle: bool = field(
        default=True,
        metadata={
            "description": "Save the prepared FITS cube as pickle file [bool]",
            "simple": False
        }
    )
    data_location: Optional[Tuple] = field(
        default=None,
        metadata={
            "description": "Only used for 'testing = True'; specify location of spectrum used for test decomposition "
                           "as (y, x) [tuple]",
            "simple": False
        }
    )
    simulation: bool = field(
        default=False,
        metadata={
            "description": "Set to 'True' if FITS cube contains simulation data without noise [bool]",
            "simple": False
        }
    )
    rms_from_data: bool = field(
        default=True,
        metadata={
            "description": "Calculate the root-mean-square noise from the data [bool]",
            "simple": True
        }
    )
    average_rms: Optional[float] = field(
        default=None,
        metadata={
            "description": "Average data of the FITS cube; if no value is supplied it is estimated from the data "
                           "[float]",
            "simple": True
        }
    )


@dataclass
class SettingsDecomposition:
    save_initial_guesses: bool = field(
        default=False,
        metadata={
            "description": "Save initial component guesses of GaussPy as pickle file [bool]",
            "simple": False
        }
    )
    alpha1: Optional[float] = field(
        default=None,
        metadata={
            "description": "First smoothing parameter [float]",
            "simple": True
        }
    )
    alpha2: Optional[float] = field(
        default=None,
        metadata={
            "description": "Second smoothing parameter (only used if 'two_phase_decomposition = True') [float]",
            "simple": True
        }
    )
    snr_thresh: Optional[float] = field(
        default=None,
        metadata={
            "description": "Signal-to-noise threshold used in GaussPy decomposition for original spectrum. "
                           "Defaults to 'snr' if not specified. [float]",
            "simple": False
        }
    )
    snr2_thresh: Optional[float] = field(
        default=None,
        metadata={
            "description": "Signal-to-noise threshold used in GaussPy decomposition for second derivative of spectrum. "
                           "Defaults to 'snr' if not specified. [float]",
            "simple": False
        }
    )
    improve_fitting: bool = field(
        default=True,
        metadata={
            "description": "Use the improved fitting routine. [bool]",
            "simple": False
        }
    )
    exclude_means_outside_channel_range: bool = field(
        default=True,
        metadata={
            "description": "Exclude Gaussian fit components if their mean position is outside the channel range. "
                           "[bool]",
            "simple": False
        }
    )
    snr_fit: Optional[float] = field(
        default=None,
        metadata={
            "description": "Required minimum signal-to-noise value for fitted components. Defaults to 'snr/2' if not "
                           "specified. [float]",
            "simple": False
        }
    )
    snr_negative: Optional[float] = field(
        default=None,
        metadata={
            "description": "Required minimum signal-to-noise value for negative data peaks. "
                           "Used in the search for negative residual peaks. Defaults to 'snr' if not specified. [float]",
            "simple": False
        }
    )
    max_amp_factor: float = field(
        default=1.1,
        metadata={
            "description": "Factor by which the maximum data value is multiplied to get a maximum limit for the "
                           "fitted amplitudes. [float]",
            "simple": False
        }
    )


@dataclass
class SettingsSpatialFitting:
    exclude_flagged: bool = field(
        default=False,
        metadata={
            "description": "Exclude all flagged spectra as possible refit solutions. [bool]",
            "simple": False
        }
    )
    rchi2_limit: Optional[float] = field(
        default=None,
        metadata={
            "description": "Maximium value for the reduced chi-squared above which the fit gets flagged [float]",
            "simple": False
        }
    )
    rchi2_limit_refit: Optional[float] = field(
        default=None,
        metadata={
            "description": "Defaults to 'rchi2_limit' if not specified. [float]",
            "simple": False
        }
    )
    max_diff_comps: int = field(
        default=1,
        metadata={
            "description": "Maximum allowed difference in the number of fitted components compared to weighted "
                           "median of immediate neighbors [int]",
            "simple": True
        }
    )
    max_jump_comps: int = field(
        default=2,
        metadata={
            "description": "Maximum allowed difference in the number of fitted components between individual "
                           "neighboring spectra [int]",
            "simple": True
        }
    )
    n_max_jump_comps: int = field(
        default=1,
        metadata={
            "description": "Maximum number of allowed 'max_jump_comps' occurrences for a single spectrum. [int]",
            "simple": True
        }
    )
    max_refitting_iteration: int = field(
        default=30,
        metadata={
            "description": "Maximum number for refitting iterations. [int]",
            "simple": False
        }
    )
    use_all_neighors: bool = field(
        default=False,
        metadata={
            "description": "Use flagged neighbors as refit solutions in case the refit was not possible with fit "
                           "solutions from unflagged neighbors. [bool]",
            "simple": True
        }
    )
    flag_blended: bool = field(
        default=True,
        metadata={
            "description": "Flag spectra with blended fit components. [bool]",
            "simple": False
        }
    )
    flag_neg_res_peak: bool = field(
        default=True,
        metadata={
            "description": "Flag spectra with negative residual features. [bool]",
            "simple": False
        }
    )
    flag_rchi2: bool = field(
        default=False,
        metadata={
            "description": "Flag spectra with high reduced chi-square values. [bool]",
            "simple": False
        }
    )
    flag_residual: bool = field(
        default=True,
        metadata={
            "description": "Flag spectra with non-Gaussian distributed residuals. [bool]",
            "simple": False
        }
    )
    flag_broad: bool = field(
        default=True,
        metadata={
            "description": "Flag spectra with broad fit components. [bool]",
            "simple": False
        }
    )
    flag_ncomps: bool = field(
        default=True,
        metadata={
            "description": "Flag spectra with number of fit components incompatible with neighbors. [bool]",
            "simple": False
        }
    )
    mean_separation: float = field(
        default=2.,
        metadata={
            "description": "Maximum difference in offset positions of fit components for grouping. [float]",
            "simple": True
        }
    )
    fwhm_separation: float = field(
        default=4.,
        metadata={
            "description": "Maximum difference in FWHM values of fit components for grouping. [float]",
            "simple": True
        }
    )
    fwhm_factor_refit: Optional[float] = field(
        default=None,
        metadata={
            "description": "Defaults to 'fwhm_factor' if not specified. [float]",
            "simple": False
        }
    )
    broad_neighbor_fraction: float = field(
        default=0.5,
        metadata={
            "description": "Spectra get flagged as containing broad components if the FWHM value of one of their "
                           "fit components exceeds the FWHM values of all fit components for this fraction of "
                           "neighbors [float]",
            "simple": False
        }
    )
    min_weight: float = field(
        default=0.5,
        metadata={
            "description": "Minimum weight threshold for phase 2 of spatially coherent refitting. [float]",
            "simple": True
        }
    )
    weight_factor: float = field(
        default=2.,
        metadata={
            "description": "Factor that determines the weight given to neighboring spectra located at a distance "
                           "of 1 and 2 pixels. [float]",
            "simple": False
        }
    )
    only_print_flags: bool = field(
        default=False,
        metadata={
            "description": "Only print flags in terminal without refitting. [bool]",
            "simple": False
        }
    )
