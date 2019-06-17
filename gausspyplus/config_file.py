# @Author: riener
# @Date:   2019-03-03T20:27:37+01:00
# @Filename: config_file.py
# @Last modified by:   riener
# @Last modified time: 2019-04-08T10:08:05+02:00

import ast
import configparser
import collections
import os

from astropy import units as u

from .utils.output import save_file


def append_keywords(config_file, dct, all_keywords=False, description=True):
    for key in dct.keys():
        if all_keywords:
            if description:
                config_file.append(
                    '\n\n# {}'.format(dct[key]['description']))
            config_file.append('\n{} = {}'.format(key, dct[key]['default']))
        else:
            if dct[key]['simple']:
                if description:
                    config_file.append(
                        '\n\n# {}'.format(dct[key]['description']))
                config_file.append('\n{} = {}'.format(key, dct[key]['default']))
    return config_file


def make(all_keywords=False, description=True, output_directory='',
         filename='gausspy+.ini'):
    """Create a GaussPy+ configuration file.

    Parameters
    ----------
    all_keywords : bool
        Default is `False`, which includes only the most essential parameters. If set to `True`, include all parameters in the configuration file.
    description : bool
        Default is `True`, which includes descriptions of the parameters in the configuration file.
    output_directory : string
        Directory to which configuration file gets saved.
    filename : string
        Name of the configuration file.

    Returns
    -------
    type
        Description of returned object.

    """
    config_file = str('#  Configuration file for GaussPy+\n\n')

    default = [
        ('log_output', {
            'default': 'True',
            'description': "log messages printed to the terminal in 'gpy_log' directory [True/False]",
            'simple': False}),
        ('verbose', {
            'default': 'True',
            'description': "print messages to the terminal [True/False]",
            'simple': False}),
        ('overwrite', {
            'default': 'True',
            'description': "overwrite files [True/False]",
            'simple': False}),
        ('suffix', {
            'default': '""',
            'description': "suffix added to filename [str]",
            'simple': False}),
        ('use_ncpus', {
            'default': 'None',
            'description': "number of CPUs used in parallel processing. By default 75% of all CPUs on the machine are used. [int]",
            'simple': True}),

        ('snr', {
            'default': '3.',
            'description': "Required minimum signal-to-noise ratio for data peak. [float]",
            'simple': True}),
        ('significance', {
            'default': '5.',
            'description': "Required minimum value for significance criterion. [float]",
            'simple': True}),
        ('snr_noise_spike', {
            'default': '5.',
            'description': "Required signal-to-noise ratio for negative data values to be counted as noise spikes. [float]",
            'simple': False}),
        ('min_fwhm', {
            'default': '1.',
            'description': "Required minimum value for FWHM values of fitted Gaussian components specified in fractions of channels. [float]",
            'simple': False}),
        ('max_fwhm', {
            'default': 'None',
            'description': "Enforced maximum value for FWHM parameter specified in fractions of channels. Use with caution! Can lead to artifacts in the fitting. [float]",
            'simple': False}),
        ('separation_factor', {
            'default': '0.8493218',
            'description': "The required minimum separation between two Gaussian components (mean1, fwhm1) and (mean2, fwhm2) is determined as separation_factor * min(fwhm1, fwhm2). [float]",
            'simple': False}),
        ('fwhm_factor', {
            'default': '2.',
            'description': "factor by which the FWHM value of a fit component has to exceed all other (neighboring) fit components to get flagged [float]",
            'simple': False}),
        ('min_pvalue', {
            'default': '0.01',
            'description': "p-value for the null hypothesis that the normalised residual resembles a normal distribution. [float]",
            'simple': False}),

        ('two_phase_decomposition', {
            'default': 'True',
            'description': "Whether to use one or two smoothing parameters for the decomposition. [True/False]",
            'simple': False}),

        ('refit_blended', {
            'default': 'True',
            'description': "Refit blended components. [True/False]",
            'simple': True}),
        ('refit_broad', {
            'default': 'True',
            'description': "Refit broad components. [True/False]",
            'simple': True}),
        ('refit_neg_res_peak', {
            'default': 'True',
            'description': "Refit negative residual features. [True/False]",
            'simple': True}),
        ('refit_rchi2', {
            'default': 'False',
            'description': "Refit spectra with high reduced chi-square value. [True/False]",
            'simple': False}),
        ('refit_residual', {
            'default': 'True',
            'description': "Refit spectra with non-Gaussian distributed residuals. [True/False]",
            'simple': True}),
        ('refit_ncomps', {
            'default': 'True',
            'description': "Refit if number of fitted components is not compatible with neighbors. [True/False]",
            'simple': True}),

        ('p_limit', {
            'default': '0.02',
            'description': "Probability threshold given in percent for features of consecutive positive or negative channels to be counted as more likely to be a noise feature [float]",
            'simple': False}),
        ('signal_mask', {
            'default': 'True',
            'description': "Constrict goodness-of-fit calculations to spectral regions estimated to contain signal [True/False]",
            'simple': False}),
        ('pad_channels', {
            'default': '5',
            'description': "Number of channels by which an interval (low, upp) gets extended on both sides, resulting in (low - pad_channels, upp + pad_channels). [int]",
            'simple': False}),
        ('min_channels', {
            'default': '100',
            'description': "Required minimum number of spectral channels that the signal ranges should contain. [int]",
            'simple': False}),
        ('mask_out_ranges', {
            'default': '[]',
            'description': "Mask out ranges in the spectrum; specified as a list of tuples [(low1, upp1), ..., (lowN, uppN)]",
            'simple': False}),

        ('random_seed', {
            'default': '111',
            'description': "Seed for random processes [int]",
            'simple': True}),

        ('main_beam_efficiency', {
            'default': 'None',
            'description': "Specify if intensity values should be corrected by the main beam efficiency given in percent. [float]",
            'simple': False}),
        ('vel_unit', {
            'default': 'u.km / u.s',
            'description': "Unit to which velocity values will be converted. [astropy.units]",
            'simple': True}),
        ('testing', {
            'default': 'False',
            'description': "Testing mode; only decomposes a single spectrum. [True/False]",
            'simple': False})
        ]
    dct_default = collections.OrderedDict(default)

    training = [
        ('n_spectra', {
            'default': '100',
            'description': "Number of spectra contained in the training set [int]",
            'simple': True}),
        ('order', {
            'default': '6',
            'description': "Minimum number of spectral channels a peak has to contain on either side [int]",
            'simple': True}),
        ('rchi2_limit', {
            'default': '1.5',
            'description': "maximium value of reduced chi-squared for decomposition result [float]",
            'simple': True}),
        ('use_all', {
            'default': 'False',
            'description': "Use all spectra in FITS cube as training set [True/False]",
            'simple': False}),

        ('params_from_data', {
            'default': 'True',
            'description': " [True/False]",
            'simple': False}),
        ('alpha1_initial', {
            'default': 'None',
            'description': " [float]",
            'simple': False}),
        ('alpha2_initial', {
            'default': 'None',
            'description': " [float]",
            'simple': False}),
        ('snr_thresh', {
            'default': 'None',
            'description': " [float]",
            'simple': False}),
        ('snr2_thresh', {
            'default': 'None',
            'description': " [float]",
            'simple': False})
    ]
    dct_training = collections.OrderedDict(training)

    preparation = [
        ('n_spectra_rms', {
            'default': '1000',
            'description': "Number of spectra used to estimate average root-mean-square noise [int]",
            'simple': False}),

        ('gausspy_pickle', {
            'default': 'True',
            'description': "Save the prepared FITS cube as pickle file [bool]",
            'simple': False}),
        ('data_location', {
            'default': 'None',
            'description': "Only used for 'testing = True'; specify location of spectrum used for test decomposition as (y, x) [tuple]",
            'simple': False}),
        ('simulation', {
            'default': 'False',
            'description': "Set to 'True' if FITS cube contains simulation data without noise [bool]",
            'simple': False}),

        ('rms_from_data', {
            'default': 'True',
            'description': "Calculate the root-mean-square noise from the data [bool]",
            'simple': True}),
        ('average_rms', {
            'default': 'None',
            'description': "Average data of the FITS cube; if no value is supplied it is estimated from the data [float]",
            'simple': True})
    ]
    dct_preparation = collections.OrderedDict(preparation)

    decomposition = [
        # ('gausspy_decomposition', {
        #     'default': 'True',
        #     'description': " [bool]",
        #     'simple': False}),
        ('save_initial_guesses', {
            'default': 'False',
            'description': "Save initial component guesses of GaussPy as pickle file [bool]",
            'simple': False}),
        ('alpha1', {
            'default': 'None',
            'description': "First smoothing parameter [float]",
            'simple': True}),
        ('alpha2', {
            'default': 'None',
            'description': "Second smoothing parameter (only used if 'two_phase_decomposition = True') [float]",
            'simple': True}),
        ('snr_thresh', {
            'default': 'None',
            'description': "Signal-to-noise threshold used in GaussPy decomposition for original spectrum. Defaults to 'snr' if not specified. [float]",
            'simple': False}),
        ('snr2_thresh', {
            'default': 'None',
            'description': "Signal-to-noise threshold used in GaussPy decomposition for second derivative of spectrum. Defaults to 'snr' if not specified. [float]",
            'simple': False}),

        ('improve_fitting', {
            'default': 'True',
            'description': "Use the improved fitting routine. [bool]",
            'simple': False}),
        ('exclude_means_outside_channel_range', {
            'default': 'True',
            'description': "Exclude Gaussian fit components if their mean position is outside the channel range. [bool]",
            'simple': False}),
        ('snr_fit', {
            'default': 'None',
            'description': "Required minimum signal-to-noise value for fitted components. Defaults to 'snr/2' if not specified. [float]",
            'simple': False}),
        ('snr_negative', {
            'default': 'None',
            'description': "Required minimum signal-to-noise value for negative data peaks. Used in the search for negative residual peaks. Defaults to 'snr' if not specified. [float]",
            'simple': False}),
        ('max_amp_factor', {
            'default': '1.1',
            'description': "Factor by which the maximum data value is multiplied to get a maximum limit for the fitted amplitudes. [float]",
            'simple': False})
    ]
    dct_decomposition = collections.OrderedDict(decomposition)

    spatial_fitting = [
        # , {
        #     'default': ,
        #     'description': " []",
        #     'simple': False},
        ('exclude_flagged', {
            'default': 'False',
            'description': "Exclude all flagged spectra as possible refit solutions. [bool]",
            'simple': False}),
        ('rchi2_limit', {
            'default': None,
            'description': "maximium value for the reduced chi-squared above which the fit gets flagged [float]",
            'simple': False}),
        ('rchi2_limit_refit', {
            'default': 'None',
            'description': "Defaults to 'rchi2_limit' if not specified. [float]",
            'simple': False}),
        ('max_diff_comps', {
            'default': '1',
            'description': "Maximum allowed difference in the number of fitted components compared to weighted median of immediate neighbors [int]",
            'simple': True}),
        ('max_jump_comps', {
            'default': '2',
            'description': "Maximum allowed difference in the number of fitted components between individual neighboring spectra [int]",
            'simple': True}),
        ('n_max_jump_comps', {
            'default': '1',
            'description': "Maximum number of allowed 'max_jump_comps' occurrences for a single spectrum. [int]",
            'simple': True}),
        ('max_refitting_iteration', {
            'default': '30',
            'description': "Maximum number for refitting iterations. [int]",
            'simple': False}),
        ('flag_blended', {
            'default': 'True',
            'description': "Flag spectra with blended fit components. [bool]",
            'simple': False}),
        ('flag_neg_res_peak', {
            'default': 'True',
            'description': "Flag spectra with negative residual features. [bool]",
            'simple': False}),
        ('flag_rchi2', {
            'default': 'False',
            'description': "Flag spectra with high reduced chi-square values. [bool]",
            'simple': False}),
        ('flag_residual', {
            'default': 'True',
            'description': "Flag spectra with non-Gaussian distributed residuals. [bool]",
            'simple': False}),
        ('flag_broad', {
            'default': 'True',
            'description': "Flag spectra with broad fit components. [bool]",
            'simple': False}),
        ('flag_ncomps', {
            'default': 'True',
            'description': "Flag spectra with number of fit components incompatible with neighbors. [bool]",
            'simple': False}),

        ('mean_separation', {
            'default': '2.',
            'description': "Maximum difference in offset positions of fit components for grouping. [float]",
            'simple': True}),
        ('fwhm_separation', {
            'default': '4.',
            'description': "Maximum difference in FWHM values of fit components for grouping. [float]",
            'simple': True}),
        ('fwhm_factor_refit', {
            'default': 'None',
            'description': "Defaults to 'fwhm_factor' if not specified. [float]",
            'simple': False}),
        ('broad_neighbor_fraction', {
            'default': '0.5',
            'description': "Spectra get flagged as containing broad components if the FWHM value of one of their fit components exceeds the FWHM values of all fit components for this fraction of neighbors [float]",
            'simple': False}),
        ('min_weight', {
            'default': '0.5',
            'description': "Minimum weight threshold for phase 2 of spatially coherent refitting. [float]",
            'simple': True}),
        ('weight_factor', {
            'default': '2',
            'description': "Factor that determines the weight given to neighboring spectra located at a distance of 1 and 2 pixels. [int/float]",
            'simple': False}),
        ('only_print_flags', {
            'default': 'False',
            'description': "Only print flags in terminal without refitting. [bool]",
            'simple': False})
    ]
    dct_spatial_fitting = collections.OrderedDict(spatial_fitting)

    config_file = []

    config_file.append('[DEFAULT]')
    config_file = append_keywords(config_file, dct_default,
                                  all_keywords=all_keywords,
                                  description=description)

    config_file.append('\n\n[training]')
    config_file = append_keywords(config_file, dct_training,
                                  all_keywords=all_keywords,
                                  description=description)

    config_file.append('\n\n[preparation]')
    config_file = append_keywords(config_file, dct_preparation,
                                  all_keywords=all_keywords,
                                  description=description)

    config_file.append('\n\n[decomposition]')
    config_file = append_keywords(config_file, dct_decomposition,
                                  all_keywords=all_keywords,
                                  description=description)

    config_file.append('\n\n[spatial fitting]')
    config_file = append_keywords(config_file, dct_spatial_fitting,
                                  all_keywords=all_keywords,
                                  description=description)

    if not output_directory:
        output_directory = os.getcwd()

    with open(os.path.join(output_directory, filename), 'w') as file:
        for line in config_file:
            file.write(line)
        save_file(filename, output_directory)


def get_values_from_config_file(self, config_file, config_key='DEFAULT'):
    """Read in values from a GaussPy+ configuration file.

    Parameters
    ----------
    config_file : str
        Filepath to configuration file of GaussPy+.
    config_key : str
        Section of GaussPy+ configuration file, whose parameters should be read in addition to 'DEFAULT'.

    """
    config = configparser.ConfigParser()
    config.read(config_file)

    for key, value in config[config_key].items():
        try:
            setattr(self, key, ast.literal_eval(value))
        except ValueError:
            if key == 'vel_unit':
                value = u.Unit(value)
                setattr(self, key, value)
            else:
                raise Exception('Could not parse parameter {} from config file'.format(key))
