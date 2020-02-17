# @Author: riener
# @Date:   2019-03-03T20:27:37+01:00
# @Filename: config_file.py
# @Last modified by:   riener
# @Last modified time: 2019-04-08T10:08:05+02:00

import ast
import configparser
import collections
import os
import textwrap

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
        ('max_ncomps', {
            'default': 'None',
            'description': "maximum number of allowed fit components per spectrum. Use with caution. [int]",
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
        ('use_all_neighors', {
            'default': 'False',
            'description': "Use flagged neighbors as refit solutions in case the refit was not possible with fit solutions from unflagged neighbors. [bool]",
            'simple': True}),
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


def default_file_structure(output_directory='', suffix=''):
    _config_ = str(
        """
        dirpath_data = ''
        dirpath_gpy = ''
        filename = ''
        """
    )

    training_set = str(
        """
        import os

        from _config_ import dirpath_data, dirpath_gpy, filename
        from gausspyplus.training_set import GaussPyTrainingSet
        from gausspyplus.plotting import plot_spectra

        #  Initialize the 'GaussPyTrainingSet' class and read in the parameter settings from 'gausspy+.ini'.
        training = GaussPyTrainingSet(config_file='gausspy+.ini')

        #  The following lines will override the corresponding parameter settings defined in 'gausspy+.ini'.

        #  Path to the FITS cube.
        training.path_to_file = os.path.join(dirpath_data, filename + '.fits')
        #  Directory to which all files produced by GaussPy+ will get saved.
        training.dirpath_gpy = dirpath_gpy
        #  Number of spectra included in the training set. We recommend to have at least 250 spectra for a good training set.
        training.n_spectra = 100
        #  (Optional) The initial seed that is used to create pseudorandom numbers. Change this value in case the spectra chosen for the training set are not ideal.
        training.random_seed = 111
        #  (Optional) We set the upper limit for the reduced chi-square value to a lower number to only include good fits in the training sample
        training.rchi2_limit = 1.2
        #  (Optional) This will enforce a maximum upper limit for the FWHM value of fitted Gaussian components, in this case 50 channels. We recommended to use this upper limit for the FWHM only for the creation of the training set.
        training.max_fwhm = 50.

        training.decompose_spectra()  # Create the training set.

        #  (Optional) Plot the fitting results of the training set.

        #  Filepath to pickled dictionary of the training set.
        path_to_training_set = os.path.join(
            training.dirpath_gpy, 'gpy_training', training.filename_out)
        #  Directory in which the plots are saved.
        path_to_plots = os.path.join(training.dirpath_gpy, 'gpy_training')
        plot_spectra(path_to_training_set,
                     path_to_plots=path_to_plots,
                     training_set=True,
                     n_spectra=100  # Plot 100 random spectra of the training set.
                     )
        """
    )

    training = str(
        """
        import os

        from _config_ import dirpath_gpy, filename
        from gausspyplus.training import GaussPyTraining

        nspectra = 250

        #  Initialize the 'GaussPyTraining' class and read in the parameter settings from 'gausspy+.ini'.
        train = GaussPyTraining(config_file='gausspy+.ini')

        #  The following lines will override the corresponding parameter settings defined in 'gausspy+.ini'.

        #  Directory in which all files produced by GaussPy+ are saved.
        train.dirpath_gpy = dirpath_gpy
        #  Filepath to the training set.
        train.path_to_training_set = os.path.join(
            train.dirpath_gpy, 'gpy_training',
            '{}-training_set_{}_spectra.pickle'.format(filename, nspectra)
        #  We select the two-phase-decomposition that uses two smoothing parameters.
        train.two_phase_decomposition = True
        #  Initial value for the first smoothing parameter.
        train.alpha1_initial = 2.
        #  Initial value for the second smoothing parameter.
        train.alpha2_initial = 6.
        #  Start the training.
        train.training()
        """
    )

    prepare = str(
        """
        import os

        from _config_ import dirpath_data, dirpath_gpy, filename
        from gausspyplus.prepare import GaussPyPrepare
        from gausspyplus.plotting import plot_spectra

        suffix = ''

        #  Initialize the 'GaussPyPrepare' class and read in the parameter settings from 'gausspy+.ini'.
        prepare = GaussPyPrepare(config_file='gausspy+.ini')

        #  The following lines will override the corresponding parameter settings defined in 'gausspy+.ini'.

        #  Path to the FITS cube.
        prepare.path_to_file = os.path.join(
            dirpath_data, filename + suffix + '.fits')
        #  Directory in which all files produced by GaussPy+ are saved.
        prepare.dirpath_gpy = dirpath_gpy
        #  Number of CPUs used in parallel processing.
        prepare.use_ncpus = 15
        #  Prepare the data cube for the decomposition
        prepare.prepare_cube()
        """
    )

    decompose = str(
        """
        import os

        from _config_ import dirpath_gpy, filename
        from gausspyplus.decompose import GaussPyDecompose

        #  Initialize the 'GaussPyDecompose' class and read in the parameter settings from 'gausspy+.ini'.
        decompose = GaussPyDecompose(config_file='gausspy+.ini')

        #  The following lines will override the corresponding parameter settings defined in 'gausspy+.ini'.

        #  Filepath to pickled dictionary of the prepared data.
        decompose.path_to_pickle_file = os.path.join(
            dirpath_gpy, 'gpy_prepared', filename + '.pickle')
        #  First smoothing parameter
        decompose.alpha1 = None
        #  Second smoothing parameter
        decompose.alpha2 = None
        #  Suffix for the filename of the pickled dictionary with the decomposition results.
        decompose.suffix = '_g+'
        #  Start the decomposition.
        decompose.decompose()
        """
    )

    spatial_refitting_p1 = str(
        """
        import os

        from _config_ import dirpath_gpy, filename
        from gausspyplus.decompose import GaussPyDecompose
        from gausspyplus.spatial_fitting import SpatialFitting

        #  Initialize the 'SpatialFitting' class and read in the parameter settings from 'gausspy+.ini'.
        sp = SpatialFitting(config_file='gausspy+.ini')

        #  The following lines will override the corresponding parameter settings defined in 'gausspy+.ini'.

        #  filepath to the pickled dictionary of the prepared data
        sp.path_to_pickle_file = os.path.join(
            dirpath_gpy, 'gpy_prepared', filename + '.pickle')
        #  Filepath to the pickled dictionary of the decomposition results
        sp.path_to_decomp_file = os.path.join(
            dirpath_gpy, 'gpy_decomposed', filename + '_g+_fit_fin.pickle')
        #  Try to refit blended fit components
        sp.refit_blended = True
        #  Try to refit spectra with negative residual features
        sp.refit_neg_res_peak = True
        #  Try to refit broad fit components
        sp.refit_broad = True
        #  Flag spectra with non-Gaussian distributed residuals
        sp.flag_residual = True
        #  Do not try to refit spectra with non-Gaussian distributed residuals
        sp.refit_residual = False
        #  Try to refit spectra for which the number of fit components is incompatible with its direct neighbors
        sp.refit_ncomps = True
        #  We set the maximum allowed difference in the number of fitted components compared to the weighted median of all immediate neighbors to 1
        sp.max_diff_comps = 1
        # We set the maximum allowed difference in the number of fitted components between individual neighboring spectra to 2
        sp.max_jump_comps = 2
        # We will flag and try to refit all spectra which show jumps in the number of components of more than 2 to at least two direct neighbors
        sp.n_max_jump_comps = 1
        # Maximum difference in offset positions of fit components for grouping.
        sp.mean_separation = 2.
        # Maximum difference in FWHM values of fit components for grouping.
        sp.fwhm_separation = 4.
        # Use flagged neighbors as refit solutions in case the refit was not possible with fit solutions from unflagged neighbors.
        sp.use_all_neighors = True

        #  Start phase 1 of the spatially coherent refitting
        sp.spatial_fitting()
        """
    )

    spatial_refitting_p2 = str(
        """
        import os

        from _config_ import dirpath_gpy, filename
        from gausspyplus.decompose import GaussPyDecompose
        from gausspyplus.spatial_fitting import SpatialFitting

        #  Initialize the 'SpatialFitting' class and read in the parameter settings from 'gausspy+.ini'.
        sp = SpatialFitting(config_file='gausspy+.ini')

        #  The following lines will override the corresponding parameter settings defined in 'gausspy+.ini'.

        #  filepath to the pickled dictionary of the prepared data
        sp.path_to_pickle_file = os.path.join(
            dirpath_gpy, 'gpy_prepared', filename + '.pickle')
        #  Filepath to the pickled dictionary of the decomposition results
        sp.path_to_decomp_file = os.path.join(
            dirpath_gpy, 'gpy_decomposed', filename + '_g+_fit_fin_sf-p1.pickle')
        #  Try to refit blended fit components
        sp.refit_blended = True
        #  Try to refit spectra with negative residual features
        sp.refit_neg_res_peak = True
        #  Try to refit broad fit components
        sp.refit_broad = True
        #  Flag spectra with non-Gaussian distributed residuals
        sp.flag_residual = True
        #  Do not try to refit spectra with non-Gaussian distributed residuals
        sp.refit_residual = False
        #  Try to refit spectra for which the number of fit components is incompatible with its direct neighbors
        sp.refit_ncomps = True
        #  We set the maximum allowed difference in the number of fitted components compared to the weighted median of all immediate neighbors to 1
        sp.max_diff_comps = 1
        # We set the maximum allowed difference in the number of fitted components between individual neighboring spectra to 2
        sp.max_jump_comps = 2
        # We will flag and try to refit all spectra which show jumps in the number of components of more than 2 to at least two direct neighbors
        sp.n_max_jump_comps = 1
        # Maximum difference in offset positions of fit components for grouping.
        sp.mean_separation = 2.
        # Maximum difference in FWHM values of fit components for grouping.
        sp.fwhm_separation = 4.
        # Use flagged neighbors as refit solutions in case the refit was not possible with fit solutions from unflagged neighbors.
        sp.use_all_neighors = True
        #  Minimum required weight for neighboring features; for the default settings this would require that either the two immediate horizontal or vertical neighbors show a common feature or one of the immediate horizontal or vertical neighbors in addition to the two outermost neighbors in the same direction
        sp.min_weight = 0.6

        #  Start phase 2 of the spatially coherent refitting
        sp.spatial_fitting(continuity=True)
        """
    )

    finalize = str(
        """
        import os

        from _config_ import dirpath_gpy, filename
        from gausspyplus.finalize import Finalize

        suffix = '_g+_fit_fin_sf-p2'

        fin = Finalize(config_file='gausspy+.ini')
        fin.path_to_pickle_file = os.path.join(
            dirpath_gpy, 'gpy_prepared', '{}.pickle'.format(filename))
        fin.path_to_decomp_file = os.path.join(
            dirpath_gpy, 'gpy_decomposed', filename + suffix + '.pickle')
        fin.dirpath_table = os.path.join(dirpath_gpy, 'gpy_tables')
        fin.dct_params = {'mean_separation': 4., '_w_start': 2/3}

        fin.produce_noise_map(save=True)
        fin.produce_rchi2_map(save=True)
        fin.produce_component_map(save=True)
        fin.finalize_dct()
        fin.make_table()
        """
    )

    file_contents = [
        _config_,
        training_set,
        training,
        prepare,
        decompose,
        spatial_refitting_p1,
        spatial_refitting_p2,
        finalize
    ]

    filenames = [
        '_config_',
        'training_set',
        'training',
        'prepare',
        'decompose',
        'spatial_refitting_p1',
        'spatial_refitting_p2',
        'finalize'
    ]

    if not output_directory:
        output_directory = os.getcwd()

    for file_content, filename in zip(file_contents, filenames):
        if filename == '_config_':
            filename += '.py'
        elif suffix:
            filename += '--{}.py'.format(suffix)
        else:
            filename += '.py'

        with open(os.path.join(output_directory, filename), 'w') as file:
            output = textwrap.dedent(file_content)
            file.write(output[1:])  # to remove first empty line
            save_file(filename, output_directory)
