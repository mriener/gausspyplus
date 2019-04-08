# @Author: riener
# @Date:   2019-04-01T20:20:22+02:00
# @Filename: training_set--grs.py
# @Last modified by:   riener
# @Last modified time: 2019-04-08T10:28:25+02:00


import os

from gausspyplus.training_set import GaussPyTrainingSet
from gausspyplus.plotting import plot_spectra

#  Initialize the 'GaussPyTrainingSet' class and read in the parameter settings from 'gausspy+.ini'.
training = GaussPyTrainingSet(config_file='gausspy+.ini')

#  The following lines will override the corresponding parameter settings defined in 'gausspy+.ini'.

#  Path to the FITS cube.
training.path_to_file = os.path.join(
    '..', 'gausspyplus', 'data', 'grs-test_field.fits')
#  Directory to which all files produced by GaussPy+ will get saved.
training.dirpath_gpy = 'decomposition_grs'
#  Number of spectra included in the training set. We recommend to have at least 250 spectra for a good training set.
training.n_spectra = 100
#  (Optional) The initial seed that is used to create pseudorandom numbers. Change this value in case the spectra chosen for the training set are not ideal.
training.random_seed = 111
#  (Optional) We set the upper limit for the reduced chi-square value to a lower number to only include good fits in the training sample
training.rchi2_limit = 1.2
#  (Optional) This will enforce a maximum upper limit for the FWHM value of fitted Gaussian components, in this case 50 channels. We recommended to use this upper limit for the FWHM only for the creation of the training set.
training.max_fwhm = 50.
#  (Optional) Here we specify the filename for the resulting pickled dictionary file. If 'filename_out' is not supplied it will be automatically generated.
training.filename_out = \
    'grs-test_field-training_set_{}_spectra.pickle'.format(training.n_spectra)

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
             n_spectra=20  # Plot 20 random spectra of the training set.
             )
