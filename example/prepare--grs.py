# @Author: riener
# @Date:   2019-04-02T17:23:06+02:00
# @Filename: prepare--grs.py
# @Last modified by:   riener
# @Last modified time: 2019-04-08T10:33:45+02:00


import os

from gausspyplus.prepare import GaussPyPrepare
from gausspyplus.plotting import plot_spectra

#  Initialize the 'GaussPyPrepare' class and read in the parameter settings from 'gausspy+.ini'.
prepare = GaussPyPrepare(config_file='gausspy+.ini')

#  The following lines will override the corresponding parameter settings defined in 'gausspy+.ini'.

#  Path to the FITS cube.
prepare.path_to_file = os.path.join(
    '..', 'gausspyplus', 'data', 'grs-test_field.fits')
#  Directory in which all files produced by GaussPy+ are saved.
prepare.dirpath_gpy = 'decomposition_grs'
#  Prepare the data cube for the decomposition
prepare.prepare_cube()
#  (Optional) Produce a FITS image with the estimated root-mean-square values
prepare.produce_noise_map()

#  (Optional) Plot some of the spectra and the estimated signal ranges

#  Filepath to pickled dictionary of the prepared data.
path_to_pickled_file = os.path.join(
    'decomposition_grs', 'gpy_prepared', 'grs-test_field.pickle')
#  Directory in which the plots are saved.
path_to_plots = os.path.join(
    'decomposition_grs', 'gpy_plots')
#  Here we select a subregion of the data cube, whose spectra we want to plot.
pixel_range = {'x': [30, 34], 'y': [25, 29]}
plot_spectra(path_to_pickled_file, path_to_plots=path_to_plots,
             signal_ranges=True, pixel_range=pixel_range)
