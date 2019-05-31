# @Author: riener
# @Date:   2019-04-02T17:42:46+02:00
# @Filename: decompose--grs.py
# @Last modified by:   riener
# @Last modified time: 31-05-2019


import os

from gausspyplus.decompose import GaussPyDecompose
from gausspyplus.plotting import plot_spectra

#  Initialize the 'GaussPyDecompose' class and read in the parameter settings from 'gausspy+.ini'.
decompose = GaussPyDecompose(config_file='gausspy+.ini')

#  The following lines will override the corresponding parameter settings defined in 'gausspy+.ini'.

#  Filepath to pickled dictionary of the prepared data.
decompose.path_to_pickle_file = os.path.join(
    'decomposition_grs', 'gpy_prepared', 'grs-test_field.pickle')
#  First smoothing parameter
decompose.alpha1 = 2.58
#  Second smoothing parameter
decompose.alpha2 = 5.14
#  Suffix for the filename of the pickled dictionary with the decomposition results.
decompose.suffix = '_g+'
#  Start the decomposition.
decompose.decompose()

#  (Optional) Produce a FITS image showing the number of fitted components
decompose.produce_component_map()
#  (Optional) Produce a FITS image showing the reduced chi-square values
decompose.produce_rchi2_map()

#  (Optional) Plot some of the spectra and the decomposition results

#  Filepath to pickled dictionary of the prepared data.
path_to_pickled_file = decompose.path_to_pickle_file
#  Filepath to pickled dictionary with the decomposition results
path_to_decomp_pickle = os.path.join(
    'decomposition_grs', 'gpy_decomposed', 'grs-test_field_g+_fit_fin.pickle')
#  Directory in which the plots are saved.
path_to_plots = os.path.join(
    'decomposition_grs', 'gpy_plots')
#  Here we select a subregion of the data cube, whose spectra we want to plot.
pixel_range = {'x': [30, 34], 'y': [25, 29]}
plot_spectra(path_to_pickled_file, path_to_plots=path_to_plots,
             path_to_decomp_pickle=path_to_decomp_pickle,
             signal_ranges=True, pixel_range=pixel_range)
