import os
import sys
from pathlib import Path

ROOT = Path(os.path.realpath("__file__")).parents[1]
sys.path.append(str(ROOT))

from gausspyplus.finalize import Finalize
from gausspyplus.spatial_fitting import SpatialFitting
from gausspyplus.plotting.plotting import plot_spectra


def main():
    #  Initialize the 'SpatialFitting' class and read in the parameter settings from 'gausspy+.ini'.
    sp = SpatialFitting(config_file="gausspy+.ini")

    #  The following lines will override the corresponding parameter settings defined in 'gausspy+.ini'.

    #  filepath to the pickled dictionary of the prepared data
    sp.path_to_pickle_file = os.path.join(
        "decomposition_grs", "gpy_prepared", "grs-test_field.pickle"
    )
    #  Filepath to the pickled dictionary of the decomposition results
    sp.path_to_decomp_file = os.path.join(
        "decomposition_grs", "gpy_decomposed", "grs-test_field_g+_fit_fin.pickle"
    )
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
    sp.mean_separation = 2.0
    # Maximum difference in FWHM values of fit components for grouping.
    sp.fwhm_separation = 4.0

    #  Start phase 1 of the spatially coherent refitting
    sp.spatial_fitting()

    #  (Optional) Plot maps of the reduced chi-square values and the number of fitted components

    #  Initialize the 'Finalize' class and read in the parameter settings from 'gausspy+.ini'.
    finalize = Finalize(config_file="gausspy+.ini")
    #  Filepath to pickled dictionary of the prepared data.
    finalize.path_to_pickle_file = sp.path_to_pickle_file
    #  Filepath to the pickled dictionary with the decomposition results
    path_to_decomp_pickle = Path(
        "decomposition_grs", "gpy_decomposed", "grs-test_field_g+_fit_fin_sf-p1.pickle"
    )
    #  Load the decomposition results
    finalize.path_to_decomp_file = path_to_decomp_pickle
    #  Produce a FITS image showing the number of fitted components
    finalize.produce_component_map()
    #  Produce a FITS image showing the reduced chi-square values
    finalize.produce_rchi2_map()

    #  (Optional) Plot some of the spectra and the decomposition results

    #  Filepath to pickled dictionary of the prepared data.
    path_to_pickled_file = sp.path_to_pickle_file
    #  Directory in which the plots are saved.
    path_to_plots = Path("decomposition_grs", "gpy_plots")
    #  Here we select a subregion of the data cube, whose spectra we want to plot.
    pixel_range = {"x": [30, 34], "y": [25, 29]}
    plot_spectra(
        path_to_pickled_file,
        path_to_plots=path_to_plots,
        path_to_decomp_pickle=path_to_decomp_pickle,
        signal_ranges=True,
        pixel_range=pixel_range,
    )


if __name__ == "__main__":
    main()
