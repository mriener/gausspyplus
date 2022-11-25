import os
import sys
from pathlib import Path

ROOT = Path(os.path.realpath("__file__")).parents[1]
sys.path.append(str(ROOT))

from gausspyplus.decomposition.decompose import GaussPyDecompose
from gausspyplus.finalize import Finalize
from gausspyplus.plotting import plot_spectra


def main():
    #  Initialize the 'GaussPyDecompose' class and read in the parameter settings from 'gausspy+.ini'.
    decompose = GaussPyDecompose(config_file="gausspy+.ini")

    #  The following lines will override the corresponding parameter settings defined in 'gausspy+.ini'.

    #  Filepath to pickled dictionary of the prepared data.
    decompose.path_to_pickle_file = Path(
        "decomposition_grs", "gpy_prepared", "grs-test_field.pickle"
    )
    #  First smoothing parameter
    decompose.alpha1 = 2.58
    #  Second smoothing parameter
    decompose.alpha2 = 5.14
    #  Suffix for the filename of the pickled dictionary with the decomposition results.
    decompose.suffix = "_g+"
    #  Start the decomposition.
    decompose.decompose()

    #  (Optional) Produce FITS maps of the decomposition results

    #  Initialize the 'Finalize' class and read in the parameter settings from 'gausspy+.ini'.
    finalize = Finalize(config_file="gausspy+.ini")
    #  filepath to the pickled dictionary of the prepared data
    finalize.path_to_pickle_file = Path(
        "decomposition_grs", "gpy_prepared", "grs-test_field.pickle"
    )
    #  Filepath to the pickled dictionary of the decomposition results
    finalize.path_to_decomp_file = Path(
        "decomposition_grs", "gpy_decomposed", "grs-test_field_g+_fit_fin.pickle"
    )
    #  Produce a FITS image showing the number of fitted components
    finalize.produce_component_map()
    #  Produce a FITS image showing the reduced chi-square values
    finalize.produce_rchi2_map()

    #  (Optional) Plot some of the spectra and the decomposition results

    #  Filepath to pickled dictionary of the prepared data.
    path_to_pickled_file = decompose.path_to_pickle_file
    #  Filepath to pickled dictionary with the decomposition results
    path_to_decomp_pickle = Path(
        "decomposition_grs", "gpy_decomposed", "grs-test_field_g+_fit_fin.pickle"
    )
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
