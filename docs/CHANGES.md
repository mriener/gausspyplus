## GaussPy+ Changelog

### 0.3.0 (2026-10-23)

* Established compatibility with Python 3.11 - 3.15 and the current scientific Python stack (numpy 2.5, scipy 1.18, astropy 8.0, lmfit 1.3.4, matplotlib 3.11.2, networkx 3.7). Python 3.11 or later is now required and the minimum versions of the required packages are enforced on installation.

* Added a pytest test suite and continuous integration on GitHub Actions. The full GaussPy+ pipeline (preparation, decomposition, spatially coherent refitting and finalization) is verified on the GRS test fields against recorded reference results; the continuous integration runs the test suite on Python 3.11 - 3.15 and additionally performs format, license and vulnerability checks.

* Fixed the file name suffix of the decomposition results in the finalization step (``gausspyplus.processing.finalize.Finalize``).

* Fixed the propagation of fit parameter bounds in the spatially coherent refitting. The bounds are now preserved when fit components are removed in the quality control, so refits can no longer return fit parameters outside the fitted bounds.

* Fit parameter uncertainties are now reported as NaN instead of 0 in cases where the estimation of the parameter uncertainties fails.

* Fixed a bug in the determination of the average noise levels (``calculate_average_rms_noise``) that could select the same noise samples multiple times.

* Fixed the flagging criterion for broad fit components, which previously depended on the exact equality of floating point values.

* Fixed an index wrap-around bug in ``count_ones_in_row`` (used in the convergence checks of the training).

* Fixed crashes in the parallel processing routine.

* Modernized the packaging: the package is now built with hatchling from PEP 621 project metadata and a uv lock file (``uv.lock``) is provided to reproduce the tested dependency versions.

* The scripts in the ``tutorials`` directory can now be executed from any directory.


### 0.2 (2020-05-19)

* Established compatibility with Python 3.8.2, Numpy 1.18.4, lmfit 1.0.1,  astropy 4.0.1, matplotlib 3.2.1.

* Added option that uses flagged neighbors in the spatial refitting routine.
If the 'use_all_neighors' parameter is set to True, flagged neighbors are used as refit solutions in case the refit was not possible with fit solutions from unflagged neighbors. See Appendix A.3 in Riener+ 2019b for more details.

* Added option to restrict spatial refitting routine to a subset of the data.
This is only meant for testing purposes, if users would like to check the effects of parameter settings in the spatially coherent refitting routines without running it on the entire data set (which can be time-consuming).
The subset of the data can be indicated with the 'pixel_range' keyword and has to be supplied as a dictionary. For example, 'pixel_range = {'y': [10, 20], 'x': [5, 10]}' restricts the spatial refitting to a subset of 50 spectra located within ``10 <= y < 20`` and ``5 <= x < 10``.

* Added the `finalize.py` module for producing tables of the final decomposition results. The following code gives an example on how to produce a table of the decomposition results from a dataset that was split into individual subcubes:

```python
import os
from gausspyplus.processing.finalize import Finalize

for subcube_nr in range(1, total_nr_of_subcubes + 1):
    filename = '{}{}'.format(ppv_cube_name, subcube_nr)

    fin = Finalize(config_file='gausspy+.ini')
    fin.path_to_pickle_file = os.path.join(
        dirpath_gpy, 'gpy_prepared', '{}.pickle'.format(filename))
    fin.path_to_decomp_file = os.path.join(
        dirpath_gpy, 'gpy_decomposed', '{}_g+_fit_fin_sf-p2.pickle'.format(filename))
    fin.dirpath_table = os.path.join(dirpath_gpy, 'gpy_tables')
    fin.dct_params = {'mean_separation': 4., '_w_start': 2 / 3}
    fin.subcube_nr = subcube_nr
    fin.finalize_dct()
    fin.make_table()
```

* Added try/except blocks to catch errors in the GaussPy decomposition.
If errors are introduced in the GaussPy decomposition step the index of the spectrum with the corresponding error is printed in the terminal. The spectrum causing the error is replaced with None in the fit results dictionary.

* Added safeguard to prevent eternal loop in ``get_signal_ranges``.

* Added flux preserving mode in ``gausspyplus.utils.spectral_cube_functions.spatial_smoothing`` if ``reproject=True``.

* Introduced ``max_ncomps`` parameter, which enforces a maximum number of fit components per spectrum.

* Added function to create a default file structure similar to the GRS test field scripts in the example directory.

* Removed HDF5 dependency.

* Many small improvements and bugfixes.


### 0.1 (2019-06-01)

* Initial release of gausspyplus.
