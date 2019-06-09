Getting started
===============

You can find an example decomposition run with ``GaussPy+`` in the ``example``
directory. All individual scripts can be run via the Jupyter notebook
``Tutorial_example-GRS.ipynb``.

The Jupyter notebook ``Tutorial_decompose_single_spectrum.ipynb`` illustrates the functionality of ``GaussPy+`` and allows users to play around with the different parameter settings to determine their effects on the decomposition.

Some advice for decomposition runs with ``GaussPy+``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We recommend to first test the ``GaussPy+`` settings on a small subsection of the dataset and check whether the chosen parameter values lead to good fitting results. ``GaussPy+`` includes many helper functions that allow users to easily create a subcube of the dataset to use for the test run.

We tested the default settings of ``GaussPy+`` on different spectral cubes of CO isotopologues, which yielded good decomposition results for all tested datasets. However, if you find that the fitting results are insufficient or stumble upon problems we recommend that you first try the following steps:


*
  For large datasets, we recommend to create smaller subcubes, on which to perform the ``GaussPy+`` decomposition. Running the decomposition directly on the full dataset will likely become too memory-intensive. We recommend a maximum individual size of about 1e5 spectra per subcube, with a maximum of about 1e8 voxels. ``GaussPy+`` includes helper functions to split the datasets into individual subcubes (\ ``gausspyplus.utils.make_subcube``\ , ``gausspyplus.utils.get_list_slice_params``\ ) and combine the results again after the decomposition.

*
  If you find that ``GaussPy+`` fits too many noise peaks, we recommend setting either the ``significance`` or ``snr`` parameters to higher values. In contrast, if you find that many low-intensity and/or narrow peaks are not fit at all, we recommend setting one or both of these parameters to lower values. ``GaussPy+`` also includes helper functions to smooth the spectral cube spatially (\ ``gausspyplus.utils.spatial_smoothing``\ ) and/or spectrally (\ ``gausspyplus.utils.spectral_smoothing``\ ).

*
  It might be necessary to modify the FITS header of the dataset, so that it is compatible with ``GaussPy+``. For example, the projection has to be supported by ``astropy.wcs`` and the ``NAXIS3`` axis of the FITS cube needs to be the spectral axis. ``GaussPy+`` includes helper functions that will try to prepare the FITS header accordingly, and also allow to swap the axes of the datacube.

*
  For phase 1 of the spatially coherent refitting, fit solutions are flagged based on user-defined criteria and ``GaussPy+`` will try to refit these flagged decomposition results by using neighboring unflagged fit solutions. In the default settings, one of the flagging criteria subjects the normalised residuals to normality tests to check whether the data points of the residual are normally distributed. This criterion might lead to a large number of spectra being flagged, which can lead to time-consuming refit attempts. If these refitting attempts should become prohibitive, we recommend to either set the ``min_pvalue`` parameter to lower values or set the ``refit_residual`` parameter to ``False``.

*
  If you run ``GaussPy+`` on HI datasets we recommend to set the ``refit_broad`` and ``refit_blended`` parameters to ``False``.
