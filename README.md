
<p align="center">
<img src="docs/images/gausspyplus_logo.png"  alt="" width = "300" />
</p>

## About
``GaussPy+`` is a fully automated Gaussian decomposition package for emission line spectra. For a detailed description about the package and results of tests performed on synthetic spectra and a test field from the Galactic Ring Survey ([Jackson et al. 2006](https://ui.adsabs.harvard.edu/abs/2006ApJS..163..145J/abstract)) please see:

[Riener et al. 2019](https://ui.adsabs.harvard.edu/abs/2019A%26A...628A..78R/abstract)

``GaussPy+`` is based on ``GaussPy``: A python tool for implementing the Autonomous Gaussian Decomposition algorithm. For a description about the Autonomous Gaussian Decomposition algorithm please see:

[Lindner et al. 2015](https://ui.adsabs.harvard.edu/abs/2015AJ....149..138L/abstract)

For tips on how to get started with ``GaussPy+`` see the section [Getting started](#gettingstarted) further below and check the [Frequently asked questions](docs/FAQ.md).

### Version

The currently recommended version of GaussPy+ is v0.2 (stable version released on 2020-05-19). See the [GaussPy+ Changelog](docs/CHANGES.md) for an overview of the major changes and improvements introduced in this version.

New updates to the code are first tested and developed in the ``dev`` branch. Users cloning the ``dev`` branch should beware that these versions are not guaranteed to be stable.

## Installation

### Dependencies

You will need the following packages to run ``GaussPy+``. We list the version of each package which we know to be compatible with ``GaussPy+``.

* [python 3.5](https://www.python.org/)
* [astropy (v3.0.4)](http://www.astropy.org/)
* [lmfit (v0.9.5)](https://lmfit.github.io/lmfit-py/intro.html)
* [matplotlib (v2.2.2)](http://matplotlib.org/)
* [networkx (v2.0)](https://networkx.github.io/)
* [numpy (v1.14.2)](http://www.numpy.org/)
* [scipy (v0.19.0)](http://www.scipy.org/)
* [tqdm (v4.19.4)](https://tqdm.github.io/)

If you do not already have Python 3.5, you can install the [Anaconda Scientific Python distribution](https://store.continuum.io/cshop/anaconda/), which comes pre-loaded with numpy and scipy.

### Optional Dependencies

If you wish to use optimization with Fortran code you will need

* [GNU Scientific Library (GSL)](http://www.gnu.org/software/gsl/)

### Download GaussPy+

Download GaussPy+ using git `$ git clone https://github.com/mriener/gausspyplus`


### Installing Dependencies on Linux

You will need several libraries which the `GSL` and `scipy` libraries depend on. Install these required packages with:

```bash
sudo apt-get install libblas-dev liblapack-dev gfortran libgsl0-dev
```

Install pip for easy installation of python packages:

```bash
sudo apt-get install python-pip
```

Then install the required python packages:

```bash
sudo pip install astropy lmfit networkx numpy scipy tqdm
```

Install the optional dependencies for plotting and optimization:

```bash
sudo pip install matplotlib
sudo apt-get install libgsl0-dev
```

### Installing Dependencies on OSX

Installation on OSX can be done easily with homebrew. Install pip for easy installation of python packages:

```bash
sudo easy_install pip
```

Then install the required python packages:

```bash
sudo pip install numpy scipy lmfit astropy networkx tqdm
```

Install the optional dependencies for plotting and optimization:

```bash
sudo pip install matplotlib
sudo brew install gsl
```

### Installing GaussPy+

To install make sure that all dependences are already installed and properly linked to python --python has to be able to load them--. Then cd to the local directory containing ``GaussPy+`` and install via

```bash
python setup.py install
```

If you don't have root access and/or wish a local installation of
``GaussPy+`` then use

```bash
python setup.py install --user
```

<a id="gettingstarted"></a>
## Getting started

You can find an example decomposition run with ``GaussPy+`` in the `example` directory. All individual scripts can be run via the Jupyter notebook [Tutorial_example-GRS.ipynb](example/Tutorial_example-GRS.ipynb).

The Jupyter notebook [Tutorial_decompose_single_spectrum.ipynb](example/Tutorial_decompose_single_spectrum.ipynb) illustrates the functionality of ``GaussPy+`` and allows users to play around with the different parameter settings to determine their effects on the decomposition.

### Some advice for decomposition runs with GaussPy+

We recommend to first test the ``GaussPy+`` settings on a small subsection of the dataset and check whether the chosen parameter values lead to good fitting results. ``GaussPy+`` includes many helper functions that allow users to easily create a subcube of the dataset to use for the test run.

We tested the default settings of ``GaussPy+`` on different spectral cubes of CO isotopologues, which yielded good decomposition results for all tested datasets. However, if you find that the fitting results are insufficient or stumble upon problems we recommend that you first try the following steps:

* For large datasets, we recommend to create smaller subcubes, on which to perform the ``GaussPy+`` decomposition. Running the decomposition directly on the full dataset will likely become too memory-intensive. We recommend a maximum individual size of about 1e5 spectra per subcube, with a maximum of about 1e8 voxels. ``GaussPy+`` includes helper functions to split the datasets into individual subcubes (``gausspyplus.utils.make_subcube``, ``gausspyplus.utils.get_list_slice_params``) and combine the results again after the decomposition.

* If you find that ``GaussPy+`` fits too many noise peaks, we recommend setting either the ``significance`` or ``snr`` parameters to higher values. In contrast, if you find that many low-intensity and/or narrow peaks are not fit at all, we recommend setting one or both of these parameters to lower values. See Sect. 3.2.1.2 and 3.2.1.3 in [Riener et al. 2019](https://arxiv.org/abs/1906.10506) for a description about the ``significance`` and ``snr`` parameters; also see App. C.3 for how changing one or both of these parameters can impact the decomposition results. ``GaussPy+`` also includes helper functions to smooth the spectral cube spatially (``gausspyplus.utils.spatial_smoothing``) and/or spectrally (``gausspyplus.utils.spectral_smoothing``).

* It might be necessary to modify the FITS header of the dataset, so that it is compatible with ``GaussPy+``. For example, the projection has to be supported by ``astropy.wcs`` and the ``NAXIS3`` axis of the FITS cube needs to be the spectral axis. ``GaussPy+`` includes helper functions that will try to prepare the FITS header accordingly, and also allow to swap the axes of the data cube.

* For phase 1 of the spatially coherent refitting, fit solutions are flagged based on user-defined criteria and ``GaussPy+`` will try to refit these flagged decomposition results by using neighboring unflagged fit solutions. In the default settings, one of the flagging criteria subjects the normalised residuals to normality tests to check whether the data points of the residual are normally distributed. This criterion might lead to a large number of spectra being flagged, which can lead to time-consuming refit attempts. If these refitting attempts should become prohibitive, we recommend to either set the ``min_pvalue`` parameter to lower values or set the ``refit_residual`` parameter to ``False``.

* For the creation of the training set, by default fit solutions are only accepted if their reduced chi-squared values are lower than 1.5. If the decomposition routine to create the training set should take a very long time, we recommend to set the the ``rchi2_limit`` parameter to a higher value.

* In case of a small number of spectral channels (< 100), we recommend to reduce the ``min_channels`` parameter (default: 100) to a smaller number. It can also be beneficial to reduce the ``pad_channels`` parameter (default: 5) to a smaller number.

* If you run ``GaussPy+`` on HI datasets we recommend to set the ``refit_broad`` and ``refit_blended`` parameters to ``False`` (see Sect. 3.2.2.2 and 3.2.2.3 in [Riener et al. 2019](https://arxiv.org/abs/1906.10506)).

* By default, the first phase of the spatially coherent refitting will only consider unflagged neighboring fit solutions. In case all neighboring fit solutions get flagged or unflagged neighboring fit solutions do not yield an improvement in the fit, it can be beneficial to consider also flagged neighboring fit solutions for the refitting. If the ``use_all_neighors`` parameter (default: ``False``) is set to 'True' in the spatial refitting, flagged neighbors are used as refit solutions in case the refit was not possible with fit solutions from unflagged neighbors. See Appendix A.3 in Riener+ 2020a for more details.

* A maximum number of fit components per spectrum can now be enforced with the ``max_ncomps`` parameter (default value: ``None``). This can be useful in case the spectrum contains instrumental artefacts, for example amplified noise oscillations in interferometric observations. It can also be useful in cases where an upper limit or maximum expected number of fit components can be predicted well and the fit solutions of ``GaussPy+`` show clear signs of overfitting. If ``max_ncomps`` is set and the total number of fit components exceeds this limit, ``GaussPy+`` will iteratively remove Gaussian components with the lowest integrated area until the total number of fit components is equal to ``max_ncomps``. We recommend to use this parameter with caution as it might lead to problems in the fitting.

<a id="faq"></a>
## Frequently asked questions
See [FAQ](docs/FAQ.md).

## Citing GaussPy+

If you make use of this package in a publication, please consider the following acknowledgements:

```
@ARTICLE{2019A&A...628A..78R,
       author = {{Riener}, M. and {Kainulainen}, J. and {Henshaw}, J.~D. and
         {Orkisz}, J.~H. and {Murray}, C.~E. and {Beuther}, H.},
        title = "{GAUSSPY+: A fully automated Gaussian decomposition package for emission line spectra}",
      journal = {\aap},
     keywords = {methods: data analysis, radio lines: general, ISM: kinematics and dynamics, ISM: lines and bands, Astrophysics - Instrumentation and Methods for Astrophysics, Astrophysics - Astrophysics of Galaxies},
         year = "2019",
        month = "Aug",
       volume = {628},
          eid = {A78},
        pages = {A78},
          doi = {10.1051/0004-6361/201935519},
archivePrefix = {arXiv},
       eprint = {1906.10506},
 primaryClass = {astro-ph.IM},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2019A&A...628A..78R},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}

@ARTICLE{Lindner2015L,
       author = {{Lindner}, Robert R. and {Vera-Ciro}, Carlos and {Murray}, Claire E. and
         {Stanimirovi{\'c}}, Sne{\v{z}}ana and {Babler}, Brian and
         {Heiles}, Carl and {Hennebelle}, Patrick and {Goss}, W.~M. and
         {Dickey}, John},
        title = "{Autonomous Gaussian Decomposition}",
      journal = {\aj},
     keywords = {ISM: atoms, ISM: clouds, ISM: lines and bands, line: identification, methods: data analysis, techniques: spectroscopic, Astrophysics - Instrumentation and Methods for Astrophysics, Astrophysics - Astrophysics of Galaxies},
         year = "2015",
        month = "Apr",
       volume = {149},
       number = {4},
          eid = {138},
        pages = {138},
          doi = {10.1088/0004-6256/149/4/138},
archivePrefix = {arXiv},
       eprint = {1409.2840},
 primaryClass = {astro-ph.IM},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2015AJ....149..138L},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}

```
Citation courtesy of [ADS](https://ui.adsabs.harvard.edu/#).

Please also consider acknowledgements to the required packages in your work.


## Feedback

We would love to get your feedback on ``GaussPy+``. If you should find that ``GaussPy+`` does not perform as intended for your dataset or if you should come across bugs or have suggestions for improvement, please get into contact with us or open a new Issue or Pull request. We are also happy to give support and advice on the decomposition.

## Contributing to GaussPy+

To contribute to ``GaussPy+``, see [Contributing to GaussPy+](docs/CONTRIBUTING.md).
