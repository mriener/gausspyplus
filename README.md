<p align="center">
<img src="docs/images/gausspyplus_logo.png"  alt="" width = "300" />
</p>

# About
``GaussPy+`` is a fully automated Gaussian decomposition package for emission
line spectra.

Manuel Riener, Jouni Kainulainen, Jonathan D. Henshaw, Jan H. Orkisz,
Claire E. Murray, Henrik Beuther

For a detailed description about the package and results of tests performed on synthetic spectra and a test field from the Galactic Ring Survey (Jackson et al. 2006) please see:

[Riener et al. 2019](https://arxiv.org/abs/1906.10506)

``GaussPy+`` is based on ``GaussPy``: A python tool for implementing the
Autonomous Gaussian Decomposition algorithm.

Robert R. Lindner, Carlos Vera-Ciro, Claire E. Murray, Elijah Bernstein-Cooper

For a description about the Autonomous Gaussian Decomposition algorithm please see:

[Lindner et al. 2015](https://arxiv.org/abs/1409.2840)

### Documentation

The ``GaussPy+`` documentation can be found on [ReadTheDocs](https://gausspyplus.readthedocs.io).

# Installation

### Dependencies

You will need the following packages to run ``GaussPy+``. We list the version of
each package which we know to be compatible with ``GaussPy+``.

* [python 3.5](https://www.python.org/)

* [numpy (v1.14.2)](http://www.numpy.org/)

* [scipy (v0.19.0)](http://www.scipy.org/)

* [lmfit (v0.9.5)](https://lmfit.github.io/lmfit-py/intro.html)

* [h5py (v2.8)](http://www.h5py.org/)

* [astropy (v3.0.4)](http://www.astropy.org/)

* [networkx (v2.0)](https://networkx.github.io/)

* [tqdm (v4.19.4)](https://tqdm.github.io/)


If you do not already have Python 3.5, you can install the [Anaconda Scientific
Python distribution](https://store.continuum.io/cshop/anaconda/), which comes
pre-loaded with numpy, scipy, and h5py.

### Optional Dependencies

If you wish to use ``GaussPy+``'s plotting capabilities you will need to install
matplotlib:

* [matplotlib (v2.2.2)](http://matplotlib.org/)

If you wish to use optimization with Fortran code you will need

* [GNU Scientific Library (GSL)](http://www.gnu.org/software/gsl/)


### Download ``GaussPy+``

Download GaussPy+ using git `$ git clone https://github.com/mriener/gausspyplus`


### Installing Dependencies on Linux

You will need several libraries which the `GSL`, `h5py`, and `scipy` libraries
depend on. Install these required packages with:

```bash
sudo apt-get install libblas-dev liblapack-dev gfortran libgsl0-dev libhdf5-serial-dev
sudo apt-get install hdf5-tools
```

Install pip for easy installation of python packages:

```bash
sudo apt-get install python-pip
```

Then install the required python packages:

```bash
sudo pip install scipy numpy h5py lmfit astropy networkx tqdm
```

Install the optional dependencies for plotting and optimization:

```bash
sudo pip install matplotlib
sudo apt-get install libgsl0-dev
```

### Installing Dependencies on OSX

Installation on OSX can be done easily with homebrew. Install pip for easy
installation of python packages:

```bash
sudo easy_install pip
```

Then install the required python packages:

```bash
sudo pip install numpy scipy h5py lmfit astropy networkx tqdm
```

Install the optional dependencies for plotting and optimization:

```bash
sudo pip install matplotlib
sudo brew install gsl
```

### Installing ``GaussPy+``

To install make sure that all dependences are already installed and properly
linked to python --python has to be able to load them--. Then cd to the local
directory containing ``GaussPy+`` and install via

```bash
python setup.py install
```

If you don't have root access and/or wish a local installation of
``GaussPy+`` then use

```bash
python setup.py install --user
```

# Getting started

You can find an example decomposition run with ``GaussPy+`` in the `example`
directory. All individual scripts can be run via the Jupyter notebook
`Tutorial_example-GRS.ipynb`.

The Jupyter notebook `Tutorial_decompose_single_spectrum.ipynb` illustrates the functionality of ``GaussPy+`` and allows users to play around with the different parameter settings to determine their effects on the decomposition.

### Some advice for decomposition runs with ``GaussPy+``

We recommend to first test the ``GaussPy+`` settings on a small subsection of the dataset and check whether the chosen parameter values lead to good fitting results. ``GaussPy+`` includes many helper functions that allow users to easily create a subcube of the dataset to use for the test run.

We tested the default settings of ``GaussPy+`` on different spectral cubes of CO isotopologues, which yielded good decomposition results for all tested datasets. However, if you find that the fitting results are insufficient or stumble upon problems we recommend that you first try the following steps:

* For large datasets, we recommend to create smaller subcubes, on which to perform the ``GaussPy+`` decomposition. Running the decomposition directly on the full dataset will likely become too memory-intensive. We recommend a maximum individual size of about 1e5 spectra per subcube, with a maximum of about 1e8 voxels. ``GaussPy+`` includes helper functions to split the datasets into individual subcubes (``gausspyplus.utils.make_subcube``, ``gausspyplus.utils.get_list_slice_params``) and combine the results again after the decomposition.

* If you find that ``GaussPy+`` fits too many noise peaks, we recommend setting either the ``significance`` or ``snr`` parameters to higher values. In contrast, if you find that many low-intensity and/or narrow peaks are not fit at all, we recommend setting one or both of these parameters to lower values. See Sect. 3.2.1.2 and 3.2.1.3 in [Riener et al. 2019](https://arxiv.org/abs/1906.10506) for a description about the ``significance`` and ``snr`` parameters; also see App. C.3 for how changing one or both of these parameters can impact the decomposition results. ``GaussPy+`` also includes helper functions to smooth the spectral cube spatially (``gausspyplus.utils.spatial_smoothing``) and/or spectrally (``gausspyplus.utils.spectral_smoothing``).

* It might be necessary to modify the FITS header of the dataset, so that it is compatible with ``GaussPy+``. For example, the projection has to be supported by ``astropy.wcs`` and the ``NAXIS3`` axis of the FITS cube needs to be the spectral axis. ``GaussPy+`` includes helper functions that will try to prepare the FITS header accordingly, and also allow to swap the axes of the datacube.

* For phase 1 of the spatially coherent refitting, fit solutions are flagged based on user-defined criteria and ``GaussPy+`` will try to refit these flagged decomposition results by using neighboring unflagged fit solutions. In the default settings, one of the flagging criteria subjects the normalised residuals to normality tests to check whether the data points of the residual are normally distributed. This criterion might lead to a large number of spectra being flagged, which can lead to time-consuming refit attempts. If these refitting attempts should become prohibitive, we recommend to either set the ``min_pvalue`` parameter to lower values or set the ``refit_residual`` parameter to ``False``.

* If you run ``GaussPy+`` on HI datasets we recommend to set the ``refit_broad`` and ``refit_blended`` parameters to ``False`` (see Sect. 3.2.2.2 and 3.2.2.3 in [Riener et al. 2019](https://arxiv.org/abs/1906.10506)).

# Citing ``GaussPy+``
===============

If you make use of this package in a publication, please consider the following
acknowledgements:

```
@ARTICLE{Riener2019,
       author = {{Riener}, Manuel and {Kainulainen}, Jouni and {Henshaw}, Jonathan D. and
         {Orkisz}, Jan H. and {Murray}, Claire E. and {Beuther}, Henrik},
        title = "{GaussPy+: A fully automated Gaussian decomposition package for emission line spectra}",
      journal = {arXiv e-prints},
     keywords = {Astrophysics - Instrumentation and Methods for Astrophysics, Astrophysics - Astrophysics of Galaxies},
         year = "2019",
        month = "Jun",
          eid = {arXiv:1906.10506},
        pages = {arXiv:1906.10506},
archivePrefix = {arXiv},
       eprint = {1906.10506},
 primaryClass = {astro-ph.IM},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2019arXiv190610506R},
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

Please also consider acknowledgements to the required packages in your work.


# Feedback

We would love to get your feedback on ``GaussPy+``. If you should find that ``GaussPy+`` does not perform as intended for your dataset or if you should come across bugs or have suggestions for improvement, please get into contact with us or open a new Issue or Pull request. We are also happy to give support and advice on the decomposition.

# Contributing to ``GaussPy+``

To contribute to ``GaussPy+``, see [Contributing to GaussPy+](CONTRIBUTING.md)
