#### What is a good size for the training set?
A training set composed of 250 to 500 representative spectra of the data set should already give good results. It is always a good idea though, to check the fit results for the training set in detail, to verify that the data cube has been sampled well and the decomposition of the training set worked out well.

By default, the limiting value rchi2 parameter for the training set has a very low value (``rchi2_limit = 1.5``). This might lead to a bias, where only simple spectra that contain few non-blended features are included in the training set. This should not be too big of a problem, as the training step is mostly needed to find out how to best smooth the noise wiggles of the spectra for initial guesses for the fitting. However, if this should become a problem, we recommend to set the ``rchi2_limit`` parameter to higher values, to also include more complex spectra.

#### How do I know if my alpha values are good enough for the decomposition?
The best way to judge whether the alpha values determined from the training step make sense, is to verify their performance visually on representative spectra in the notebook [Tutorial_decompose_single_spectrum.ipynb](example/Tutorial_decompose_single_spectrum.ipynb).

Determining the best value for the alpha parameters is an essential key step for the original GaussPy algorithm.
However, ``GaussPy+`` contains many additional routines that should be able to mitigate non-ideal choices for the alpha parameters.
While the value of the alpha parameters is important, smaller deviations from the perfect value should not have too much of a negative effect on the ``GaussPy+`` decomposition (see also Appendix B.5 in [Riener et al. 2019](https://arxiv.org/abs/1906.10506)).

#### What are the best parameter settings for my dataset?
Unfortunately, there is no definitive answer to that. The ideal settings will likely vary depending on the characteristics of the used data set. We recommend to check the notebook [Tutorial_decompose_single_spectrum.ipynb](example/Tutorial_decompose_single_spectrum.ipynb), which allows to play around with ``GaussPy+`` settings and verify the changes directly on individual spectra.
For identifying best parameters for the spatially coherent refitting stages, we recommend to test and check parameter variations on a small representative subcube of the data set first.
To verify the performance of correctness of the decomposition, users can also try to create their own synthetic spectra mimicking the expected spectral profiles in the observational data set.
Synthetic spectra?

#### How sensitive is the decomposition to the estimated noise?
A correct estimate of the rms-noise is essential for the decomposition. Multiple steps of the decomposition process depend on the noise estimation, so if an incorrect rms-noise value is assumed, the decomposition will very likely not yield good results. ``GaussPy+`` contains an automated method to determine the noise, which should yield good noise estimates. However, in certain instances (e.g. when the spectrum contains only a few channels without signal from which the noise can be sampled) the default values of ``GaussPy+`` for the noise estimation can lead to problems (see next question).

#### In the preparation step I get warning messages that an average noise has been assumed; is that normal?
If this warning pops up a couple of times it is not a problem, but if you should find that a couple of hundred or even more of the spectra are affected it could indicate issues with the noise estimation. Most likely, ``GaussPy+`` cannot identify enough signal-free channels in the spectrum to confidently determine the noise, and instead assumes an average rms-noise for the spectrum that it determined from sampling a number of random spectra (default: ``n_spectra_rms = 1000``).
Common reasons for this behavior are:
- the ``min_channels`` and ``pad_channels`` are set to too high values. Try to reduce these to much smaller values (e.g. 20 - 50 for ``min_channels`` and 1 - 2 for ``pad_channels``) and check whether that solves the problem.
- the ``mask_out_ranges`` parameter was used to mask out a significant fraction of the spectrum, thus not leaving enough channels for the noise estimation. In this case it is recommended to estimate the noise for the entire unmasked spectrum first and feed it into the preparation step as a noise map (see next question).

#### Can I feed in my own noise estimation instead of using the automated method?
Yes, it is possible to supply ``GaussPy+`` with a noise map in the FITS format (similar to what is produced with the ``prepare.produce_noise_map()`` command in the preparation step of ``GaussPy+``). This noise map can be supplied to the training set creation and preparation steps via the ``path_to_noise_map = '<filepath to your noise map>'`` command.

#### Is it better to change the S/N or the significance parameter?
Since these two parameters are not independent (the significance parameter corresponds essentially to the integrated area of the fitted Gaussian function), changing only one of the parameters might not lead to good results. See Sect. 3.2.1.2 and 3.2.1.3 in [Riener et al. 2019](https://arxiv.org/abs/1906.10506) for a description about the ``significance`` and ``snr`` parameters; also see App. C.3 for how changing one or both of these parameters can impact the decomposition results.
In case the data set is known to contain many low-amplitude signals with very broad linewidths, it could help to increase only the ``significance`` parameter but not the S/N limit.

#### Why does the training step take so much time?
The training is composed of two different steps (training set creation and the training itself), which are in general both quite time-consuming (see discussion in Appendix C.1 of [Riener et al. 2019](https://arxiv.org/abs/1906.10506)).
However, there are additional non-ideal settings that can slow down these steps even more significantly:
- In case of complex spectra (i.e. more than ~200 spectral channels and multiple blended spectral features), training set sizes of > 500 can lead to a very slow performance. We recommend individual training set sizes of 250 - 500 spectra. If users would like to have a more thorough training or do not think that the data set can be well sampled with only a single training set, we recommend to perform the training with multiple individual training sets of 250 - 500 spectra and assume the median or mean value of the resulting alpha parameter values for the decomposition.
- If it takes a long time to create the training set itself the assumed ``rchi2_limit`` value might be set too low (the default value is only 1.5). We recommend to increase this parameter to a higher value and check if the decomposition of the training set spectra still performed well.
- The initial values for ``alpha1`` and ``alpha2`` in the training step are not ideal and lead to many iterations in the gradient descent routine. After 20-30 iterations it should already become clear to which values ``alpha1`` and ``alpha2`` will converge, so these values can be chosen as the initial values for a new run (in case a different training set is put through the training step). Users should beware that the ``alpha1`` and ``alpha2`` parameters can sometimes be stuck in a *local* minimum, whereas we want the *global* minimum. Varying the initial values for ``alpha1`` and ``alpha2`` is thus in general a good idea to avoid this problem.

#### How can I verify the correct runtime performance of GaussPy+, and check warnings or error messages that appeared throughout the runtime?
``GaussPy+`` by default creates a log file, to which it saves all messages printed to the terminal. This log file is saved to the ``gpy_log`` directory that is created when running ``GaussPy+``.

#### Why does my decomposition take so long to run?
There can be multiple reasons responsible for prolonging the decomposition step:
- The size of the data cube might be too large, making the decomposition step prohibitively time-consuming. In such a case it helps to split the original cube into smaller individual subcubes (via the ``gausspyplus.utils.make_subcube`` and ``gausspyplus.utils.get_list_slice_params`` functions).
- Non-ideal flags or flagging criteria have been set in the Stage 2 and 3 of the decomposition (i.e. the spatially coherent refitting stages). We recommend setting the ``flag_residual`` parameter to ``False`` in case the noise distribution is expected to deviate from Gaussian noise. Similarly, the ``flag_rchi2`` parameter can lead to many unsuccessful and time-consuming refitting attempts.

#### Do I have to run both phases of the spatially coherent refitting steps?
It depends on the complexity of the spectra. If the spectral features are relatively simple, that means high S/N ratio and no strong degree of blendedness between features, the decomposition may already converge to satisfying results after Stage 1, so the first decomposition step. It then may not change the results significantly if the spatially coherent refitting is performed in addition to this step.
Likewise, Stage 3 (spatially coherent refitting phase 2) may not significantly improve the decomposition anymore, as Stage 2 (spatially coherent refitting phase 1) can already lead to very good spatial consistency.

#### What is the best strategy for finetuning the parameter settings?
To test the effects of changing essential and advanced parameter values, it is recommended to play around with the tutorial [Tutorial_decompose_single_spectrum.ipynb](example/Tutorial_decompose_single_spectrum.ipynb).
Later on, we recommend to focus on a small subcube of the full dataset (with a couple of 1e2 to 1e3 spectra max) to test the full decomposition routine and check how the results change if parameter values are varied.

#### What does the red-shaded area in the plots of the fitted spectra indicate?
This interval indicates the regions inferred by ``GaussPy+`` to contain signal peaks. Only spectral channels of these regions are used for the calculation of the reduced chi-square parameter. If no signal peaks were detected in the spectrum, the interval is by default extended to the whole spectrum.

#### My decomposition results show very high reduced chi-square values. Is that normal or indicative of problems?
``GaussPy+`` by default restricts the calculation of the reduced chi-square value to regions inferred to contain signal peaks (see previous question and Sect. 3.2.1 in [Riener et al. 2019](https://arxiv.org/abs/1906.10506)), which can lead to higher values compared to estimates of the reduced chi-square value performed over the full spectrum.
It is recommended to inspect the FITS maps of reduced chi-square values (``decompose.produce_rchi2_map()``) and check whether there exist regions of elevated reduced chi-square values. If so, users are advised to plot the fitted spectra and check whether all peaks have been fitted correctly and the noise properties are Gaussian.

#### GaussPy+ is fitting relatively simple high S/N spectra with many small individual components. What is the problem?
As a first step, it is recommended to set the ``snr`` and ``significance`` parameters to higher values (e.g. ``snr = 5`` and ``significance = 10``), and check whether this already alleviates the problem. Also tweaking some of the advanced parameters in the spatially coherent refitting phases might be able to improve the decomposition results.
This problem of overfitting can occur also for interferometric data sets, which might have strong non-Gaussian noise properties (e.g. noise amplifications at the spectral channels of signal peaks). These noise peaks might be misidentified as signal and tweaking the ``snr`` and ``significance`` parameters might not solve the problem fully. In such a case it can be beneficial to set the ``ncomps_max`` parameter, if an upper limit for expected signal peaks in the spectrum can be reliably estimated.  it can cal

#### I have distinctly different shapes of emission lines in my spectra. Can GaussPy+ handle this?
Yes, ``GaussPy+`` can fit both broad and narrow emission lines. You can test the performance of ``GaussPy+`` in the notebook [Tutorial_decompose_single_spectrum.ipynb](example/Tutorial_decompose_single_spectrum.ipynb) for an individual spectrum, which gives a visual representation of the decomposition step and shows how this separation of narrow and broad emission lines is performed.

#### Why does the preparation step take such a long time?
The most likely reason is that the number of cores (which can be specified with the ``n_cpus`` parameter) has been set to a value that is too high, which increases the overhead in the multiprocessing routine.
Reducing the number of used cores should lead to a speedup of the preparation step.

#### Is it beneficial to mask out all spectral channels of a spectrum that I am not interested in to speed up the decomposition?
Yes. Users can supply the ranges of spectral channels that should be masked out with the ``mask_out_ranges`` parameter. For example, for a spectrum with 200 spectral channels in total, ``mask_out_ranges = ([0, 100], [150, 200])`` would mask out the first 100 and the last 50 spectral channels. Note that for the decomposition itself the masked channels get replaced with artificial noise sampled from the estimated rms-noise value.

#### The decomposition is done, how can I retrieve the relevant physical parameters?
See the ``example_fits_first_component--grs.py`` in the ``example`` directory for an example of how to produce maps of the mean position, FWHM value, and amplitude value of only the first fit components in the spectra.

#### How can I produce a PPV cube from the decomposition results?
The finalize module contains a function to easily produce a model PPV cube of the decomposition results (``finalize.make_cube``).

#### Should I smooth the data before performing the decomposition?
Depending on the characteristics of the data set (e.g. only weak signal peaks with low S/N ratio), smoothing the data set spatially or spectrally before the decomposition can be beneficial. ``GaussPy+`` contains helper functions that perform the spatial and spectral smoothing of the data set (``gausspyplus.utils.spatial_smoothing``, ``gausspyplus.utils.spectral_smoothing``).

#### Can I run GaussPy+ on data sets of simulations?
Yes. However, ``GaussPy+`` cannot deal with noiseless spectra, as its functionality will break down in case two neighboring spectral channels have the exact same. For such cases of smooth spectra, it is necessary to overlay the spectra with noise sampled from a supplied rms-noise value. This can be achieved by setting the parameter ``simulation`` to ``True`` and supplying an average rms-noise value with the ``average_rms`` parameter.

#### Can GaussPy+ fit absorption lines?
In its current version, ``GaussPy+`` is not able to fit emission and absorption lines simultaneously. However, by inverting the spectra along the y-axis (i.e. artificially changing the appearance of absorption lines to emission lines and vice versa), ``GaussPy+`` can be used to model absorption lines, if they are not strongly blended with emission lines.

#### Can GaussPy+ fit hyperfine lines or non-Gaussian shapes?
In the current version of ``GaussPy+``, it is not possible to model emission lines that have a non-Gaussian shape (e.g. Voigt or Lorenz profiles). A decomposition of such data sets is still possible, but will still use a combination of Gaussian profiles to model the spectra. It depends on the data set in question, whether this might still be of use, for example to automatically identify the number and position of peaks that can then be fed into another semi-automated decomposition algorithm.

#### How does GaussPy+ deal with optically thick lines?
In its current version, ``GaussPy+`` is not able to deal with optically thick lines. Depending on the strength of the optical depth effects, an optically thick emission line will be either modelled with a single or two Gaussian profiles. If users suspect the presence of strong optical depth effects, we recommend to set the parameters ``refit_neg_res_peak = False`` and ``refit_blended = True``.

#### Does GaussPy+ automatically perform a continuum subtraction?
No. ``GaussPy+`` assumes that the baseline of the spectra is centered around zero, so any leftover continuum will thus negatively impact the decomposition.

#### How can I quickly check if GaussPy+ performed well on my dataset?
``GaussPy+`` includes many helper functions that produce maps and plots that can be used for quick verifications of the decomposition results. In the example decomposition (see notebook [Tutorial_example-GRS.ipynb](example/Tutorial_example-GRS.ipynb)) FITS maps of the rms-noise, number of fit components, reduced chi-square values and plots of fit results of an individual regions are produced, all of which can be helpful to identify problems in the decomposition. It is also useful to compare zeroth moment maps of the original data set with a data set recreated from the decomposition results. ``GaussPy+`` contains helper functions (``gausspyplus.utils.moment_masking``) that allow to produce Moment masked versions of the data sets.

#### Which of the flags should I set in the spatially coherent refitting routine?
This strongly depends on the characteristics of the data set. For typical data sets of CO emission line surveys that are not dominated by strong optical depth effects, the default settings should perform well. In the following we list some examples of how flagging criteria might be changed for other data sets, but note that the performance should always be verified by the user:  
- for HI data sets, we recommend to set ``flag_blended`` and ``flag_broad`` to ``False``
- for data sets dominated by strong optical depth effects, we recommend to set ``flag_neg_res_peak`` to ``False`` and ``flag_blended`` to ``True``
- if noise values are expected to deviate from Gaussian noise, we recommend to set ``flag_residual`` and ``flag_rchi2`` to ``False``

#### What should I do if I suspect that my noise does not show a Gaussian distribution?
In case the noise is not Gaussian distributed, we recommend to set the following four parameters to ``False``: ``refit_rchi2``, ``refit_resdiual``, ``flag_residual``, and ``flag_rchi2``.
In addition, we recommend to visually check and verify some of the fit results for individual spectra and the FITS map showing the rms-noise values.

#### Will the decomposition lead to artifacts at the boundaries if I split my cube into individual subcubes?
While splitting the data set into individual subcubes for the decomposition can lead to discontinuities at the edges of the split, this was not found to be an issue in the tests we performed. However, it is always a good idea to verify this yourself by looking for discontinuities on the map showing the number of fit components and by inspecting the fit solutions of spectra across such borders.

#### How do I know if the spectrum was fit correctly?
This strongly depends on the data set in question, but the following checks can be indicative of problems with the decomposition:
- Does the decomposition show many small fit components that do not really improve the decomposition? If yes, increasing the ``snr`` and ``significance`` parameters to higher values might be beneficial. Also using the ``max_ncomps`` could fix this issue, but users should be careful in only using this parameter if a clear upper limit for components can be established.
- Do single components with large FWHM values get fit over multiple separate peaks? If yes, check if ``refit_broad`` and ``flag_broad`` is set to ``True`` and try decreasing the ``significance`` and/or ``snr`` parameters. This problem can also be indicative of non-ideal ``alpha1`` and ``alpha2`` parameters or wrongly estimated rms-noise values.

#### Can I use the decomposition results from one data set (e.g. absorption spectra of a region) as input for the decomposition of another data set (e.g. emission spectra of the same region)?
This is not possible in the current version of ``GaussPy+``, but we are working on enabling this option in a future version.

#### My data set only has a couple of hundred or thousand spectra - is it still worthwhile to use GaussPy+?
``GaussPy+`` is fully automated and one of its biggest advantages is thus the reproducibility of its results. ``GaussPy+`` also has many options for finetuning parameters and a fast runtime, thus enabling quick comparisons of decompositions with different parameter settings for smaller data sets.

#### I still find spatial inconsistencies between neighboring spectra after the spatially coherent refitting step. What went wrong?
There are multiple reasons that can prevent perfect spatial consistency of the fit results.
``GaussPy+`` will *try* to refit spectra based on neighbors but it will *not* enforce this step at the cost of other quality metrics.
The chosen flagging and refitting criteria for the decomposition might thus prevent the refitting of spectra with fit solutions that show better spatial consistency.
By default, the FWHM values of the fit solutions are never enforced to be within limits established by neighboring fit solutions. To enforce these FWHM constraints, set ``constrain_fwhm = True``, but note that this can lead to artifacts in the fitting (see next question).

#### Is it possible to more strongly enforce fitting constraints?
By default, FWHM values are not forcefully constrained in the spatially coherent refitting stages, as tests showed that this can lead to artifacts (e.g. FWHM values of fit components ending up at the exact enforced limits). Users can nonetheless choose to strongly enforce the FWHM values of fit components based on neighbouring fit solutions by setting ``constrain_fwhm = True``, but are strongly advised to check that this does not negatively impact the decomposition (by e.g. checking whether the histogram or distribution of all fitted FWHM values shows artifacts).

#### I found a bug/error, have a problem that has not been discussed here, or have an idea for improving GaussPy+. What should I do?
In case of bugs and errors please open a new issue or contact us directly. For more details see [Contributing to GaussPy+](CONTRIBUTING.md).
