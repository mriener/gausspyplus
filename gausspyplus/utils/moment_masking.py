# @Author: riener
# @Date:   2019-01-09T12:27:55+01:00
# @Filename: moment_masking.py
# @Last modified by:   riener
# @Last modified time: 2019-04-08T10:17:48+02:00

"""Moment masking procedure from Dame (2011)."""

import os
import itertools
import warnings

import numpy as np

from astropy.io import fits
from astropy import units as u
from astropy.wcs import WCS
from tqdm import tqdm

from .noise_estimation import get_max_consecutive_channels, calculate_average_rms_noise
from .spectral_cube_functions import remove_additional_axes, spatial_smoothing, spectral_smoothing, open_fits_file, moment_map, pv_map, correct_header
from .grouping_functions import get_neighbors


def say(message, verbose=False):
    """Diagnostic messages."""
    if verbose is True:
        print(message)


class MomentMask(object):
    """Moment masking procedure from Dame (2011)."""

    def __init__(self):
        self.path_to_file = None
        self.output_directory = None
        self.slice_params = None
        self.p_limit = 0.025
        self.pad_channels = 5
        self.use_ncpus = None
        self.path_to_noise_map = None
        self.masking_cube = None
        self.target_resolution_spatial = None
        self.target_resolution_spectral = None
        self.current_resolution_spatial = None
        self.current_resolution_spectral = None
        self.number_rms_spectra = 1000
        self.clipping_level = 5
        self.verbose = True
        self.random_seed = 111

    def check_settings(self):
        if self.path_to_file is None:
            raise Exception("Need to specify 'path_to_file'")
        self.dirname = os.path.dirname(self.path_to_file)
        self.file = os.path.basename(self.path_to_file)
        self.filename, self.file_extension = os.path.splitext(self.file)

        if self.output_directory is not None:
            if not os.path.exists(self.output_directory):
                os.makedirs(self.output_directory)
        else:
            self.output_directory = os.path.dirname(self.path_to_file)

    def prepare_cube(self):
        # self.check_settings()

        hdu = fits.open(self.path_to_file)[0]
        self.data = hdu.data
        self.header = hdu.header

        self.header = correct_header(self.header)
        self.wcs = WCS(self.header)

        self.data, self.header = remove_additional_axes(self.data, self.header)

        yMax = self.data.shape[1]
        xMax = self.data.shape[2]
        self.locations = list(itertools.product(range(yMax), range(xMax)))

        self.n_channels = self.data.shape[0]
        self.max_consecutive_channels = get_max_consecutive_channels(self.n_channels, self.p_limit)

        self.nan_mask = np.isnan(self.data)
        self.nan_mask_2D = np.zeros((yMax, xMax))
        for ypos, xpos in self.locations:
            self.nan_mask_2D[ypos, xpos] = self.nan_mask[:, ypos, xpos].all()
        self.nan_mask_2D = self.nan_mask_2D.astype('bool')

        if self.path_to_noise_map is not None:
            self.noiseMap = open_fits_file(self.path_to_noise_map, get_header=False)

        if self.current_resolution_spatial is None:
            self.current_resolution_spatial = abs(
                self.wcs.wcs.cdelt[0]) * self.wcs.wcs.cunit[0]

        if self.current_resolution_spectral is None:
            self.current_resolution_spectral = abs(
                self.wcs.wcs.cdelt[2]) * self.wcs.wcs.cunit[2]

        if self.target_resolution_spatial is None:
            self.target_resolution_spatial = 2*self.current_resolution_spatial
            warnings.warn('No smoothing resolution specified. Will smooth to a resolution of {}'.format(self.target_resolution_spatial))

        if self.target_resolution_spectral is None:
            self.target_resolution_spectral = 2*self.current_resolution_spectral
            warnings.warn('No smoothing resolution specified. Will smooth to a resolution of {}'.format(self.target_resolution_spectral))

        if self.target_resolution_spatial <= self.current_resolution_spatial:
            raise Exception('target_resolution_spatial had to be >= current_resolution_spatial')
        if self.target_resolution_spectral <= self.current_resolution_spectral:
            raise Exception('target_resolution_spectral had to be >= current_resolution_spectral')

        self.n_s = round(0.5*self.target_resolution_spatial.value /
                         self.current_resolution_spatial.value)
        self.n_v = round(0.5*self.target_resolution_spectral.value /
                         self.current_resolution_spectral.value)

    def moment_masking(self):
        say('Preparing cube ...', verbose=self.verbose)
        self.check_settings()
        self.prepare_cube()

        if self.masking_cube is None:
            self.moment_masking_first_steps()

        self.moment_masking_final_step()

    def moment_masking_first_steps(self):
        # 1) Determine the rms noise in T(v,x,y) (if noise map is not supplied)

        # 2) Generate a smoothed version of the data cube T_S(v,x,y) by degrading the resolution spatially by a factor of ~2 and in velocity to the width of the narrowest spectral lines generally observed.

        say('Smoothing cube spatially to a resolution of {} ...'.format(self.target_resolution_spatial), verbose=self.verbose)

        self.dataSmoothed, self.headerSmoothed = spatial_smoothing(
            self.data.copy(), self.header, target_resolution=self.target_resolution_spatial,
            current_resolution=self.current_resolution_spatial)

        say('Smoothing cube spectrally to a resolution of {} ...'.format(self.target_resolution_spectral), verbose=self.verbose)

        self.dataSmoothed, self.headerSmoothed = spectral_smoothing(
            self.dataSmoothed, self.headerSmoothed, target_resolution=self.target_resolution_spectral,
            current_resolution=self.current_resolution_spectral)

        self.dataSmoothedWithNans = self.dataSmoothed.copy()
        for ypos, xpos in self.locations:
            nan_mask = self.nan_mask[:, ypos, xpos]
            self.dataSmoothedWithNans[:, ypos, xpos][nan_mask] = np.nan

        # 3) Determine the rms noise in T_S(v,x,y)
        # TODO: Take care that your smoothing algorithm does not zero (rather than blank) edge pixels since this would artificially lower the rms. Likewise be aware of under-sampled regions that were filled by linear interpolation, since these will have higher rms in the smoothed cube.

        if self.path_to_noise_map is None:
            self.calculate_rms_noise()
        else:
            say('Using rms values from {} for the thresholding step for the smoothed cube...'.format(os.path.basename(self.path_to_noise_map)), verbose=self.verbose)

        # 4) Generate a masking cube M(v,x,y) initially filled with zeros with the same dimensions as T and TS. The moment masked cube TM(v,x,y) will be calculated as M*T.

        say('Moment masking ...', verbose=self.verbose)
        self.masking_cube = np.zeros(self.data.shape)

        # 5) For each pixel T_S(vi, xj, yk) > Tc, unmask (set to 1) the corresponding pixel in M. Also unmask all pixels in M within the smoothing kernel of T_S(vi, xj, yk), since all of these pixels weigh into the value of T_S(vi, xj, yk). That is, unmask within n_v pixels in velocity and within n_s pixels spatially, where n_v = 0.5*fwhm_v / dv and n_s = 0.5*fwhm_s / ds

        pbar = tqdm(total=len(self.locations))

        for ypos, xpos in self.locations:
            pbar.update()
            spectrum_smoothed = self.dataSmoothed[:, ypos, xpos]
            nan_mask = self.nan_mask[:, ypos, xpos]
            spectrum_smoothed[nan_mask] = 0
            rms_smoothed = self.noiseSmoothedCube[ypos, xpos]

            #  do not unmask anything if rms could not be calculated
            if np.isnan(rms_smoothed):
                continue

            if np.isnan(spectrum_smoothed).any():
                print("Nans", ypos, xpos)
            mask_v = spectrum_smoothed > self.clipping_level*rms_smoothed
            mask_v = self.mask_pixels_in_velocity(mask_v)

            position_of_spectra_within_n_s = get_neighbors(
                (ypos, xpos), exclude_p=False, shape=self.data.shape[1:], nNeighbors=self.n_s)
            for pos in position_of_spectra_within_n_s:
                self.masking_cube[:, pos[0], pos[1]][mask_v] = 1
        pbar.close()

    def moment_masking_final_step(self):
        for ypos, xpos in self.locations:
            nan_mask = self.nan_mask[:, ypos, xpos]
            mask = self.masking_cube[:, ypos, xpos]
            mask[nan_mask] = 0
            mask = mask.astype('bool')
            self.data[:, ypos, xpos][~mask] = 0

    def calculate_rms_noise(self):
        say('Determining average rms noise from {} spectra ...'.format(self.number_rms_spectra), verbose=self.verbose)
        average_rms = calculate_average_rms_noise(
            self.dataSmoothedWithNans, self.number_rms_spectra,
            max_consecutive_channels=self.max_consecutive_channels,
            pad_channels=self.pad_channels, random_seed=self.random_seed)
        say('Determined average rms value of {}'.format(average_rms), verbose=self.verbose)

        say('Determining noise of smoothed cube ...', verbose=self.verbose)

        self.noiseSmoothedCube = np.empty(
            (self.data.shape[1], self.data.shape[2]))

        import gausspyplus.parallel_processing
        gausspyplus.parallel_processing.init([self.locations, [self.dataSmoothedWithNans, self.max_consecutive_channels, self.pad_channels, average_rms]])

        results_list = gausspyplus.parallel_processing.func(use_ncpus=self.use_ncpus, function='noise')

        for i, rms in tqdm(enumerate(results_list)):
            if not isinstance(rms, np.float):
                warnings.warn('Problems with entry {} from resulting parallel_processing list, skipping entry'.format(i))
                continue
            else:
                ypos, xpos = self.locations[i]
                self.noiseSmoothedCube[ypos, xpos] = rms

    def mask_pixels_in_velocity(self, mask):
        mask = mask.astype('float')
        mask_new = mask.copy()
        for i in range(1, self.n_v + 1):
            # unmask element to the left
            mask_new += np.append(mask[i:], np.zeros(i))
            # unmask element to the right
            mask_new += np.append(np.zeros(i), mask[:-i])
        mask_new = mask_new.astype('bool')
        return mask_new

    def make_moment_map(self, order=0, save=True, get_hdu=False,
                        vel_unit=u.km/u.s, restore_nans=True, slice_params=None,
                        suffix=''):
        path_to_output_file = self.get_path_to_output_file(
            suffix='{}_mom_{}_map'.format(suffix, order))
        hdu = fits.PrimaryHDU(
            data=self.data.copy(), header=self.header.copy())
        if slice_params is None:
            slice_params = self.slice_params
        moment_map(hdu=hdu, slice_params=slice_params, save=save,
                   order=order, path_to_output_file=path_to_output_file,
                   vel_unit=vel_unit, apply_noise_threshold=False,
                   get_hdu=get_hdu, restore_nans=restore_nans,
                   nan_mask=self.nan_mask_2D)

    def make_pv_map(self, save=True, get_hdu=False, vel_unit=u.km/u.s,
                    sum_over_latitude=True, suffix='', slice_params=None):
        path_to_output_file = self.get_path_to_output_file(
            suffix='{}_pv_map'.format(suffix))
        hdu = fits.PrimaryHDU(
            data=self.data.copy(), header=self.header.copy())
        if slice_params is None:
            slice_params = self.slice_params
        pv_map(hdu=hdu, slice_params=slice_params, get_hdu=False, save=True,
               path_to_output_file=path_to_output_file, apply_noise_threshold=False,
               sum_over_latitude=sum_over_latitude)

    def get_path_to_output_file(self, suffix=''):
        if self.output_directory is not None:
            filename = '{}{}.fits'.format(self.filename, suffix)
            path_to_output_file = os.path.join(self.output_directory, filename)
        else:
            path_to_output_file = None
        return path_to_output_file
