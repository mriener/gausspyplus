import numpy as np
import os

from astropy import units as u
from astropy.io import fits
from astropy.wcs import WCS

from gausspyplus.utils.spectral_cube_functions import correct_header, change_header, update_header


def pickle_load(path_to_file, binary=True, encoding='latin1'):
    import pickle
    read = 'r'
    if binary:
        read = 'rb'
    with open(path_to_file, read) as pickled_file:
        pickled_data = pickle.load(pickled_file, encoding=encoding)
    return pickled_data


#  combine individual dictionaries for mosaicked tiles into one big dictionary
data = pickle_load(
    os.path.join('decomposition_grs', 'gpy_prepared', 'grs-test_field.pickle'))

decomp = pickle_load(
    os.path.join('decomposition_grs', 'gpy_decomposed', 'grs-test_field_g+_fit_fin_sf-p2.pickle'))

header = correct_header(data['header'])
wcs = WCS(header)
_, _, velocity_offset = wcs.wcs_pix2world(0, 0, 0, 0)
to_kms = wcs.wcs.cdelt[2]*wcs.wcs.cunit[2].to(u.km/u.s)  # conversion factor from channel units to km/s
velocity_offset_kms = velocity_offset * wcs.wcs.cunit[2].to(u.km/u.s)  # offset in spectral axis

header_pp = change_header(header, format='pp')  # create header for positon-position fits file

array = np.ones((header_pp['NAXIS2'], header_pp['NAXIS1'])) * np.nan
array_amp = array.copy()
array_fwhm = array.copy()
array_mean = array.copy()

for (y, x), fwhms, amps, means in zip(
        data['location'],
        decomp['fwhms_fit'],
        decomp['amplitudes_fit'],
        decomp['means_fit']):
    if fwhms is None:
        continue

    if len(fwhms) == 0:
        continue
    else:
        idx_fc = np.argmin(means)  # get index of first component
        amp_fc = amps[idx_fc]  # get intensity of first component
        fwhm_fc = fwhms[idx_fc] * to_kms  # get FWHM of first component and convert to km/s
        mean_fc = means[idx_fc] * to_kms + velocity_offset_kms  # get centroid of first component and convert to km/s

        array_amp[y, x] = amp_fc
        array_mean[y, x] = mean_fc
        array_fwhm[y, x] = fwhm_fc

for array, param, name in zip(
        [array_amp, array_mean, array_fwhm],
        ['Amplitude', 'Centroid velocity', 'FWHM'],
        ['amp', 'vel', 'fwhm']):
    filename = os.path.join(
        'decomposition_grs', 'gpy_maps', 'grs-test_field_g+_fit_fin_sf-p2-map_{}_fc.fits'.format(name))
    comments = [
        'GaussPy+ decomposition results.',
        '{} of first fit component.'.format(param)]
    header = update_header(
        header_pp, comments=comments, remove_old_comments=True)
    fits.writeto(
        filename, array.astype('float32'), header, overwrite=True)
