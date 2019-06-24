# @Author: riener
# @Date:   2019-02-18T16:27:12+01:00
# @Filename: spectral_cube_functions.py
# @Last modified by:   riener
# @Last modified time: 09-04-2019


import getpass
import itertools
import os
import socket
import warnings

import numpy as np

from astropy import units as u
from astropy.convolution import Gaussian1DKernel, Gaussian2DKernel, convolve
from astropy.io import fits
from astropy.wcs import WCS
from datetime import datetime
from tqdm import tqdm

from .output import check_if_value_is_none, check_if_all_values_are_none, format_warning, save_file
from .noise_estimation import get_max_consecutive_channels, determine_noise

warnings.showwarning = format_warning


def transform_header_from_crota_to_pc(header):
    """Replace CROTA* keywords with PC*_* keywords."""
    cdelt1 = header['CDELT1']
    cdelt2 = header['CDELT2']

    if 'CROTA1' in header.keys():
        crota = np.radians(header['CROTA1'])
        if crota != 0:
            warnings.warn(
                "Replacing 'CROTA*' with 'PC*_*' keywords in FITS header.")
            header['PC1_1'] = np.cos(crota)
            header['PC1_2'] = -(cdelt2 / cdelt1) * np.sin(crota)
            header['PC2_1'] = (cdelt1 / cdelt1) * np.sin(crota)
            header['PC2_2'] = np.cos(crota)
        else:
            warnings.warn("'CROTA*' keywords with value 0. present in FITS header.")

    warnings.warn("Removing 'CROTA*' keywords from FITS header.")

    for key in list(header.keys()):
        if key.startswith('CROTA'):
            header.remove(key)

    return header


def correct_header(header, check_keywords={'BUNIT': 'K', 'CUNIT1': 'deg',
                                           'CUNIT2': 'deg', 'CUNIT3': 'm/s'},
                   keep_only_wcs_keywords=False):
    """Correct FITS header by checking keywords or removing unnecessary keywords.

    If 'CROTA*' keywords are present they either get deleted (if their value is 0.) or they are transformed to 'PC*-*' keywords.

    Parameters
    ----------
    header : astropy.io.fits.Header
        FITS header.
    check_keywords : dict
        Dictionary of FITS header keywords and corresponding values. The FITS header is checked for the presence of these keywords; if the keyword is not existing they are written to the header with the supplied values.
    keep_only_wcs_keywords : bool
        Default is `False`. If set to `True`, the FITS header is stripped of all keywords other than the required minimum WCS keywords.

    Returns
    -------
    header : astropy.io.fits.Header
        Corrected FITS header.

    """
    for keyword, value in check_keywords.items():
        if keyword not in list(header.keys()):
            warnings.warn("{a} keyword not found in header. Assuming {a}={b}".format(a=keyword, b=value), stacklevel=2)
            header[keyword] = value
    if 'CTYPE3' in list(header.keys()):
        if header['CTYPE3'] == 'VELOCITY':
            warnings.warn("Changed header keyword CTYPE3 from VELOCITY to VELO-LSR")
            header['CTYPE3'] = 'VELO-LSR'
    if keep_only_wcs_keywords:
        wcs = WCS(header)
        dct_naxis = {}
        for keyword in list(header.keys()):
            if keyword.startswith('NAXIS'):
                dct_naxis[keyword] = header[keyword]
        header = wcs.to_header()
        for keyword, value in dct_naxis.items():
            header[keyword] = value
    if 'CROTA1' in list(header.keys()):
        header = transform_header_from_crota_to_pc(header)
    return header


def update_header(header, comments=[], remove_keywords=[], update_keywords={},
                  remove_old_comments=False, write_meta=True):
    """Update FITS header.

    Parameters
    ----------
    header : astropy.io.fits.Header
        FITS header.
    comments : list
        List of comments that get written to the FITS header with the 'COMMENT' keyword.
    remove_keywords : list
        List of FITS header keywords that should be removed.
    update_keywords : dict
        Dictionary of FITS header keywords that get updated.
    remove_old_comments : bool
        Default is `False`. If set to `True`, existing 'COMMENT' keywords of the FITS header are removed.
    write_meta : bool
        Default is `True`. Adds or updates 'AUTHOR', 'ORIGIN', and 'DATE' FITS header keywords.

    Returns
    -------
    header : astropy.io.fits.Header
        Updated FITS header.

    """
    if remove_old_comments:
        while ['COMMENT'] in header.keys():
            header.remove('COMMENT')

    for keyword in remove_keywords:
        if keyword in header.keys():
            header.remove(keyword)

    for keyword, value in update_keywords.items():
        header[keyword] = value[0][1]

    if write_meta:
        header['AUTHOR'] = getpass.getuser()
        header['ORIGIN'] = socket.gethostname()
        header['DATE'] = (datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S'), '(GMT)')

    for comment in comments:
        header['COMMENT'] = comment

    return header


def change_wcs_header_reproject(header, header_new, ppv=True):
    """Change the WCS information of a header for reprojection purposes.

    Parameters
    ----------
    header : astropy.io.fits.Header
        Header of the FITS array.
    header_new : astropy.io.fits.Header
        Header of the FITS array to which the data gets reprojected.
    ppv : bool
        Default is `True`, if the header should remain in the PPV format. If set to `False`, the header will be transformed to a PP format.

    Returns
    -------
    header : astropy.io.fits.Header
        Updated header of the FITS array.

    """
    wcs_new = WCS(correct_header(header_new))
    while wcs_new.wcs.naxis > 2:
        axes = range(wcs_new.wcs.naxis)
        wcs_new = wcs_new.dropaxis(axes[-1])
    wcs_header_new = wcs_new.to_header()

    header_diff = fits.HeaderDiff(header, wcs_header_new)

    if ppv:
        update_header(header, update_keywords=header_diff.diff_keyword_values,
                      write_meta=False)
        header['WCSAXES'] = 3
    else:
        wcs = WCS(correct_header(header))
        wcs_header = wcs.to_header()
        wcs_header_diff = fits.HeaderDiff(wcs_header, wcs_header_new)
        remove_keywords = []
        if wcs_header_diff.diff_keywords:
            remove_keywords = wcs_header_diff.diff_keywords[0]
        update_header(header, remove_keywords=remove_keywords,
                      update_keywords=header_diff.diff_keyword_values,
                      write_meta=False)
        if 'NAXIS3' in header.keys():
            header.remove('NAXIS3')
        header['NAXIS'] = 2

    header['NAXIS1'] = header_new['NAXIS1']
    header['NAXIS2'] = header_new['NAXIS2']
    return header


def remove_additional_axes(data, header, max_dim=3,
                           keep_only_wcs_keywords=False):
    """Remove additional axes (Stokes, etc.) from spectral cube.

    The old name of the function was 'remove_stokes'.

    Parameters
    ----------
    data : numpy.ndarray
        Data of the FITS array.
    header : astropy.io.fits.Header
        Header of the FITS array.
    max_dim : int
        Maximum number of dimensions the final data array should have. The default value is '3'.
    keep_only_wcs_keywords : bool
        Default is `False`. If set to `True`, the FITS header is stripped of all keywords other than the required minimum WCS keywords.

    Returns
    -------
    data : numpy.ndarray
        Data of the FITS array, corrected for additional unwanted axes.
    header : astropy.io.fits.Header
        Updated FITS header.

    """
    wcs = WCS(header)

    if header['NAXIS'] <= max_dim and wcs.wcs.naxis <= max_dim:
        return data, header

    warnings.warn('remove additional axes (Stokes, etc.) from cube and/or header')

    while data.ndim > max_dim:
        data = np.squeeze(data, axis=(0,))

    wcs_header_old = wcs.to_header()
    while wcs.wcs.naxis > max_dim:
        axes = range(wcs.wcs.naxis)
        wcs = wcs.dropaxis(axes[-1])
    wcs_header_new = wcs.to_header()

    if keep_only_wcs_keywords:
        hdu = fits.PrimaryHDU(data=data, header=wcs_header_new)
        return hdu.data, hdu.header

    wcs_header_diff = fits.HeaderDiff(wcs_header_old, wcs_header_new)
    header_diff = fits.HeaderDiff(header, wcs_header_new)
    update_header(header, remove_keywords=wcs_header_diff.diff_keywords[0],
                  update_keywords=header_diff.diff_keyword_values,
                  write_meta=False)

    return data, header


def swap_axes(data, header, new_order):
    """Swap the axes of a FITS cube.

    Parameters
    ----------
    data : numpy.ndarray
        Data of the FITS array.
    header : astropy.io.fits.Header
        Header of the FITS array.
    new_order : tuple
        New order of the axes of the FITS array, e.g. (2, 1, 0).

    Returns
    -------
    data : numpy.ndarray
        FITS array with swaped axes.
    header : astropy.io.fits.Header
        Updated FITS header.

    """
    dims = data.ndim
    data = np.transpose(data, new_order)
    hdu = fits.PrimaryHDU(data=data)
    header_new = hdu.header

    if 'CD1_1' in list(header.keys()):
        raise Exception('Cannot swap_axes for CDX_X keywords. Convert them to CDELTX.')

    for keyword in list(header.keys()):
        for axis in range(dims):
            if keyword.endswith(str(axis + 1)):
                keyword_new = keyword.replace(str(axis + 1), str(new_order[axis] + 1))
                header_new[keyword_new] = header[keyword]

    header_diff = fits.HeaderDiff(header, header_new)
    for keyword, value in header_diff.diff_keyword_values.items():
        header[keyword] = value[0][1]
    return data, header


def get_spectral_axis(header=None, channels=None, wcs=None, to_unit=None):
    """Return the spectral axis of a Spectral cube in physical values.

    Parameters
    ----------
    header : astropy.io.fits.Header
        Header of the FITS array.
    channels : numpy.ndarray
        Array of the channels [0, ..., N].
    wcs : astropy.wcs.wcs.WCS
        WCS parameters of the FITS array.
    to_unit : astropy.units.quantity.Quantity
        Valid unit to which the values of the spectral axis will be converted.

    Returns
    -------
    spectral_axis : numpy.ndarray
        The (unitless) spectral axis of the spectral cube, converted to 'to_unit' (if specified).

    """
    check_if_all_values_are_none(header, wcs, 'header', 'wcs')
    check_if_all_values_are_none(header, channels, 'header', 'channels')
    if header:
        wcs = WCS(header)
        channels = np.arange(header['NAXIS3'])

    x_wcs, y_wcs, spectral_axis = wcs.wcs_pix2world(0, 0, channels, 0)

    if to_unit:
        conversion_factor = wcs.wcs.cunit[2].to(to_unit)
        spectral_axis *= conversion_factor
    return spectral_axis


def get_slice_parameters(path_to_file=None, header=None, wcs=None,
                         range_x_wcs=[None, None], range_y_wcs=[None, None], range_z_wcs=[None, None],
                         x_unit=None, y_unit=None, z_unit=None,
                         include_max_val=True, get_slices=True):
    """Get slice parameters in pixels for given coordinate ranges.

    Parameters
    ----------
    header : astropy.io.fits.Header
        Header of the FITS array.
    wcs : astropy.wcs.wcs.WCS
        WCS parameters of the FITS array.
    range_x_wcs : list
        Coordinate ranges in the x coordinate given as [xmin, xmax].
    range_y_wcs : list
        Coordinate ranges in the y coordinate given as [ymin, ymax].
    range_z_wcs : list
        Coordinate ranges in the y coordinate given as [zmin, zmax].
    x_unit : astropy.units.quantity.Quantity
        Unit of x coordinates (default is u.deg).
    y_unit : astropy.units.quantity.Quantity
        Unit of y coordinates (default is u.deg).
    z_unit : astropy.units.quantity.Quantity
        Unit of z coordinates (default is u.m/u.s).
    include_max_val : bool
        Default is `True`. Includes the upper coordinate value in the slice parameters.
    get_slices : bool
        Default is `True`. If set to `False`, a tuple of the slice parameters is returned instead of the slices.

    Returns
    -------
    slices : tuple
        Slice parameters given in pixel values in the form (slice(zmin, zmax), slice(ymin, ymax), slice(xmin, xmax)). If 'get_slices' is set to `False`, ((zmin, zmax), (ymin, ymax), (xmin, xmax)) is returned instead.

    """
    if x_unit is None:
        x_unit = u.deg
        warnings.warn('No unit for x_unit supplied. Assuming {} for x_unit.'.format(x_unit))
    if y_unit is None:
        y_unit = u.deg
        warnings.warn('No unit for y_unit supplied. Assuming {} for y_unit.'.format(y_unit))
    if z_unit is None:
        z_unit = u.m/u.s
        warnings.warn('No unit for z_unit supplied. Assuming {} for z_unit.'.format(z_unit))

    if path_to_file:
        header = correct_header(fits.getheader(path_to_file))
        wcs = WCS(header)
    elif header:
        wcs = WCS(correct_header(header))

    range_x = [val if val is not None else 0 for val in range_x_wcs]
    range_y = [val if val is not None else 0 for val in range_y_wcs]
    range_z = [val if val is not None else 0 for val in range_z_wcs]

    x_wcs_min, x_wcs_max = (range_x * x_unit).to(wcs.wcs.cunit[0]).value
    y_wcs_min, y_wcs_max = (range_y * y_unit).to(wcs.wcs.cunit[1]).value
    z_wcs_min, z_wcs_max = (range_z * z_unit).to(wcs.wcs.cunit[2]).value

    x_pix_min, y_pix_min, z_pix_min = wcs.wcs_world2pix(x_wcs_min, y_wcs_min, z_wcs_min, 0)
    x_pix_max, y_pix_max, z_pix_max = wcs.wcs_world2pix(x_wcs_max, y_wcs_max, z_wcs_max, 0)

    xmin = int(max(0, min(x_pix_min, x_pix_max)))
    ymin = int(max(0, min(y_pix_min, y_pix_max)))
    zmin = int(max(0, min(z_pix_min, z_pix_max)))

    if include_max_val:
        xmax = int(max(x_pix_min, x_pix_max) + 2)
        ymax = int(max(y_pix_min, y_pix_max) + 2)
        zmax = int(max(z_pix_min, z_pix_max) + 2)
    else:
        xmax = int(max(x_pix_min, x_pix_max))
        ymax = int(max(y_pix_min, y_pix_max))
        zmax = int(max(z_pix_min, z_pix_max))

    xmin = None if range_x_wcs[0] is None else xmin
    xmax = None if range_x_wcs[1] is None else xmax
    ymin = None if range_y_wcs[0] is None else ymin
    ymax = None if range_y_wcs[1] is None else ymax
    zmin = None if range_z_wcs[0] is None else zmin
    zmax = None if range_z_wcs[1] is None else zmax

    if not get_slices:
        return ((zmin, zmax), (ymin, ymax), (xmin, xmax))
    return (slice(zmin, zmax), slice(ymin, ymax), slice(xmin, xmax))


def get_slices(size, n):
    """Calculate slices in individual direction."""
    limits, slices = ([] for _ in range(2))

    for i in range(n):
        limits.append(i * size)
    limits.append(None)

    for a, b in zip(limits[:-1], limits[1:]):
        slices.append(slice(a, b))

    return slices


def get_list_slice_params(path_to_file=None, hdu=None, ncols=1, nrows=1,
                          velocity_slice=slice(None, None)):
    """Calculate required slices to split a PPV cube into chosen number of subcubes.

    The total number of subcubes is ncols * nrows.

    Parameters
    ----------
    path_to_file : str
        Filepath to the FITS cube.
    hdu : astropy.io.fits.HDUList
        Header/Data unit of the FITS cube.
    ncols : int
        Number of subcubes along 'NAXIS1'.
    nrows : int
        Number of subcubes along 'NAXIS2'.
    velocity_slice : slice
        Slice parameters for 'NAXIS3'. In the default settings the subcubes contain the full 'NAXIS3' range.

    Returns
    -------
    slices: list
        List containing slicing parameters for all three axes ('NAXIS1', 'NAXIS2', 'NAXIS3') of the FITS cube.

    """
    check_if_all_values_are_none(hdu, path_to_file, 'hdu', 'path_to_file')

    if path_to_file is not None:
        hdu = fits.open(path_to_file)[0]

    x = hdu.header['NAXIS1']
    y = hdu.header['NAXIS2']

    x_size = int(x / ncols)
    y_size = int(y / nrows)

    x_slices = get_slices(x_size, ncols)
    y_slices = get_slices(y_size, nrows)

    slices = []

    for y_slice, x_slice in itertools.product(y_slices, x_slices):
        slices.append([velocity_slice, y_slice, x_slice])
    return slices


def get_locations(data=None, header=None):
    """Get a list of index values [(y0, x0), ..., (yN, xM)] for (N x M) array."""
    if data is not None:
        yValues = np.arange(data.shape[1])
        xValues = np.arange(data.shape[2])
    else:
        yValues = np.arange(header['NAXIS2'])
        xValues = np.arange(header['NAXIS1'])
    return list(itertools.product(yValues, xValues))


def save_fits(data, header, path_to_file, verbose=True):
    """Save data array and header as FITS file.

    Parameters
    ----------
    data : numpy.ndarray
        Data array.
    header : astropy.io.fits.Header
        Header of the FITS array.
    path_to_file : str
        Filepath to which FITS array should get saved.
    verbose : bool
        Default is `True`. Writes message to terminal about where the FITS file was saved.

    """
    if not os.path.exists(os.path.dirname(path_to_file)):
        os.makedirs(os.path.dirname(path_to_file))
    fits.writeto(path_to_file, data, header=header, overwrite=True)
    if verbose:
        save_file(os.path.basename(path_to_file), os.path.dirname(path_to_file))


def open_fits_file(path_to_file, get_hdu=False, get_data=True, get_header=True,
                   remove_stokes=True, check_wcs=True):
    """Open a FITS file and return the HDU or data and/or header.

    Parameters
    ----------
    path_to_file : str
        Filepath to FITS array.
    get_hdu : bool
        Default is `False`. If set to `True`, an astropy.io.fits.HDUList is returned. Overrides 'get_data' and 'get_header'.
    get_data : bool
        Default is `True`. Returns a numpy.ndarray of the FITS array.
    get_header : bool
        Default is `True`. Returns a astropy.io.fits.Header of the FITS array.
    remove_stokes : bool
        Default is `True`. If the FITS array contains more than three axes, these additional axes are removed so that the FITS array contains only ('NAXIS1', 'NAXIS2', 'NAXIS3').
    check_wcs : bool
        Default is `True`. Corrects the FITS header with the default settings.

    Returns
    -------
    astropy.io.fits.HDUList or numpy.ndarray and/or astropy.io.fits.Header.

    """

    data = fits.getdata(path_to_file)
    header = fits.getheader(path_to_file)

    if remove_stokes:
        data, header = remove_additional_axes(data, header)

    if check_wcs:
        header = correct_header(header)

    if get_hdu:
        return fits.PrimaryHDU(data, header)
    elif get_data and (not get_header):
        return data
    elif (not get_data) and get_header:
        return header
    else:
        return data, header


def reproject_data(input_data, output_projection, shape_out):
    """Reproject data to a different projection.

    Parameters
    ----------
    input_data : type
        Description of parameter `input_data`.
    output_projection : type
        Description of parameter `output_projection`.
    shape_out : type
        Description of parameter `shape_out`.

    Returns
    -------
    type
        Description of returned object.

    """
    from reproject import reproject_interp

    data_reprojected, footprint = reproject_interp(
        input_data, output_projection, shape_out=shape_out)
    return data_reprojected


def spatial_smoothing(data, header, save=False, path_to_output_file=None,
                      suffix=None, current_resolution=None,
                      target_resolution=None, verbose=True,
                      reproject=False, header_projection=None):
    """Smooth a FITS cube spatially and update its header.

    The data can only be smoothed to a circular beam.

    Parameters
    ----------
    data : numpy.ndarray
        Data array of the FITS cube.
    header : astropy.io.fits.Header
        Header of the FITS cube.
    save : bool
        Default is `False`. If set to `True`, the smoothed FITS cube is saved under 'path_to_output_file'.
    path_to_output_file : str
        Filepath to which smoothed FITS cube gets saved.
    suffix : str
        Suffix that gets added to the filename.
    current_resolution : astropy.units.quantity.Quantity
        Current size of the resolution element (FWHM of the beam).
    target_resolution : astropy.units.quantity.Quantity
        Final resolution element after smoothing.
    verbose : bool
        Default is `True`. Writes diagnostic messages to the terminal.
    reproject : bool
        Default is `False`. Set to `True` if data should be reprojected to `header_projection`.
    header_projection : astropy.io.fits.Header
        Header of the FITS array to which data should be reprojected to.

    Returns
    -------
    data : numpy.ndarray
        Smoothed data array of the FITS cube.
    header : astropy.io.fits.Header
        Updated header of the FITS cube.

    """
    check_if_value_is_none(save, path_to_output_file, 'save', 'path_to_output_file')
    check_if_all_values_are_none(current_resolution, target_resolution,
                                 'current_resolution', 'target_resolution')

    header_pp = correct_header(header.copy())
    header_pp = change_wcs_header_reproject(
        header_pp, header_pp, ppv=False)
    wcs_pp = WCS(header_pp)

    fwhm_factor = np.sqrt(8*np.log(2))
    pixel_scale = abs(wcs_pp.wcs.cdelt[0]) * wcs_pp.wcs.cunit[0]

    if target_resolution is None:
        target_resolution = 2*current_resolution
        warnings.warn('No smoothing resolution specified. Will smooth to a resolution of {}'.format(target_resolution))

    current_resolution = current_resolution.to(u.deg)
    target_resolution = target_resolution.to(u.deg)
    pixel_scale = pixel_scale.to(u.deg)

    if 'BMAJ' in header.keys() and 'BMIN' in header.keys():
        if header['BMAJ'] != header['BMIN']:
            warnings.warn(str(
                'BMAJ != BMIN for input FITS array. '
                'Smoothing to circular beam with HPBW of {}'.format(
                    target_resolution)))
        header['BMAJ'] = target_resolution.value
        header['BMIN'] = target_resolution.value

    kernel_fwhm = np.sqrt(target_resolution.value**2 -
                          current_resolution.value**2)
    kernel_std = (kernel_fwhm / fwhm_factor) / pixel_scale.value
    kernel = Gaussian2DKernel(kernel_std)

    if reproject:
        shape_out = (header_projection['NAXIS2'], header_projection['NAXIS1'])
        header_projection_pp = correct_header(header_projection.copy())
        header_projection_pp = change_wcs_header_reproject(
            header_projection_pp, header_projection_pp, ppv=False)
        output_projection = WCS(header_projection_pp)

    if data.ndim == 2:
        data = convolve(data, kernel, normalize_kernel=True)
        if reproject:
            data_reprojected = reproject_data(
                (data, wcs_pp), output_projection, shape_out)
            header = change_wcs_header_reproject(header, header_projection)
    else:
        nSpectra = data.shape[0]
        if reproject:
            data_reprojected = np.zeros((data.shape[0], shape_out[0], shape_out[1]))
        for i in tqdm(range(nSpectra)):
            channel = data[i, :, :]
            channel_smoothed = convolve(channel, kernel, normalize_kernel=True)
            data[i, :, :] = channel_smoothed
            if reproject:
                channel = data[i, :, :]
                data_reprojected[i, :, :] = reproject_data(
                    (channel, wcs_pp), output_projection, shape_out)
        if reproject:
            data = data_reprojected
            header = change_wcs_header_reproject(header, header_projection)

    comments = ['spatially smoothed to a resolution of {}'.format(
        target_resolution)]
    header = update_header(header, comments=comments)

    if save:
        save_fits(data, header, path_to_output_file, verbose=verbose)

    return data, header


def spectral_smoothing(data, header, save=False, path_to_output_file=None,
                       suffix=None, current_resolution=None,
                       target_resolution=None, verbose=True):
    """Smooth a FITS cube spectrally and update its header.

    Parameters
    ----------
    data : numpy.ndarray
        Data array of the FITS cube.
    header : astropy.io.fits.Header
        Header of the FITS cube.
    save : bool
        Default is `False`. If set to `True`, the smoothed FITS cube is saved under 'path_to_output_file'.
    path_to_output_file : str
        Filepath to which smoothed FITS cube gets saved.
    suffix : str
        Suffix that gets added to the filename.
    current_resolution : astropy.units.quantity.Quantity
        Current size of the spectral resolution element (velocity channel).
    target_resolution : astropy.units.quantity.Quantity
        Final spectral resolution element after smoothing.
    verbose : bool
        Default is `True`. Writes diagnostic messages to the terminal.

    Returns
    -------
    data : numpy.ndarray
        Smoothed data array of the FITS cube.
    header : astropy.io.fits.Header
        Updated header of the FITS cube.

    """
    check_if_value_is_none(save, path_to_output_file, 'save', 'path_to_output_file')

    wcs = WCS(header)
    # cube = SpectralCube(data=data, wcs=wcs, header=header)

    fwhm_factor = np.sqrt(8*np.log(2))
    pixel_scale = wcs.wcs.cdelt[2] * wcs.wcs.cunit[2]

    if target_resolution is None:
        target_resolution = 2*current_resolution
        warnings.warn('No smoothing resolution specified. Will smooth to a resolution of {}'.format(target_resolution))

    current_resolution = current_resolution.to(u.m/u.s)
    target_resolution = target_resolution.to(u.m/u.s)
    pixel_scale = pixel_scale.to(u.m/u.s)

    gaussian_width = (
        (target_resolution.value**2 - current_resolution.value**2)**0.5 /
        pixel_scale.value / fwhm_factor)
    kernel = Gaussian1DKernel(gaussian_width)

    #  the next line doesn't work because of a bug in spectral_cube
    #  the new_cube.mask attribute is set to None if cube is defined with data=X and header=X instead of reading it in from the cube; this leads to an error with spectral_smooth
    # new_cube = cube.spectral_smooth(kernel)
    # data = new_cube.hdu.data
    # header = new_cube.hdu.header

    yMax = data.shape[1]
    xMax = data.shape[2]
    locations = list(
            itertools.product(range(yMax), range(xMax)))
    for ypos, xpos in tqdm(locations):
        spectrum = data[:, ypos, xpos]
        spectrum_smoothed = convolve(spectrum, kernel)
        data[:, ypos, xpos] = spectrum_smoothed

    comments = ['spectrally smoothed cube to a resolution of {}'.format(target_resolution)]
    header = update_header(header, comments=comments)

    if save:
        save_fits(data, header, path_to_output_file, verbose=verbose)

    return data, header


def get_path_to_output_file(path_to_file, suffix='_',
                            filename='foo.fits'):
    """Determine filepath for output file.

    Parameters
    ----------
    path_to_file : str
        Filepath of the input data.
    suffix : str
        Suffix to add to the filename.
    filename : str
        Name of the output file.

    Returns
    -------
    path_to_output_file
        Filepath for the output file.

    """
    if path_to_file is None:
        path_to_output_file = os.path.join(os.getcwd(), filename)
    else:
        dirname = os.path.dirname(path_to_file)
        filename = os.path.basename(path_to_file)
        fileBase, fileExtension = os.path.splitext(path_to_file)
        filename = '{}{}{}'.format(fileBase, suffix, fileExtension)
        path_to_output_file = os.path.join(dirname, filename)
    return path_to_output_file


def add_noise(average_rms, path_to_file=None, hdu=None, save=False,
              overwrite=True, path_to_output_file=None, get_hdu=False,
              get_data=True, get_header=True, random_seed=111):
    """Add noise to spectral cube.

    Parameters
    ----------
    average_rms : type
        Description of parameter `average_rms`.
    path_to_file : str
        Filepath to the FITS cube.
    hdu : astropy.io.fits.HDUList
        Header/Data unit of the FITS cube.
    save : bool
        Default is `False`. If set to `True`, the resulting FITS cube is saved under 'path_to_output_file'.
    overwrite : bool
        If set to `True`, overwrites any already existing files saved in `path_to_output_file`.
    path_to_output_file : str
        Filepath to which FITS cube with added noise gets saved.
    get_hdu : bool
        Default is `False`. If set to `True`, an astropy.io.fits.HDUList is returned. Overrides 'get_data' and 'get_header'.
    get_data : bool
        Default is `True`. Returns a numpy.ndarray of the FITS array.
    get_header : bool
        Default is `True`. Returns a astropy.io.fits.Header of the FITS array.
    random_seed : int
        Initializer for np.random package.

    """
    print('\nadding noise (rms = {}) to data...'.format(average_rms))

    check_if_all_values_are_none(hdu, path_to_file, 'hdu', 'path_to_file')

    np.random.seed(random_seed)

    if path_to_file is not None:
        hdu = fits.open(path_to_file)[0]

    data = hdu.data
    header = hdu.header

    channels = data.shape[0]
    yValues = np.arange(data.shape[1])
    xValues = np.arange(data.shape[2])
    locations = list(itertools.product(yValues, xValues))
    for y, x in locations:
        data[:, y, x] += np.random.randn(channels) * average_rms

    header['COMMENT'] = "Added rms noise of {} ({})".format(
            average_rms, datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S'))

    if save:
        if path_to_output_file is None:
            path_to_output_file = get_path_to_output_file(path_to_file, suffix='_w_noise', filename='cube_w_noise.fits')

        save_fits(data, header, path_to_output_file, verbose=True)

    if get_hdu:
        return fits.PrimaryHDU(data, header)
    elif get_data and (not get_header):
        return data
    elif (not get_data) and get_header:
        return header
    else:
        return data, header


def transform_coordinates_to_pixel(coordinates, header):
    """Transform PPV coordinates to pixel positions within the PPV cube.

    Parameters
    ----------
    coordinates : list
        List of coordinates given as [position_coord_1, position_coord_2, velocity_coord]. The coordinates must be specified with astropy.units.
    header : astropy.io.fits.Header
        Header of the FITS cube.

    Returns
    -------
    list
        List of corresponding pixel positions within the PPV cube.

    """
    if not isinstance(coordinates, list):
        coordinates = list(coordinates)
    wcs = WCS(header)
    units = wcs.wcs.cunit

    for i, (coordinate, unit) in enumerate(zip(coordinates, units)):
        if isinstance(coordinate, u.Quantity):
            coordinates[i] = coordinate.to(unit)
        else:
            raise Exception('coordinates must be specified with astropy.units')

    lon, lat, vel = coordinates
    x, y, z = wcs.wcs_world2pix(lon, lat, vel, 0)
    return [max(int(x), 0), max(int(y), 0), max(0, int(z))]


def make_subcube(slice_params, path_to_file=None, hdu=None, dtype='float32',
                 save=False, overwrite=True, path_to_output_file=None,
                 get_hdu=False, get_data=True, get_header=True):
    """Extract subcube from a spectral cube.

    Parameters
    ----------
    slice_params : list
        List of slice parameters for cube, if only a subset of the data should be used.
    path_to_file : str
        Filepath to the FITS cube.
    hdu : astropy.io.fits.HDUList
        Header/Data unit of the FITS cube.
    dtype : str
        Data type to which the array should be transformed. Default is `float32`.
    save : bool
        Default is `False`. If set to `True`, the resulting FITS cube is saved under 'path_to_output_file'.
    overwrite : bool
        If set to `True`, overwrites any already existing files saved in `path_to_output_file`.
    path_to_output_file : type
        Filepath to which subcube gets saved.
    get_hdu : bool
        Default is `False`. If set to `True`, an astropy.io.fits.HDUList is returned. Overrides 'get_data' and 'get_header'.
    get_data : bool
        Default is `True`. Returns a numpy.ndarray of the FITS array.
    get_header : bool
        Default is `True`. Returns a astropy.io.fits.Header of the FITS array.

    """
    print('\nmaking subcube with the slice parameters {}...'.format(
        slice_params))

    check_if_all_values_are_none(hdu, path_to_file, 'hdu', 'path_to_file')

    if path_to_file is not None:
        hdu = fits.open(path_to_file)[0]

    data = hdu.data
    header = hdu.header

    data = data[slice_params[0], slice_params[1], slice_params[2]]
    data = data.astype(dtype)
    wcs = WCS(correct_header(header))
    wcs_cropped = wcs[slice_params[0], slice_params[1], slice_params[2]]
    header.update(wcs_cropped.to_header())

    header['NAXIS1'] = data.shape[2]
    header['NAXIS2'] = data.shape[1]
    header['NAXIS3'] = data.shape[0]
    header['COMMENT'] = "Cropped FITS file ({})".format(
            datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S'))

    if save:
        if path_to_output_file is None:
            path_to_output_file = get_path_to_output_file(
                path_to_file, suffix='_sub', filename='subcube.fits')

        save_fits(data, header, path_to_output_file, verbose=True)

    if get_hdu:
        return fits.PrimaryHDU(data, header)
    elif get_data and (not get_header):
        return data
    elif (not get_data) and get_header:
        return header
    else:
        return data, header


def clip_noise_below_threshold(data, snr=3, path_to_noise_map=None,
                               slice_params=(slice(None), slice(None)),
                               p_limit=0.02, pad_channels=5, use_ncpus=None):
    """Set all data values below a specified signal-to-noise ratio to zero.

    Parameters
    ----------
    data : numpy.ndarray
        3D array of the input data.
    snr : int, float
        Signal-to-noise ratio for `apply_noise_threshold`.
    path_to_noise_map : str
        Filepath to the noise map of the input FITS cube.
    slice_params : list
        List of slice parameters for cube, if only a subset of the data should be used.
    p_limit : float
        Maximum probability for consecutive positive/negative channels being
        due to chance.
    pad_channels : int
        Number of channels by which an interval (low, upp) gets extended on both sides, resulting in (low - pad_channels, upp + pad_channels).
    use_ncpus : int
        Number of CPUs used in parallel processing. By default 75% of all CPUs on the machine are used.

    Returns
    -------
    data : numpy.ndarray
        3D array of the input data with all values below 'snr' set to zero.

    """
    yMax = data.shape[1]
    xMax = data.shape[2]
    n_channels = data.shape[0]
    locations = list(
            itertools.product(range(yMax), range(xMax)))

    if path_to_noise_map is not None:
        print('\nusing supplied noise map to apply noise threshold '
              'with snr={}...'.format(snr))
        noiseMap = open_fits_file(
            path_to_noise_map, get_header=False, remove_stokes=False, check_wcs=False)
        noiseMap = noiseMap[slice_params]
    else:
        print('\napplying noise threshold to data with snr={}...'.format(snr))
        noiseMap = np.zeros((yMax, xMax))
        max_consecutive_channels = get_max_consecutive_channels(n_channels, p_limit)

        import gausspyplus.parallel_processing
        gausspyplus.parallel_processing.init([locations, determine_noise, [data, max_consecutive_channels, pad_channels]])

        results_list = gausspyplus.parallel_processing.func(use_ncpus=use_ncpus, function='noise')

        for i, rms in tqdm(enumerate(results_list)):
            if not isinstance(rms, np.float):
                warnings.warn('Problems with entry {} from resulting parallel_processing list, skipping entry'.format(i))
                continue
            else:
                ypos, xpos = locations[i]
                noiseMap[ypos, xpos] = rms

    for idx, (y, x) in enumerate(locations):
        spectrum = data[:, y, x]
        noise = noiseMap[y, x]
        spectrum = spectrum - snr*noise

        if not np.isnan(spectrum).any():
            if len(spectrum[np.nonzero(spectrum)]) == 0:
                spectrum = np.array([0.0])*data.shape[0]
            elif not (spectrum > 0).all():
                mask = np.nan_to_num(spectrum) < 0.  # snr*noise
                spectrum[mask] = 0
                spectrum[~mask] += snr*noise
            else:
                """
                To be implemented -> What to do when spectrum only has
                positive values?
                """
        elif not np.isnan(spectrum).all():
            mask = np.nan_to_num(spectrum) < 0.  # snr*noise
            spectrum[mask] = 0
            spectrum[~mask] += snr*noise

        data[:, y, x] = spectrum

    return data


def change_header(header, format='pp', keep_axis='1', comments=[], dct_keys={}):
    """Change the FITS header of a file.

    Parameters
    ----------
    header : astropy.io.fits.Header
        Header of the FITS array.
    format : 'pp' or 'pv'
        Describes the format of the resulting 2D header: 'pp' for position-position data and 'pv' for position-velocity data.
    keep_axis : '1' or '2'
        If format is set to 'pv', this specifies which spatial axis is kept: '1' - NAXIS1 stays and NAXIS2 gets removed, '2' - NAXIS2 stays and NAXIS1 gets removed
    comments : list
        Comments that are added to the FITS header under the COMMENT keyword.
    dct_keys : dict
        Dictionary that specifies which keywords and corresponding values should be added to the FITS header.

    Returns
    -------
    astropy.io.fits.Header
        Updated FITS header.

    """
    prihdr = fits.Header()
    for key in ['SIMPLE', 'BITPIX']:
        prihdr[key] = header[key]

    prihdr['NAXIS'] = 2
    prihdr['WCSAXES'] = 2

    keys = ['NAXIS', 'CRPIX', 'CRVAL', 'CDELT', 'CUNIT', 'CROTA']

    if format == 'pv':
        keep_axes = [keep_axis, '3']
        prihdr['CTYPE1'] = '        '
        prihdr['CTYPE2'] = '        '
    else:
        keep_axes = ['1', '2']
        keys += ['CTYPE']

    for key in keys:
        if key + keep_axes[0] in header.keys():
            prihdr[key + '1'] = header[key + keep_axes[0]]
            prihdr.comments[key + '1'] = header.comments[key + keep_axes[0]]
        if key + keep_axes[1] in header.keys():
            prihdr[key + '2'] = header[key + keep_axes[1]]
            prihdr.comments[key + '2'] = header.comments[key + keep_axes[1]]

    for key_new, axis in zip(['CDELT1', 'CDELT2'], keep_axes):
        key = 'CD{a}_{a}'.format(a=axis)
        if key in header.keys():
            prihdr[key_new] = header[key]

    prihdr['AUTHOR'] = getpass.getuser()
    prihdr['ORIGIN'] = socket.gethostname()
    prihdr['DATE'] = (datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S'), '(GMT)')

    for comment in comments:
        prihdr['COMMENT'] = comment

    for key, val in dct_keys.items():
        prihdr[key] = val

    return prihdr


def get_moment_map(data, header, order=0, vel_unit=u.km/u.s):
    """Produce moment map.

    Parameters
    ----------
    data : numpy.ndarray
        Data of the FITS array.
    header : astropy.io.fits.Header
        Header of the FITS array.
    order : int
        Specify the moment map that should be produced: 0 - zeroth moment map, 1 - first moment map, 2 - second moment map.
    vel_unit : astropy.units.quantity.Quantity
        Valid unit to which the values of the spectral axis will be converted.

    Returns
    -------
    hdu : astropy.io.fits.HDUList
        Header/Data unit of the resulting FITS array.

    """
    wcs = WCS(header)

    #  convert from the velocity unit of the cube to the desired unit
    factor = wcs.wcs.cunit[2].to(vel_unit)
    wcs.wcs.cunit[2] = vel_unit
    wcs.wcs.cdelt[2] *= factor
    wcs.wcs.crval[2] *= factor

    header.update(wcs.to_header())

    bunit = u.Unit('')
    velocity_bin = wcs.wcs.cdelt[2]
    spectral_channels = get_spectral_axis(header=header, to_unit=vel_unit)
    moment_data = np.zeros(data.shape[1:])

    def moment0(spectrum):
        return velocity_bin * np.nansum(spectrum)

    def moment1(spectrum):
        nanmask = np.logical_not(np.isnan(spectrum))
        return np.nansum(spectral_channels[nanmask]) * spectrum[nanmask]

    def moment2(spectrum):
        nanmask = np.logical_not(np.isnan(spectrum))
        numerator = np.nansum(
            (spectral_channels[nanmask] - moment1(spectrum[nanmask]))**2 * spectrum[nanmask])
        denominator = np.nansum(spectrum[nanmask])
        return np.sqrt(numerator / denominator)

    if order == 0:
        moment_data = np.apply_along_axis(moment0, 0, data)
        bunit = u.Unit(header['BUNIT'])
    elif order == 1:
        moment_data = np.apply_along_axis(moment1, 0, data)
    elif order == 2:
        moment_data = np.apply_along_axis(moment2, 0, data)

    header = change_header(
        header, comments=['moment {} map'.format(order)],
        dct_keys={'BUNIT': (bunit * vel_unit).to_string()})

    return fits.PrimaryHDU(moment_data, header)


def moment_map(hdu=None, path_to_file=None, slice_params=None,
               path_to_output_file=None,
               apply_noise_threshold=False, snr=3, order=0,
               p_limit=0.02, pad_channels=5,
               vel_unit=u.km/u.s, path_to_noise_map=None,
               save=False, get_hdu=True, use_ncpus=None,
               restore_nans=False, nan_mask=None, dtype='float32'):
    """Create a moment map of the input data.

    The type of moment map can be specified with the 'order' keyword:
    0 - zeroth moment map
    1 - first moment map
    2 - second moment map

    Parameters
    ----------
    hdu : astropy.io.fits.HDUList
        Header/Data unit of the FITS cube.
    path_to_file : str
        Filepath to the FITS cube.
    slice_params : list
        List of slice parameters for cube, if only a subset of the data should be used.
    path_to_output_file : str
        Filepath to which FITS array of PV map gets saved.
    path_to_noise_map : str
        Filepath to the noise map of the input FITS cube.
    apply_noise_threshold : bool
        Default is `False`. If set to `True`, all data values with a signal-to-noise ratio below `snr` are set to zero.
    snr : int, float
        Signal-to-noise ratio for `apply_noise_threshold`.
    order : int
        Specify the moment map that should be produced: 0 - zeroth moment map, 1 - first moment map, 2 - second moment map.
    p_limit : float
        Maximum probability for consecutive positive/negative channels being
        due to chance.
    pad_channels : int
        Number of channels by which an interval (low, upp) gets extended on both sides, resulting in (low - pad_channels, upp + pad_channels).
    vel_unit : astropy.units.quantity.Quantity
        Valid unit to which the values of the spectral axis will be converted.
    path_to_noise_map : str
        Filepath to the noise map of the input FITS cube.
    save : bool
        Default is `False`. If set to `True`, the resulting FITS array is saved under 'path_to_output_file'.
    get_hdu : bool
        Default is `True`, which returns an astropy.io.fits.HDUList of the resulting PV map.
    use_ncpus : int
        Number of CPUs used in parallel processing. By default 75% of all CPUs on the machine are used.
    restore_nans : bool
        Default is `False`. If set to `True` restore NaN values present in the input FITS cube. [Nota bene: Not sure if the current implementation is correct!]
    nan_mask : numpy.array
        Array of boolean values specifying whether NaN values are present.


    """
    print('\ncreate a moment{} fits file from the cube'.format(order))

    check_if_value_is_none(restore_nans, nan_mask, 'restore_nans', 'nan_mask')
    check_if_all_values_are_none(hdu, path_to_file, 'hdu', 'path_to_file')

    if hdu is None:
        hdu = open_fits_file(path_to_file, get_hdu=True)

    if slice_params is not None:
        hdu = make_subcube(slice_params, hdu=hdu, get_hdu=True)
        slice_params = (slice_params[1], slice_params[2])
    else:
        slice_params = (slice(None), slice(None))

    data = hdu.data
    header = hdu.header
    # wcs = WCS(header)

    if apply_noise_threshold:
        data = clip_noise_below_threshold(data, snr=snr, slice_params=slice_params,
                                          path_to_noise_map=path_to_noise_map,
                                          p_limit=p_limit, pad_channels=pad_channels,
                                          use_ncpus=use_ncpus)

    hdu = get_moment_map(data, header, order=order, vel_unit=vel_unit)

    # TODO: check if this is correct
    if restore_nans:
        locations = list(
            itertools.product(
                range(hdu.data.shape[0]), range(hdu.data.shape[1])))
        for ypos, xpos in locations:
            if nan_mask[ypos, xpos]:
                hdu.data[ypos, xpos] = np.nan

    if save:
        suffix = 'mom{}_map'.format(order)
        if path_to_output_file is None:
            path_to_output_file = get_path_to_output_file(
                path_to_file, suffix=suffix,
                filename='moment{}_map.fits'.format(order))

        save_fits(hdu.data.astype(dtype), hdu.header, path_to_output_file,
                  verbose=True)

    if get_hdu:
        return hdu


def get_pv_map(data, header, sum_over_axis=1, slice_z=slice(None, None),
               vel_unit=u.km/u.s):
    """Produce a position-velocity map.

    Parameters
    ----------
    data : numpy.ndarray
        Data of the FITS array.
    header : astropy.io.fits.Header
        Header of the FITS array.
    sum_over_axis : int
        Specify spatial axis over which the data should be integrated for position-velocity map.
    slice_z : slice
        Specified slice parameters in case only subset of velocity axis should be used.
    vel_unit : astropy.units.quantity.Quantity
        Valid unit to which the values of the spectral axis will be converted.

    Returns
    -------
    hdu : astropy.io.fits.HDUList
        Header/Data unit of the resulting FITS array.

    """
    wcs = WCS(header)
    if wcs.wcs.cunit[2] == '':
        warnings.warn('Assuming m/s as spectral unit')
        wcs.wcs.cunit[2] = u.m/u.s
    factor = wcs.wcs.cunit[2].to(vel_unit)
    wcs.wcs.cunit[2] = vel_unit
    wcs.wcs.cdelt[2] *= factor
    wcs.wcs.crval[2] *= factor
    header.update(wcs.to_header())

    data = np.nansum(data[slice_z, :, :], sum_over_axis)

    if sum_over_axis == 1:
        keep_axis = '1'
    else:
        keep_axis = '2'

    header = change_header(header, format='pv', keep_axis=keep_axis)

    hdu = fits.PrimaryHDU(data=data, header=header)

    return hdu


def pv_map(path_to_file=None, hdu=None, slice_params=None,
           path_to_output_file=None, path_to_noise_map=None,
           apply_noise_threshold=False, snr=3, p_limit=0.02, pad_channels=5,
           sum_over_latitude=True, vel_unit=u.km/u.s,
           save=False, get_hdu=True, use_ncpus=None, dtype='float32'):
    """Create a position-velocity map of the input data.

    Parameters
    ----------
    path_to_file : str
        Filepath to the FITS cube.
    hdu : astropy.io.fits.HDUList
        Header/Data unit of the FITS cube.
    slice_params : list
        List of slice parameters for cube, if only a subset of the data should be used.
    path_to_output_file : str
        Filepath to which FITS array of PV map gets saved.
    path_to_noise_map : str
        Filepath to the noise map of the input FITS cube.
    apply_noise_threshold : bool
        Default is `False`. If set to `True`, all data values with a signal-to-noise ratio below `snr` are set to zero.
    snr : int, float
        Signal-to-noise ratio for `apply_noise_threshold`.
    p_limit : float
        Maximum probability for consecutive positive/negative channels being
        due to chance.
    pad_channels : int
        Number of channels by which an interval (low, upp) gets extended on both sides, resulting in (low - pad_channels, upp + pad_channels).
    sum_over_latitude : bool
        Default is `True`. Integrate over latitude axis (NAXIS2) for the PV map.
    vel_unit : astropy.units.quantity.Quantity
        Valid unit to which the values of the spectral axis will be converted.
    save : bool
        Default is `False`. If set to `True`, the resulting FITS array is saved under 'path_to_output_file'.
    get_hdu : bool
        Default is `True`, which returns an astropy.io.fits.HDUList of the resulting PV map.
    use_ncpus : int
        Number of CPUs used in parallel processing. By default 75% of all CPUs on the machine are used.

    """
    print('\ncreate a PV fits file from the cube')

    check_if_all_values_are_none(hdu, path_to_file, 'hdu', 'path_to_file')

    if hdu is None:
        hdu = open_fits_file(path_to_file, get_hdu=True)

    if slice_params is not None:
        data, header = make_subcube(slice_params, hdu=hdu)
        slice_params_pp = (slice_params[1], slice_params[2])
    else:
        slice_params_pp = (slice(None), slice(None))

    data = hdu.data
    header = hdu.header

    if apply_noise_threshold:
        data = clip_noise_below_threshold(
            data, snr=snr, slice_params=slice_params_pp,
            path_to_noise_map=path_to_noise_map, p_limit=p_limit,
            pad_channels=pad_channels, use_ncpus=use_ncpus)

    wcs = WCS(header)
    #  have to reverse the axis since we change between FITS and np standards
    if sum_over_latitude:
        sum_over_axis = wcs.wcs.naxis - wcs.wcs.lat - 1
    else:
        sum_over_axis = wcs.wcs.naxis - wcs.wcs.lng - 1

    if slice_params:
        slice_z = slice_params[0]

    hdu = get_pv_map(data, header, sum_over_axis=sum_over_axis,
                     slice_z=slice_z, vel_unit=vel_unit)
    data = hdu.data
    header = hdu.header

    if save:
        if path_to_output_file is None:
            path_to_output_file = get_path_to_output_file(path_to_file, suffix='_pv', filename='pv_map.fits')

        save_fits(hdu.data.astype(dtype), hdu.header, path_to_output_file,
                  verbose=True)

    if get_hdu:
        return hdu


def combine_fields(list_path_to_fields=[], ncols=3, nrows=2, save=False,
                   header=None, path_to_output_file=None, comments=[], verbose=True, dtype='float32'):
    """Combine FITS files to a mosaic by stacking them in the spatial coordinates.

    This will only yield a correct combined mosaic if the original mosaic was split in a similar way as obtained by the get_list_slice_params method

    Parameters
    ----------
    list_path_to_fields : list
        List of filepaths to the fields that should be mosaicked together.
    ncols : int
        Number of fields in the X direction.
    nrows : int
        Number of fields in the Y direction.
    save : bool
        Set to `True` if the resulting mosaicked file should be saved.
    header : astropy.io.fits.header.Header
        FITS header that will be used for the combined mosaic.
    path_to_output_file : str
        Filepath to which the combined mosaic gets saved if 'save' is set to `True`.
    comment : str
        Comment that will be written in the FITS header of the combined mosaic.
    verbose : bool
        Set to `False` if diagnostic messages should not be printed to the terminal.

    Returns
    -------
    data : numpy.ndarray
        Array of the combined mosaic.
    header : astropy.io.fits.header.Header
        FITS header of the combined mosaic.

    """
    check_if_value_is_none(save, path_to_output_file, 'save', 'path_to_output_file')

    combined_rows = []

    first = True
    for i, path_to_file in enumerate(list_path_to_fields):
        if first:
            combined_row = open_fits_file(
                path_to_file=path_to_file, get_header=False, check_wcs=False)
            axes = range(combined_row.ndim)
            axis_1 = axes[-1]
            axis_2 = axes[-2]
            first = False
        else:
            data = open_fits_file(
                path_to_file=path_to_file, get_header=False, check_wcs=False)
            combined_row = np.concatenate((combined_row, data), axis=axis_1)

        if i == 0 and header is None:
            header = open_fits_file(
                path_to_file=path_to_file, get_data=False, check_wcs=False)
        elif (i + 1) % ncols == 0:
            combined_rows.append(combined_row)
            first = True

    for combined_row in combined_rows:
        if first:
            data_combined = combined_row
            first = False
        else:
            data_combined = np.concatenate(
                (data_combined, combined_row), axis=axis_2)

    if comments:
        header = update_header(header, comments=comments)

    if save:
        save_fits(data_combined.astype(dtype), header, path_to_output_file,
                  verbose=verbose)

    return data, header
