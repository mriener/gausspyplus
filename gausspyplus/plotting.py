import itertools
import os
import pickle
import random
import sys
from pathlib import Path
from typing import Optional, Union

import numpy as np

import matplotlib
# matplotlib.use('PDF', warn=False)
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

from astropy import units as u
from tqdm import tqdm

from gausspyplus.utils.gaussian_functions import gaussian, combined_gaussian
from gausspyplus.utils.spectral_cube_functions import get_spectral_axis, correct_header
from gausspyplus.definitions import Spectrum


def get_points_for_colormap(vmin, vmax, central_val=0.):
    lower_interval = abs(central_val - vmin)
    upper_interval = abs(vmax - central_val)

    if lower_interval > upper_interval:
        start = 0.
        stop = 0.5 + (upper_interval / lower_interval)*0.5
    else:
        start = 0.5 - (lower_interval / upper_interval)*0.5
        stop = 1.
    return start, stop


def shiftedColorMap(cmap, start=0, midpoint=0.5, stop=1.0, name='shiftedcmap'):
    """Function to offset the "center" of a colormap.

    Useful for data with a negative min and positive max and you want the middle of the colormap's dynamic range to
    be at zero.

    Input
    -----
      cmap : The matplotlib colormap to be altered
      start : Offset from lowest point in the colormap's range.
          Defaults to 0.0 (no lower offset). Should be between
          0.0 and `midpoint`.
      midpoint : The new center of the colormap. Defaults to
          0.5 (no shift). Should be between 0.0 and 1.0. In
          general, this should be  1 - vmax / (vmax + abs(vmin))
          For example if your data range from -15.0 to +5.0 and
          you want the center of the colormap at 0.0, `midpoint`
          should be set to  1 - 5/(5 + 15)) or 0.75
      stop : Offset from highest point in the colormap's range.
          Defaults to 1.0 (no upper offset). Should be between
          `midpoint` and 1.0.
    """
    cdict = {
        'red': [],
        'green': [],
        'blue': [],
        'alpha': []
    }

    # regular index to compute the colors
    reg_index = np.linspace(start, stop, 257)

    # shifted index to match the data
    shift_index = np.hstack([
        np.linspace(0.0, midpoint, 128, endpoint=False),
        np.linspace(midpoint, 1.0, 129, endpoint=True)
    ])

    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)

        cdict['red'].append((si, r, r))
        cdict['green'].append((si, g, g))
        cdict['blue'].append((si, b, b))
        cdict['alpha'].append((si, a, a))

    newcmap = matplotlib.colors.LinearSegmentedColormap(name, cdict)
    plt.register_cmap(cmap=newcmap)

    return newcmap


# TODO: Use this also elsewhere and not only in this module?
def _pickle_load_file(pathToFile):
    with open(os.path.join(pathToFile), "rb") as pickle_file:
        if sys.version_info > (3, 0):
            return pickle.load(pickle_file, encoding='latin1')
        else:
            return pickle.load(pickle_file)


def _get_x_positions_from_pixel_range(pixel_range: dict) -> np.ndarray:
    xmin, xmax = pixel_range['x']
    return np.arange(xmin, xmax + 1)


def _get_y_positions_from_pixel_range(pixel_range: dict) -> np.ndarray:
    ymin, ymax = pixel_range['y']
    return np.arange(ymin, ymax + 1)[::-1]


def _get_x_positions_from_data(data: dict) -> np.ndarray:
    xmin = min(data['location'], key=lambda pos: pos[1])
    xmax = max(data['location'], key=lambda pos: pos[1])
    return np.arange(xmin, xmax + 1)


def _get_y_positions_from_data(data: dict) -> np.ndarray:
    ymin = min(data['location'], key=lambda pos: pos[0])
    ymax = max(data['location'], key=lambda pos: pos[0])
    return np.arange(ymin, ymax + 1)[::-1]


def _get_grid_layout(data, subcube=False, pixel_range=None):
    if not subcube or (pixel_range is None):
        return None
    x_positions = _get_x_positions_from_data(data) if subcube else _get_x_positions_from_pixel_range(pixel_range)
    y_positions = _get_y_positions_from_data(data) if subcube else _get_y_positions_from_pixel_range(pixel_range)
    n_cols = len(x_positions)
    n_rows = len(y_positions)
    return [n_cols, n_rows]


def _data_contains_only_nans(data, index):
    if 'nan_mask' not in data.keys():
        return False
    y_pos, x_pos = data['location'][index]
    return data['nan_mask'][:, y_pos, x_pos].all()


def _get_list_indices(data: dict,
                      subcube=False,
                      pixel_range: Optional[dict] = None,
                      list_indices: Optional[list] = None,
                      n_spectra: Optional[int] = None,
                      random_seed: int = 111) -> list[int]:
    random.seed(random_seed)
    # TODO: incorporate the nan_mask in this scheme
    if subcube or (pixel_range is not None):
        x_positions = _get_x_positions_from_data(data) if subcube else _get_x_positions_from_pixel_range(pixel_range)
        y_positions = _get_y_positions_from_data(data) if subcube else _get_y_positions_from_pixel_range(pixel_range)
        list_indices = [data['location'].index(location) for location in itertools.product(y_positions, x_positions)]
    elif (list_indices is None) and (n_spectra is not None):
        list_indices = []
        indices = list(range(len(data['data_list'])))
        for index in random.sample(indices, len(indices)):
            if not _data_contains_only_nans(data, index):
                list_indices.append(index)
                if len(list_indices) == n_spectra:
                    break
    elif list_indices is None:
        list_indices = np.arange(len(data['data_list']))
    return [i for i in list_indices if data['data_list'][i] is not None]


def _get_figure_params(n_channels, n_spectra, n_cols, rowsize, max_rows_per_figure, grid_layout):
    colsize = round(rowsize*n_channels/659, 2) if n_channels > 700 else rowsize
    if grid_layout is None:
        n_rows_total = int(n_spectra / (n_cols))
        if n_spectra % n_cols != 0:
            n_rows_total += 1

        multiple_pdfs = True
        if n_rows_total < max_rows_per_figure:
            max_rows_per_figure = n_rows_total
            multiple_pdfs = False
    else:
        n_cols, n_rows_total = grid_layout
        max_rows_per_figure = n_rows_total
        multiple_pdfs = False

    if (max_rows_per_figure*rowsize*100 > 2**16) or (n_cols*colsize*100 > 2**16):
        errorMessage = \
            "Image size is too large. It must be less than 2^16 pixels in each direction. Restrict the number of columns or rows."
        raise Exception(errorMessage)

    return n_cols, n_rows_total, max_rows_per_figure, colsize, multiple_pdfs


def _xlabel_from_header(header, vel_unit):
    return 'Channels' if header is None or 'CTYPE3' not in header.keys() else f"{header['CTYPE3']} [{vel_unit}]"


def _ylabel_from_header(header):
    if header is None:
        return 'Intensity'
    btype = header['BTYPE'] if 'BTYPE' in header.keys() else 'Intensity'
    bunit = f" [{header['BUNIT']}]" if 'BUNIT' in header.keys() else ''
    return btype + bunit


def add_figure_properties(ax, rms, spectral_channels,
                          header=None, residual=False, fontsize=10, vel_unit=u.km/u.s):
    # TODO: read labels automatically from header
    ax.set_xlim(spectral_channels[0], spectral_channels[-1])
    ax.set_xlabel(_xlabel_from_header(header, vel_unit), fontsize=fontsize)
    ax.set_ylabel(_ylabel_from_header(header), fontsize=fontsize)

    ax.tick_params(labelsize=fontsize - 2)

    ax.axhline(color='black', ls='solid', lw=0.5)
    ax.axhline(y=rms, color='red', ls='dotted', lw=0.5)
    ax.axhline(y=-rms, color='red', ls='dotted', lw=0.5)

    if not residual:
        ax.axhline(y=3*rms, color='red', ls='dashed', lw=1)
    else:
        ax.set_title('Residual', fontsize=fontsize)


def _plot_signal_ranges(ax, spectrum, spectral_channels, signal_ranges):
    if signal_ranges and spectrum.signal_intervals is not None:
        for low, upp in spectrum.signal_intervals:
            ax.axvspan(spectral_channels[low], spectral_channels[upp - 1], alpha=0.1, color='indianred')


def _get_title(spectrum, idx):
    idx_string = ('' if spectrum.index is None or spectrum.index == idx
                  else f' (Idx$_{{data}}$={spectrum.index})')
    y_pos, x_pos = spectrum.position_yx
    loc_string = ('' if y_pos is None
                  else f', X={x_pos}, Y={y_pos}')
    ncomps_string = ('' if spectrum.n_fit_components is None
                     else f', N$_{{comp}}$={spectrum.n_fit_components}')
    rchi2_string = ('' if spectrum.reduced_chi2_value is None
                    else f', $\\chi_{{red}}^{{2}}$={spectrum.reduced_chi2_value:.3f}')
    return f'Idx={idx}{idx_string}{loc_string}{ncomps_string}{rchi2_string}'


def _scale_fontsize(rowsize, rowsize_scale=4):
    return (
        10 + int(rowsize - rowsize_scale)
        if rowsize >= rowsize_scale
        else 10 - int(rowsize - rowsize_scale)
    )


def _get_path_to_plots(pathToDataPickle, path_to_decomp_pickle):
    return Path(pathToDataPickle).parent if path_to_decomp_pickle is None else Path(path_to_decomp_pickle).parent


def _get_spectrum(data, idx):
    return Spectrum(
        index=data['index'][idx] if 'index' in data.keys() else None,
        intensity_values=data['data_list'][idx],
        position_yx=(data['location'][idx] if 'location' in data.keys() else (None, None)),
        rms_noise=data['error'][idx][0],
        signal_intervals=(data['signal_ranges'][idx] if 'signal_ranges' in data.keys() else None),
        noise_spike_intervals=(data['noise_spike_intervals'][idx] if 'noise_spike_intervals' in data.keys() else None),
    )


def _update_spectrum_with_fit_results_from_decomposition(spectrum: Spectrum, decomp, idx):
    return spectrum._replace(
        n_fit_components=len(decomp['amplitudes_fit'][idx]),
        amplitude_values=decomp['amplitudes_fit'][idx],
        mean_values=decomp['means_fit'][idx],
        fwhm_values=decomp['fwhms_fit'][idx],
        reduced_chi2_value=decomp['best_fit_rchi2'][idx]
    )


def _update_spectrum_with_fit_results_from_training_set(spectrum: Spectrum, data, idx):
    return spectrum._replace(
        n_fit_components=len(data['amplitudes'][idx]),
        amplitude_values=data['amplitudes'][idx],
        mean_values=data['means'][idx],
        fwhm_values=data['fwhms'][idx],
        reduced_chi2_value=data['best_fit_rchi2'][idx]
    )


def _plot_individual_components(ax, spectral_channels, channels, spectrum, gaussians):
    if not gaussians:
        return
    for amp, fwhm, mean in zip(spectrum.amplitude_values, spectrum.fwhm_values, spectrum.mean_values):
        gauss = gaussian(amp, fwhm, mean, channels)
        ax.plot(spectral_channels, gauss, ls='solid', lw=1, color='orangered')


def _get_ax_for_residual_plot(idx_subplot, count_figures, n_cols, max_rows_per_figure, rows_in_figure):
    row_i = int((idx_subplot - count_figures * (max_rows_per_figure * n_cols)) / n_cols) * 3 + 2
    col_i = idx_subplot % n_cols
    return plt.subplot2grid((3 * rows_in_figure, n_cols), (row_i, col_i))


def _prepare_figure(n_cols, n_rows, colsize, rowsize):
    fig = plt.figure(figsize=(n_cols * colsize, n_rows * rowsize))
    fig.patch.set_facecolor('white')
    fig.patch.set_alpha(1.0)
    # fig.subplots_adjust(hspace=0.5)
    return fig


def plot_spectra(pathToDataPickle: Union[str, Path],
                 path_to_plots: Optional[Union[str, Path]] = None,
                 path_to_decomp_pickle: Optional[Union[str, Path]] = None,
                 training_set: True = False,
                 n_cols: float = 5,
                 rowsize: float = 7.75,
                 max_rows_per_figure: int = 50,
                 dpi: int = 50,
                 n_spectra: Optional[int] = None,
                 suffix: int = '',
                 subcube: bool = False,
                 pixel_range: Optional[dict] = None,
                 list_indices: Optional[list] = None,
                 gaussians: bool = True,
                 residual: bool = True,
                 signal_ranges: bool = True,
                 random_seed: int = 111,
                 vel_unit=u.km/u.s):

    print("\nPlotting...")
    
    path_to_plots = Path(path_to_plots) if path_to_plots is not None else _get_path_to_plots(pathToDataPickle,
                                                                                             path_to_decomp_pickle)
    path_to_plots.mkdir(parents=True, exist_ok=True)

    filename = Path(pathToDataPickle).stem

    data = _pickle_load_file(pathToDataPickle)
    if (not training_set) and path_to_decomp_pickle is not None:
        decomp = _pickle_load_file(path_to_decomp_pickle)
        filename = Path(path_to_decomp_pickle).stem

    channels = data['x_values']
    n_channels = len(channels)

    if 'header' in data.keys():
        header = correct_header(data['header'])
        spectral_channels = get_spectral_axis(header=header, to_unit=vel_unit)
    else:
        header = None
        spectral_channels = channels

    grid_layout = _get_grid_layout(data, subcube=subcube, pixel_range=pixel_range)
    list_indices = _get_list_indices(data,
                                     subcube=subcube,
                                     pixel_range=pixel_range,
                                     list_indices=list_indices,
                                     n_spectra=n_spectra,
                                     random_seed=random_seed)
    n_spectra = len(list_indices)

    n_cols, n_rows_total, max_rows_per_figure, colsize, multiple_pdfs = _get_figure_params(
        n_channels, n_spectra, n_cols, rowsize, max_rows_per_figure, grid_layout)

    fontsize = _scale_fontsize(rowsize)

    fig = _prepare_figure(n_cols=n_cols,
                          n_rows=max_rows_per_figure,
                          colsize=colsize,
                          rowsize=rowsize)
    count_figures = 0
    rows_in_figure = min(n_rows_total, max_rows_per_figure)

    pbar = tqdm(total=n_spectra)

    for idx_subplot, idx_data in enumerate(list_indices):
        spectrum = _get_spectrum(data, idx_data)

        row_i = int((idx_subplot - (count_figures*max_rows_per_figure*n_cols)) / n_cols)*3
        col_i = idx_subplot % n_cols
        ax = plt.subplot2grid(shape=(3*rows_in_figure, n_cols), loc=(row_i, col_i), rowspan=2)

        ax.step(spectral_channels, spectrum.intensity_values, color='black', lw=0.5, where='mid')

        if path_to_decomp_pickle is not None or training_set:
            # TODO: homogenize this, so the same keys are used for training_set and decomp
            #  Currently training_set uses 'fwhms', 'means' and 'amplitudes' but decomposition uses
            #  'fwhms_fit', 'means_fit', and 'amplitudes_fit'
            if path_to_decomp_pickle is not None:
                spectrum = _update_spectrum_with_fit_results_from_decomposition(spectrum, decomp, idx_data)
            else:
                spectrum = _update_spectrum_with_fit_results_from_training_set(spectrum, data, idx_data)

            modelled_spectrum = combined_gaussian(amps=spectrum.amplitude_values,
                                                  fwhms=spectrum.fwhm_values,
                                                  means=spectrum.mean_values,
                                                  x=channels)
            ax.plot(spectral_channels, modelled_spectrum, lw=2, color='orangered')

            _plot_individual_components(ax, spectral_channels, channels, spectrum, gaussians)

        _plot_signal_ranges(ax, spectrum, spectral_channels, signal_ranges)

        ax.set_title(_get_title(spectrum, idx_data), fontsize=fontsize)

        add_figure_properties(ax, spectrum.rms_noise, spectral_channels,
                              header=header, fontsize=fontsize, vel_unit=vel_unit)

        if residual and (path_to_decomp_pickle is not None or training_set):
            ax = _get_ax_for_residual_plot(idx_subplot, count_figures, n_cols, max_rows_per_figure, rows_in_figure)
            ax.step(spectral_channels, spectrum.intensity_values - modelled_spectrum, color='black', lw=0.5, where='mid')
            _plot_signal_ranges(ax, spectrum, spectral_channels, signal_ranges)

            add_figure_properties(ax, spectrum.rms_noise, spectral_channels, header=header,
                                  residual=True, fontsize=fontsize, vel_unit=vel_unit)
        pbar.update(1)

        if ((idx_subplot + 1) % (max_rows_per_figure*n_cols) == 0) or ((idx_subplot + 1) == n_spectra):
            fig.tight_layout()
            suffix_for_multipage_plots = f'_plots_part_{count_figures + 1}' if multiple_pdfs else ''
            fig.savefig(path_to_plots / f'{filename}{suffix}_plots{suffix_for_multipage_plots}.pdf',
                        dpi=dpi,
                        overwrite=True)
            plt.close()

            #  close progress bar before print statement to avoid duplicate progress bars
            if pbar.n >= n_spectra:
                pbar.close()
            print("\n\033[92mSAVED FILE:\033[0m '{}' in '{}'".format(filename, path_to_plots))

            remaining_rows = n_rows_total - (count_figures + 1) * max_rows_per_figure
            if remaining_rows <= 0:
                break

            count_figures += 1
            rows_in_figure = min(remaining_rows, max_rows_per_figure)
            fig = _prepare_figure(n_cols=n_cols,
                                  n_rows=rows_in_figure,
                                  colsize=colsize,
                                  rowsize=rowsize)
    plt.close()
