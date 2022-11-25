import functools
import itertools
import os
import pickle
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union, Literal, List, Dict

import astropy
import numpy as np

import matplotlib

import matplotlib.pyplot as plt

from astropy import units as u
from tqdm import tqdm

from gausspyplus.decomposition.fit_quality_checks import goodness_of_fit
from gausspyplus.utils.gaussian_functions import (
    single_component_gaussian_model,
    multi_component_gaussian_model,
)
from gausspyplus.utils.spectral_cube_functions import get_spectral_axis, correct_header
from gausspyplus.definitions import Spectrum


def get_points_for_colormap(vmin, vmax, central_val=0.0):
    lower_interval = abs(central_val - vmin)
    upper_interval = abs(vmax - central_val)

    if lower_interval > upper_interval:
        start = 0.0
        stop = 0.5 + (upper_interval / lower_interval) * 0.5
    else:
        start = 0.5 - (lower_interval / upper_interval) * 0.5
        stop = 1.0
    return start, stop


def shiftedColorMap(cmap, start=0, midpoint=0.5, stop=1.0, name="shiftedcmap"):
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
    cdict = {"red": [], "green": [], "blue": [], "alpha": []}

    # regular index to compute the colors
    reg_index = np.linspace(start, stop, 257)

    # shifted index to match the data
    shift_index = np.hstack(
        [
            np.linspace(0.0, midpoint, 128, endpoint=False),
            np.linspace(midpoint, 1.0, 129, endpoint=True),
        ]
    )

    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)

        cdict["red"].append((si, r, r))
        cdict["green"].append((si, g, g))
        cdict["blue"].append((si, b, b))
        cdict["alpha"].append((si, a, a))

    newcmap = matplotlib.colors.LinearSegmentedColormap(name, cdict)
    plt.register_cmap(cmap=newcmap)

    return newcmap


# TODO: Use this also elsewhere and not only in this module?
def _pickle_load_file(pathToFile):
    with open(os.path.join(pathToFile), "rb") as pickle_file:
        if sys.version_info > (3, 0):
            return pickle.load(pickle_file, encoding="latin1")
        else:
            return pickle.load(pickle_file)


def _get_positions_from(
    data_or_pixel_range: dict, axis: Literal["x", "y"] = "x"
) -> np.ndarray:
    try:
        # only data has the key location; pixel_range has the keys 'x' and 'y'
        pos_min = min(
            data_or_pixel_range["location"], key=lambda pos: pos[int(axis == "x")]
        )
        pos_max = max(
            data_or_pixel_range["location"], key=lambda pos: pos[int(axis == "x")]
        )
    except KeyError:
        pos_min, pos_max = data_or_pixel_range[axis]
    return np.arange(pos_min, pos_max + 1)[:: (1 if axis == "x" else -1)]


def _get_grid_layout(data, subcube=False, pixel_range=None):
    if not subcube or (pixel_range is None):
        return None
    x_positions = (
        _get_positions_from(data, axis="x")
        if subcube
        else _get_positions_from(pixel_range, axis="x")
    )
    y_positions = (
        _get_positions_from(data, axis="y")
        if subcube
        else _get_positions_from(pixel_range, axis="y")
    )
    n_cols = len(x_positions)
    n_rows = len(y_positions)
    return [n_cols, n_rows]


def _data_contains_only_nans(data, index):
    if "nan_mask" not in data.keys():
        return False
    y_pos, x_pos = data["location"][index]
    return data["nan_mask"][:, y_pos, x_pos].all()


def _get_list_indices(
    data: dict,
    subcube=False,
    pixel_range: Optional[dict] = None,
    list_indices: Optional[list] = None,
    n_spectra: Optional[int] = None,
    random_seed: int = 111,
) -> list[int]:
    random.seed(random_seed)
    # TODO: incorporate the nan_mask in this scheme
    if subcube or (pixel_range is not None):
        x_positions = (
            _get_positions_from(data, axis="x")
            if subcube
            else _get_positions_from(pixel_range, axis="x")
        )
        y_positions = (
            _get_positions_from(data, axis="y")
            if subcube
            else _get_positions_from(pixel_range, axis="y")
        )
        list_indices = [
            data["location"].index(location)
            for location in itertools.product(y_positions, x_positions)
        ]
    elif (list_indices is None) and (n_spectra is not None):
        list_indices = []
        indices = list(range(len(data["data_list"])))
        for index in random.sample(indices, len(indices)):
            if not _data_contains_only_nans(data, index):
                list_indices.append(index)
                if len(list_indices) == n_spectra:
                    break
    elif list_indices is None:
        list_indices = np.arange(len(data["data_list"]))
    return [i for i in list_indices if data["data_list"][i] is not None]


def _plot_signal_ranges(ax, spectrum, spectral_channels, signal_ranges):
    if signal_ranges and spectrum.signal_intervals is not None:
        for low, upp in spectrum.signal_intervals:
            ax.axvspan(
                spectral_channels[low],
                spectral_channels[upp - 1],
                alpha=0.1,
                color="indianred",
            )


def _get_title(spectrum, idx):
    idx_string = (
        ""
        if spectrum.index is None or spectrum.index == idx
        else f" (Idx$_{{data}}$={spectrum.index})"
    )
    y_pos, x_pos = spectrum.position_yx
    loc_string = "" if y_pos is None else f", X={x_pos}, Y={y_pos}"
    ncomps_string = (
        ""
        if spectrum.n_fit_components is None
        else f", N$_{{comp}}$={spectrum.n_fit_components}"
    )
    rchi2_string = (
        ""
        if spectrum.reduced_chi2_value is None
        else f", $\\chi_{{red}}^{{2}}$={spectrum.reduced_chi2_value:.3f}"
    )
    return f"Idx={idx}{idx_string}{loc_string}{ncomps_string}{rchi2_string}"


def _get_path_to_plots(pathToDataPickle, path_to_decomp_pickle):
    return (
        Path(pathToDataPickle).parent
        if path_to_decomp_pickle is None
        else Path(path_to_decomp_pickle).parent
    )


def _plot_individual_components(ax, spectral_channels, channels, spectrum, gaussians):
    if not gaussians:
        return
    for amp, fwhm, mean in zip(
        spectrum.amplitude_values, spectrum.fwhm_values, spectrum.mean_values
    ):
        gauss = single_component_gaussian_model(amp, fwhm, mean, channels)
        ax.plot(spectral_channels, gauss, ls="solid", lw=1, color="orangered")


@dataclass
class Data:
    path_to_data_pickle: Union[str, Path]
    path_to_decomp_pickle: Optional[Union[str, Path]] = None
    training_set: bool = False
    vel_unit: u = u.km / u.s

    @functools.cached_property
    def data(self):
        return _pickle_load_file(self.path_to_data_pickle)

    @functools.cached_property
    def decomposition(self):
        return (
            _pickle_load_file(self.path_to_decomp_pickle)
            if self.path_to_decomp_pickle
            else None
        )

    @functools.cached_property
    def channels(self):
        return self.data["x_values"]

    @functools.cached_property
    def n_channels(self):
        return len(self.channels)

    @functools.cached_property
    def header(self):
        return (
            correct_header(self.data["header"])
            if "header" in self.data.keys()
            else None
        )

    @functools.cached_property
    def spectral_channels(self):
        return (
            self.channels
            if self.header is None
            else get_spectral_axis(header=self.header, to_unit=self.vel_unit)
        )

    def _update_spectrum_with_fit_results_from_decomposition(
        self, spectrum: Spectrum, idx
    ):
        return spectrum._replace(
            n_fit_components=len(self.decomposition["amplitudes_fit"][idx]),
            amplitude_values=self.decomposition["amplitudes_fit"][idx],
            mean_values=self.decomposition["means_fit"][idx],
            fwhm_values=self.decomposition["fwhms_fit"][idx],
            reduced_chi2_value=self.decomposition["best_fit_rchi2"][idx],
        )

    def _update_spectrum_with_fit_results_from_training_set(
        self, spectrum: Spectrum, idx
    ):
        return spectrum._replace(
            n_fit_components=len(self.data["amplitudes"][idx]),
            amplitude_values=self.data["amplitudes"][idx],
            mean_values=self.data["means"][idx],
            fwhm_values=self.data["fwhms"][idx],
            reduced_chi2_value=self.data["best_fit_rchi2"][idx],
        )

    def get_spectrum(self, idx):
        spectrum = Spectrum(
            index=self.data["index"][idx] if "index" in self.data.keys() else None,
            intensity_values=self.data["data_list"][idx],
            position_yx=(
                self.data["location"][idx]
                if "location" in self.data.keys()
                else (None, None)
            ),
            rms_noise=self.data["error"][idx][0],
            signal_intervals=(
                self.data["signal_ranges"][idx]
                if "signal_ranges" in self.data.keys()
                else None
            ),
            noise_spike_intervals=(
                self.data["noise_spike_intervals"][idx]
                if "noise_spike_intervals" in self.data.keys()
                else None
            ),
        )
        # TODO: homogenize this, so the same keys are used for training_set and decomposition
        #  Currently training_set uses 'fwhms', 'means' and 'amplitudes' but decomposition uses
        #  'fwhms_fit', 'means_fit', and 'amplitudes_fit'
        if self.decomposition:
            spectrum = self._update_spectrum_with_fit_results_from_decomposition(
                spectrum, idx
            )
        elif self.training_set:
            spectrum = self._update_spectrum_with_fit_results_from_training_set(
                spectrum, idx
            )
        return spectrum


@dataclass
class Figure:
    header: astropy.io.fits.header.Header
    n_cols: int = 5
    rowsize: float = 7.75
    max_rows_per_figure: int = 50
    rows_in_figure: int = 0
    dpi: int = 50
    n_spectra: Optional[int] = None
    n_channels: int = 0
    grid_layout: Optional[List[int]] = None
    suffix: str = ""
    subcube: bool = False
    pixel_range: Optional[dict] = None
    list_indices: Optional[list] = None
    plot_individual_gaussians: bool = True
    plot_residual: bool = True
    plot_signal_ranges: bool = True
    random_seed: int = 111
    count_figures: int = 0
    vel_unit: u = u.km / u.s

    @functools.cached_property
    def fontsize(self):
        fontsize_scaling = int(self.rowsize - 4)
        # TODO: check if that makes sense, fontsize seems to be scaled the same way for both options
        return 10 + fontsize_scaling if fontsize_scaling > 0 else 10 - fontsize_scaling

    @functools.cached_property
    def n_rows_total(self):
        return (
            -int(-self.n_spectra / self.n_cols)
            if self.grid_layout is None
            else self.grid_layout[1]
        )

    @functools.cached_property
    def colsize(self):
        return (
            round(self.rowsize * self.n_channels / 659, 2)
            if self.n_channels > 700
            else self.rowsize
        )

    @functools.cached_property
    def multiple_pdfs(self):
        return self.grid_layout is None and self.n_rows_total > self.max_rows_per_figure

    # TODO: change property name to be distinct from instance name
    @functools.cached_property
    def max_rows_per_figure(self):
        return min(self.n_rows_total, self.max_rows_per_figure)

    @functools.cached_property
    def x_label(self):
        return (
            "Channels"
            if self.header is None or "CTYPE3" not in self.header.keys()
            else f"{self.header['CTYPE3']} [{self.vel_unit}]"
        )

    @functools.cached_property
    def y_label(self):
        if self.header is None:
            return "Intensity"
        btype = self.header["BTYPE"] if "BTYPE" in self.header.keys() else "Intensity"
        bunit = f" [{self.header['BUNIT']}]" if "BUNIT" in self.header.keys() else ""
        return btype + bunit

    def prepare_figure(self):
        fig = plt.figure(
            figsize=(self.n_cols * self.colsize, self.rows_in_figure * self.rowsize)
        )
        fig.patch.set_facecolor("white")
        fig.patch.set_alpha(1.0)
        # fig.subplots_adjust(hspace=0.5)
        return fig

    def get_axis(self, idx_subplot, residual=False):
        row_i = (
            int(
                (
                    idx_subplot
                    - self.count_figures * (self.max_rows_per_figure * self.n_cols)
                )
                / self.n_cols
            )
            * 3
        )
        col_i = idx_subplot % self.n_cols
        return plt.subplot2grid(
            shape=(3 * self.rows_in_figure, self.n_cols),
            loc=(row_i + (2 if residual else 0), col_i),
            rowspan=(1 if residual else 2),
        )

    def add_figure_properties(self, ax, rms, spectral_channels, residual=False):
        # TODO: read labels automatically from header
        ax.set_xlim(spectral_channels[0], spectral_channels[-1])
        ax.set_xlabel(self.x_label, fontsize=self.fontsize)
        ax.set_ylabel(self.y_label, fontsize=self.fontsize)

        ax.tick_params(labelsize=self.fontsize - 2)

        ax.axhline(color="black", ls="solid", lw=0.5)
        ax.axhline(y=rms, color="red", ls="dotted", lw=0.5)
        ax.axhline(y=-rms, color="red", ls="dotted", lw=0.5)

        if not residual:
            ax.axhline(y=3 * rms, color="red", ls="dashed", lw=1)
        else:
            ax.set_title("Residual", fontsize=self.fontsize)

    def check_figure_size(self):
        if (self.max_rows_per_figure * self.rowsize * 100 > 2**16) or (
            self.n_cols * self.colsize * 100 > 2**16
        ):
            errorMessage = "Image size is too large. It must be less than 2^16 pixels in each direction. Restrict the number of columns or rows."
            raise Exception(errorMessage)

    def is_completed(self, idx_subplot):
        return ((idx_subplot + 1) % (self.max_rows_per_figure * self.n_cols) == 0) or (
            idx_subplot + 1
        ) == self.n_spectra


def plot_spectra(
    pathToDataPickle: Union[str, Path],
    path_to_plots: Optional[Union[str, Path]] = None,
    path_to_decomp_pickle: Optional[Union[str, Path]] = None,
    training_set: bool = False,
    n_cols: int = 5,
    rowsize: float = 7.75,
    max_rows_per_figure: int = 50,
    dpi: int = 50,
    n_spectra: Optional[int] = None,
    suffix: str = "",
    subcube: bool = False,
    pixel_range: Optional[dict] = None,
    list_indices: Optional[list] = None,
    gaussians: bool = True,
    residual: bool = True,
    signal_ranges: bool = True,
    random_seed: int = 111,
    vel_unit: u = u.km / u.s,
):

    print("\nPlotting...")

    path_to_plots = (
        Path(path_to_plots)
        if path_to_plots is not None
        else _get_path_to_plots(pathToDataPickle, path_to_decomp_pickle)
    )
    path_to_plots.mkdir(parents=True, exist_ok=True)

    data = Data(
        path_to_data_pickle=pathToDataPickle,
        path_to_decomp_pickle=path_to_decomp_pickle,
        training_set=training_set,
        vel_unit=vel_unit,
    )

    filename = (
        Path(path_to_decomp_pickle).stem
        if path_to_decomp_pickle
        else Path(pathToDataPickle).stem
    )

    list_indices = _get_list_indices(
        data.data,
        subcube=subcube,
        pixel_range=pixel_range,
        list_indices=list_indices,
        n_spectra=n_spectra,
        random_seed=random_seed,
    )

    figure = Figure(
        header=data.header,
        n_cols=n_cols,
        rowsize=rowsize,
        max_rows_per_figure=max_rows_per_figure,
        dpi=dpi,
        n_spectra=len(list_indices),
        n_channels=data.n_channels,
        grid_layout=_get_grid_layout(
            data.data, subcube=subcube, pixel_range=pixel_range
        ),
        suffix=suffix,
        subcube=subcube,
        pixel_range=pixel_range,
        list_indices=list_indices,
        plot_individual_gaussians=gaussians,
        plot_residual=residual,
        plot_signal_ranges=signal_ranges,
        random_seed=random_seed,
        vel_unit=vel_unit,
    )

    figure.rows_in_figure = min(figure.n_rows_total, figure.max_rows_per_figure)
    fig = figure.prepare_figure()

    pbar = tqdm(total=figure.n_spectra)

    for idx_subplot, idx_data in enumerate(list_indices):
        spectrum = data.get_spectrum(idx_data)
        ax = figure.get_axis(idx_subplot)
        ax.step(
            data.spectral_channels,
            spectrum.intensity_values,
            color="black",
            lw=0.5,
            where="mid",
        )

        if data.decomposition or data.training_set:
            modelled_spectrum = multi_component_gaussian_model(
                amps=spectrum.amplitude_values,
                fwhms=spectrum.fwhm_values,
                means=spectrum.mean_values,
                x=data.channels,
            )
            ax.plot(data.spectral_channels, modelled_spectrum, lw=2, color="orangered")

            _plot_individual_components(
                ax, data.spectral_channels, data.channels, spectrum, gaussians
            )

        _plot_signal_ranges(ax, spectrum, data.spectral_channels, signal_ranges)

        ax.set_title(_get_title(spectrum, idx_data), fontsize=figure.fontsize)

        figure.add_figure_properties(
            ax=ax, rms=spectrum.rms_noise, spectral_channels=data.spectral_channels
        )

        if residual and (data.decomposition or data.training_set):
            ax = figure.get_axis(idx_subplot, residual=True)
            ax.step(
                data.spectral_channels,
                spectrum.intensity_values - modelled_spectrum,
                color="black",
                lw=0.5,
                where="mid",
            )
            _plot_signal_ranges(ax, spectrum, data.spectral_channels, signal_ranges)

            figure.add_figure_properties(
                ax=ax,
                rms=spectrum.rms_noise,
                spectral_channels=data.spectral_channels,
                residual=True,
            )
        pbar.update(1)

        if figure.is_completed(idx_subplot):
            fig.tight_layout()
            suffix_for_multipage_plots = (
                f"_plots_part_{figure.count_figures + 1}"
                if figure.multiple_pdfs
                else ""
            )
            fig.savefig(
                path_to_plots
                / f"{filename}{suffix}_plots{suffix_for_multipage_plots}.pdf",
                dpi=dpi,
            )
            plt.close()

            #  close progress bar before print statement to avoid duplicate progress bars
            if pbar.n >= figure.n_spectra:
                pbar.close()
            print(
                "\n\033[92mSAVED FILE:\033[0m '{}' in '{}'".format(
                    filename, path_to_plots
                )
            )

            remaining_rows = (
                figure.n_rows_total - (figure.count_figures + 1) * max_rows_per_figure
            )
            if remaining_rows <= 0:
                break

            figure.count_figures += 1
            figure.rows_in_figure = min(remaining_rows, max_rows_per_figure)
            fig = figure.prepare_figure()
    plt.close()


def plot_fit_stages(
    data: np.ndarray,
    errors: np.ndarray,
    vel: np.ndarray,
    params_fit: List,
    ncomps_guess_final: int,
    improve_fitting: bool,
    perform_final_fit: bool,
    ncomps_guess_phase2: int,
    agd_phase1: Dict,
    phase,
    residuals,
    agd_phase2,
    best_fit_info: Optional[Dict] = None,
    params_fit_phase1: Optional[List] = None,
):
    #                       P L O T T I N G
    print(("params_fit:", params_fit))

    best_fit_final = multi_component_gaussian_model(
        *np.split(np.array(params_fit), 3), vel
    )

    if improve_fitting:
        rchi2 = best_fit_info["rchi2"]
        name = "GaussPy+"
    else:
        name = "GaussPy"
        #  TODO: define mask from signal_ranges
        rchi2 = goodness_of_fit(data, best_fit_final, errors, ncomps_guess_final)

    # Set up figure
    fig = plt.figure("AGD results", [16, 12])
    ax1 = fig.add_axes([0.1, 0.5, 0.4, 0.4])  # Initial guesses (alpha1)
    ax2 = fig.add_axes([0.5, 0.5, 0.4, 0.4])  # D2 fit to peaks(alpha2)
    ax3 = fig.add_axes([0.1, 0.1, 0.4, 0.4])  # Initial guesses (alpha2)
    ax4 = fig.add_axes([0.5, 0.1, 0.4, 0.4])  # Final fit

    # Decorations
    plt.figtext(0.52, 0.47, f"Final fit ({name}+)")
    if perform_final_fit:
        plt.figtext(0.52, 0.45, f"Reduced Chi2: {rchi2:3.3f}")
        plt.figtext(0.52, 0.43, f"N components: {ncomps_guess_final}")

    plt.figtext(0.12, 0.47, "Phase-two initial guess")
    plt.figtext(0.12, 0.45, f"N components: {ncomps_guess_phase2}")

    plt.figtext(0.12, 0.87, "Phase-one initial guess")
    plt.figtext(0.12, 0.85, f"N components: {agd_phase1['N_components']}")

    plt.figtext(0.52, 0.87, "Intermediate fit")

    # Initial Guesses (Panel 1)
    # -------------------------
    ax1.xaxis.tick_top()
    u2_scale = 1.0 / np.max(np.abs(agd_phase1["u2"])) * np.max(data) * 0.5
    ax1.axhline(color="black", linewidth=0.5)
    ax1.plot(vel, data, "-k")
    ax1.plot(vel, agd_phase1["u2"] * u2_scale, "-r")
    ax1.plot(vel, np.ones(len(vel)) * agd_phase1["thresh"], "--k")
    ax1.plot(vel, np.ones(len(vel)) * agd_phase1["thresh2"] * u2_scale, "--r")

    for amp, fwhm, mean in zip(
        agd_phase1["amps"], agd_phase1["fwhms"], agd_phase1["means"]
    ):
        ax1.plot(vel, single_component_gaussian_model(amp, fwhm, mean, vel), "-g")

    # Plot intermediate fit components (Panel 2)
    # ------------------------------------------
    ax2.xaxis.tick_top()
    ax2.axhline(color="black", linewidth=0.5)
    ax2.plot(vel, data, "-k")
    ax2.yaxis.tick_right()
    for amp, fwhm, mean in zip(*np.split(np.array(params_fit_phase1), 3)):
        ax2.plot(
            vel,
            single_component_gaussian_model(amp, fwhm, mean, vel),
            "-",
            color="blue",
        )

    # Residual spectrum (Panel 3)
    # -----------------------------
    if phase == "two":
        u2_phase2_scale = 1.0 / np.abs(agd_phase2["u2"]).max() * np.max(residuals) * 0.5
        ax3.axhline(color="black", linewidth=0.5)
        ax3.plot(vel, residuals, "-k")
        ax3.plot(vel, np.ones(len(vel)) * agd_phase2["thresh"], "--k")
        ax3.plot(
            vel, np.ones(len(vel)) * agd_phase2["thresh2"] * u2_phase2_scale, "--r"
        )
        ax3.plot(vel, agd_phase2["u2"] * u2_phase2_scale, "-r")

        for amp, fwhm, mean in zip(
            agd_phase2["amps"], agd_phase2["fwhms"], agd_phase2["means"]
        ):
            ax3.plot(vel, single_component_gaussian_model(amp, fwhm, mean, vel), "-g")

    # Plot best-fit model (Panel 4)
    # -----------------------------
    if perform_final_fit:
        ax4.yaxis.tick_right()
        ax4.axhline(color="black", linewidth=0.5)
        ax4.plot(vel, data, label="data", color="black")
        for amp, fwhm, mean in zip(*np.split(np.array(params_fit), 3)):
            ax4.plot(
                vel,
                single_component_gaussian_model(amp, fwhm, mean, vel),
                "--",
                color="orange",
            )
        ax4.plot(vel, best_fit_final, "-", color="orange", linewidth=2)

    plt.show()
