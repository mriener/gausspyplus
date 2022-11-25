"""Functions used for grouping."""
import collections
from typing import Tuple, Optional, Union

import numpy as np
import networkx
from networkx.algorithms.components.connected import connected_components

from gausspyplus.decomposition.gaussian_functions import upper_limit_for_amplitude


def get_neighbors(
    location: Tuple,
    exclude_location: bool = True,
    shape: Optional[Tuple] = None,
    n_neighbors: int = 1,
    direction: Optional[str] = None,
    return_indices: bool = True,
    return_coordinates: bool = False,
    return_mask: bool = False,
) -> Union[
    np.ndarray, Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]
]:
    """Determine pixel coordinates of neighboring pixels.

    Adapted from https://stackoverflow.com/a/34908879.

    :param location: Gives the coordinates `(y, x)` of the central pixel
    :param exclude_location: Whether to exclude `location` from the resulting list.
    :param shape: Describes the dimensions of the total array (NAXIS2, NAXIS1).
    :param n_neighbors: How many neighbors to determine in the specified directions.
    :param direction: Direction for which to determine neighbors. If no direction is given, all neighboring pixels
    (also diagonally) are considered.
    :param return_indices: Whether to return the indices of the determined neighbors for a flattened version of shape
    :param return_coordinates: Whether to return the pixel coordinates of the determined neighboring pixels in the
    form of `[[y1, x1], [y2, x2], ...]`
    :param return_mask: Whether to return the mask of the determined neighbors for a flattened version of shape
    :returns: A tuple of numpy arrays depending on which of the return conditions were selected as `True`. Arrays are
    returned in the order (indices, coordinates, mask).
    """

    ndim = len(location)

    # Generate an (m, ndims) array containing all combinations of 0 to n_neighbors
    offset_idx = np.indices((n_neighbors * 2 + 1,) * ndim).reshape(ndim, -1).T

    # use these to index into np.array([-1, 0, 1]) to get offsets
    offsets = np.r_[range(-n_neighbors, n_neighbors + 1)].take(offset_idx)

    if direction == "horizontal":
        offsets = offsets[np.flatnonzero(offsets[:, 0] == 0)]
    elif direction == "vertical":
        offsets = offsets[np.flatnonzero(offsets[:, 1] == 0)]
    elif direction == "diagonal_ul":
        offsets = offsets[np.flatnonzero(offsets[:, 0] == offsets[:, 1])]
    elif direction == "diagonal_ur":
        offsets = offsets[np.flatnonzero(offsets[:, 0] == -offsets[:, 1])]

    # Optional: exclude offsets of 0, 0, ..., 0 (i.e. the location itself)
    if exclude_location:
        offsets = offsets[np.any(offsets, 1)]

    # Apply offsets to locations
    neighbors = location + offsets

    # Optional: exclude out-of-bounds indices
    is_valid_neighbor = (
        np.all((neighbors < np.array(shape)) & (neighbors >= 0), axis=1)
        if shape is not None
        else np.ones(len(neighbors), dtype=bool)
    )
    neighbors = neighbors[is_valid_neighbor]

    return_tuple = ()
    if return_indices:
        return_tuple += (np.ravel_multi_index(neighbors.T, shape).astype(int),)
    if return_coordinates:
        return_tuple += (neighbors,)
    if return_mask:
        return_tuple += (is_valid_neighbor,)
    return return_tuple[0] if len(return_tuple) == 1 else return_tuple


def to_edges(l):
    """Treat 'l' as a Graph and return its edges.

    to_edges(['a', 'b', 'c', 'd']) -> [(a, b), (b, c), (c, d)]

    Credit: Jochen Ritzel
    https://stackoverflow.com/questions/4842613/merge-lists-that-share-common-elements
    """
    it = iter(l)
    last = next(it)

    for current in it:
        yield last, current
        last = current


def to_graph(l):
    """
    Credit: Jochen Ritzel
    https://stackoverflow.com/questions/4842613/merge-lists-that-share-common-elements
    """
    G = networkx.Graph()
    for part in l:
        # each sublist is a bunch of nodes
        G.add_nodes_from(part)
        # it also implies a number of edges:
        G.add_edges_from(to_edges(part))
    return G


def group_fit_solutions(
    amps_tot: np.ndarray,
    means_tot: np.ndarray,
    fwhms_tot: np.ndarray,
    split_fwhm: bool = True,
    mean_separation: Union[int, float] = 2.0,
    fwhm_separation: Union[int, float] = 4.0,
) -> collections.OrderedDict:
    """Grouping according to mean position values only or mean position values and FWHM values.

    Parameters
    ----------
    amps_tot : Array of amplitude values (sorted according to mean position).
    means_tot : Array of sorted mean position values.
    fwhms_tot : Array of FWHM values (sorted according to mean position).
    split_fwhm : Whether to group according to mean position and FWHM values ('True') or only according to mean
        position values ('False').

    Returns
    -------
    ordered_fit_components : Ordered dictionary containing the results of the grouping.

    """
    #  group with regards to mean positions only
    split_indices = np.flatnonzero(np.ediff1d(means_tot, to_begin=0) > mean_separation)
    split_means_tot = np.split(means_tot, split_indices)
    split_fwhms_tot = np.split(fwhms_tot, split_indices)
    split_amps_tot = np.split(amps_tot, split_indices)

    fit_components = {}

    for amps, fwhms, means in zip(split_amps_tot, split_fwhms_tot, split_means_tot):
        if (len(means) == 1) or not split_fwhm:
            key = f"{len(fit_components) + 1}"
            fit_components[key] = {"amps": amps, "means": means, "fwhms": fwhms}
            continue

        #  also group with regards to FWHM values

        lst_of_grouped_indices = []
        for i in range(len(means)):
            grouped_indices_means = np.where(
                (np.abs(means - means[i]) < mean_separation)
            )[0]
            grouped_indices_fwhms = np.where(
                (np.abs(fwhms - fwhms[i]) < fwhm_separation)
            )[0]
            ind = np.intersect1d(grouped_indices_means, grouped_indices_fwhms)
            lst_of_grouped_indices.append(list(ind))

        #  merge all sublists from lst_of_grouped_indices that share common indices

        G = to_graph(lst_of_grouped_indices)
        lst = list(connected_components(G))
        lst = [list(l) for l in lst]

        for sublst in lst:
            key = f"{len(fit_components) + 1}"
            fit_components[key] = {
                "amps": amps[sublst],
                "means": means[sublst],
                "fwhms": fwhms[sublst],
            }

    ordered_fit_components = collections.OrderedDict()
    for i, k in enumerate(
        sorted(
            fit_components,
            key=lambda k: len(fit_components[k]["amps"]),
            reverse=True,
        )
    ):
        ordered_fit_components[str(i + 1)] = fit_components[k]

    return ordered_fit_components


def determine_average_values(
    intensity_values: np.ndarray,
    fit_components: collections.OrderedDict,
    intensity_threshold: float,
    mean_separation: float = 2.0,
    constrain_fwhm: bool = False,
    fwhm_separation: float = 4.0,
) -> collections.OrderedDict:
    """Determine average values for fit components obtained by grouping.

    Parameters
    ----------
    intensity_values : Intensity values of spectrum to refit.
    rms : Root-mean-square noise value of the spectrum.
    fit_components : Ordered dictionary containing results of the grouping.

    Returns
    -------
    fit_components : Updated ordered dictionary containing average values for the fit components obtained via the
        grouping.

    """
    for key in fit_components.copy().keys():
        amps = np.array(fit_components[key]["amps"])
        #  TODO: also exclude all groups with two points?
        if len(amps) == 1:
            fit_components.pop(key)
            continue
        means = np.array(fit_components[key]["means"])
        fwhms = np.array(fit_components[key]["fwhms"])

        # TODO: take the median instead of the mean??
        amp_ini = np.mean(amps)
        mean_ini = np.mean(means)
        fwhm_ini = np.mean(fwhms)

        if (
            amp_max := upper_limit_for_amplitude(intensity_values, mean_ini, fwhm_ini)
        ) < intensity_threshold:
            fit_components.pop(key)
            continue

        #  determine fitting constraints for mean value
        lower_interval = max(abs(mean_ini - np.min(means)), mean_separation)
        mean_min = max(0, mean_ini - lower_interval)

        upper_interval = max(abs(mean_ini - np.max(means)), mean_separation)
        mean_max = mean_ini + upper_interval

        fwhm_min = 0
        fwhm_max = None

        if constrain_fwhm:
            #  determine fitting constraints for fwhm value
            lower_interval = max(abs(fwhm_ini - np.min(fwhms)), fwhm_separation)
            fwhm_min = max(0, fwhm_ini - lower_interval)

            upper_interval = max(abs(fwhm_ini - np.max(fwhms)), fwhm_separation)
            fwhm_max = fwhm_ini + upper_interval

        fit_components[key].update(
            {
                "amp_ini": amp_ini,
                "mean_ini": mean_ini,
                "fwhm_ini": fwhm_ini,
                "amp_bounds": [0.0, 1.1 * amp_max],
                "mean_bounds": [mean_min, mean_max],
                "fwhm_bounds": [fwhm_min, fwhm_max],
            }
        )
    return fit_components
