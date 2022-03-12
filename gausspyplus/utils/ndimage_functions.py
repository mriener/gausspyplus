from typing import Union

import numpy as np


def weighted_median(data):
    # TODO: add type hints
    # TODO: what happens at the borders?
    """Adapted from: https://gist.github.com/tinybike/d9ff1dad515b66cc0d87"""
    w_1 = 1
    w_2 = 1 / np.sqrt(2)
    weights = np.array([w_2, w_1, w_2, w_1, w_1, w_2, w_1, w_2])
    #  Skip if central spectrum was masked out.
    if np.isnan(central_value := data[4]):
        return 0
    data = np.delete(data, 4)
    #  Remove all neighbors that are NaN.
    mask = ~np.isnan(data)
    data = data[mask]
    weights = weights[mask]
    #  Skip if there are no valid available neighbors.
    if data.size == 0:
        return 0

    s_data, s_weights = map(np.array, zip(*sorted(zip(data, weights))))
    midpoint = 0.5 * sum(s_weights)
    if any(weights > midpoint):
        return (data[weights == np.max(weights)])[0]
    cs_weights = np.cumsum(s_weights)
    idx = np.where(cs_weights <= midpoint)[0][-1]
    return np.mean(s_data[idx:idx+2]) if cs_weights[idx] == midpoint else s_data[idx + 1]


def number_of_component_jumps(values: np.ndarray, max_jump_comps: int) -> int:
    """Determine the number of component jumps towards neighboring fits.

    A component jump occurs if the number of components is different by more than 'self.max_jump_comps' components.

    Parameters
    ----------
    values : Array of the number of fit components for a spectrum and its 8 immediate neighbors.

    Returns
    -------
    Number of component jumps.

    """
    if np.isnan(central_value := values[4]):
        return 0
    values = np.delete(values, 4)
    return sum(
        not np.isnan(value)
        and np.abs(central_value - value) > max_jump_comps
        for value in values
    )


def broad_components(values: np.ndarray,
                     fwhm_factor: float,
                     fwhm_separation: float,
                     broad_neighbor_fraction: float) -> Union[float, int]:
    """Check for the presence of broad fit components.

    This check is performed by comparing the broadest fit components of a spectrum with its 8 immediate neighbors.

    A fit component is defined as broad if its FWHM value exceeds the FWHM value of the largest fit components of
    more than 'self.broad_neighbor_fraction' of its neighbors by at least a factor of 'self.fwhm_factor'.

    In addition we impose that the minimum difference between the compared FWHM values has to exceed
    'self.fwhm_separation' to avoid flagging narrow components.

    Parameters
    ----------
    values : Array of FWHM values of the broadest fit components for a spectrum and its 8 immediate neighbors.

    Returns
    -------
    FWHM value in case of a broad fit component, 0 otherwise.

    """
    #  Skip if central spectrum was masked out.
    if np.isnan(central_value := values[4]):
        return 0
    values = np.delete(values, 4)
    #  Remove all neighbors that are NaN.
    values = values[~np.isnan(values)]
    #  Skip if there are no valid available neighbors.
    if values.size == 0:
        return 0
    #  Compare the largest FWHM value of the central spectrum with the largest FWHM values of its neighbors.
    counter = sum(
        not np.isnan(value)
        and central_value > value * fwhm_factor
        and (central_value - value) > fwhm_separation
        for value in values
    )
    return central_value if counter > values.size * broad_neighbor_fraction else 0
