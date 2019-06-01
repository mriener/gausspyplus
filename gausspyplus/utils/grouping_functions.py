"""Functions used for grouping."""

import numpy as np
import networkx


def get_neighbors(p, exclude_p=True, shape=None, nNeighbors=1,
                  get_indices=False, direction=None, get_mask=False):
    """Determine pixel coordinates of neighboring pixels.

    Includes also all pixels that neighbor diagonally.

    Parameters
    ----------
    p : tuple
        Gives the coordinates (y, x) of the central pixel
    exclude_p : boolean
        Whether or not to exclude the pixel with position p from the resulting list.
    shape : tuple
        Describes the dimensions of the total array (NAXIS2, NAXIS1).

    Returns
    -------
    neighbors: numpy.ndarray
        Contains all pixel coordinates of the neighboring pixels
        [[y1, x1], [y2, x2], ...]

    Adapted from:
    https://stackoverflow.com/questions/34905274/how-to-find-the-neighbors-of-a-cell-in-an-ndarray

    """
    ndim = len(p)
    n = nNeighbors*2 + 1

    # generate an (m, ndims) array containing all combinations of 0, 1, 2
    offset_idx = np.indices((n,) * ndim).reshape(ndim, -1).T

    # use these to index into np.array([-1, 0, 1]) to get offsets
    lst = list(range(-(nNeighbors), nNeighbors + 1))
    offsets = np.r_[lst].take(offset_idx)

    if direction == 'horizontal':
        indices = np.where(offsets[:, 0] == 0)
    elif direction == 'vertical':
        indices = np.where(offsets[:, 1] == 0)
    elif direction == 'diagonal_ul':
        indices = np.where(offsets[:, 0] == offsets[:, 1])
    elif direction == 'diagonal_ur':
        indices = np.where(offsets[:, 0] == -offsets[:, 1])

    if direction is not None:
        offsets = offsets[indices]

    # optional: exclude offsets of 0, 0, ..., 0 (i.e. p itself)
    if exclude_p:
        offsets = offsets[np.any(offsets, 1)]

    neighbours = p + offsets  # apply offsets to p

    # optional: exclude out-of-bounds indices
    if shape is not None:
        valid = np.all((neighbours < np.array(shape)) & (neighbours >= 0), axis=1)
        neighbours = neighbours[valid]

    if get_mask:
        return valid

    if get_indices:
        indices_neighbours = np.array([])
        for neighbour in neighbours:
            indices_neighbours = np.append(
                indices_neighbours, np.ravel_multi_index(neighbour, shape)).astype('int')
        return indices_neighbours

    return neighbours


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


def remove_components(lst, remove_indices):
    for idx, sublst in enumerate(lst):
        lst[idx] = [val for i, val in enumerate(sublst)
                    if i not in remove_indices]
    return lst
