from typing import Union, List

import numpy as np


def remove_elements_at_indices(
    array: Union[List, np.ndarray],
    indices: Union[int, List, np.ndarray],
    n_subarrays: int = 1,
) -> List:
    """Remove elements at specified indices from an array.

    If n_subarrays > 1, the input array is split into n_sublist arrays first and the indices are removed for each
    individual subarray. These subarrays are then concatenated again and returned as a single array.

    :param array: List or numpy array for which elements should be removed.
    :param indices: Remove all elements at these indices.
    :param n_subarrays: Whether the input array should be treated as a flattened version of n subarrays.
    :return: Updated list from with all elements at the supplied indices removed.
    """

    # TODO: Return this as an array and check all the code that uses this if it can deal with an array
    return np.concatenate(
        [np.delete(arr, indices) for arr in np.split(np.array(array), n_subarrays)]
    ).tolist()
