# @Author: riener
# @Date:   2018-12-19T17:26:54+01:00
# @Filename: parallel_processing.py
# @Last modified by:   riener
# @Last modified time: 2019-04-08T10:18:02+02:00
"""Parallelization routines."""

import multiprocessing
import signal
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

from .utils.noise_estimation import determine_noise
from .prepare import GaussPyPrepare
from .spatial_fitting import SpatialFitting
from .training_set import GaussPyTrainingSet

# ------------MULTIPROCESSING------------


def init_worker_ts():
    """Worker initializer to ignore Keyboard interrupt."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)


def init(mp_info):
    global mp_ilist, mp_data, mp_params
    mp_data, mp_params = mp_info
    mp_ilist = np.arange(len(mp_data))


def calculate_noise(i):
    xpos = mp_data[i][1]
    ypos = mp_data[i][0]
    spectrum = mp_params[0][:, ypos, xpos]
    result = determine_noise(spectrum, max_consecutive_channels=mp_params[1], pad_channels=mp_params[2], idx=i, average_rms=mp_params[3])
    return result


def refit_spectrum_1(i):
    result = SpatialFitting.refit_spectrum_phase_1(mp_params[0], mp_data[i], i)
    return result


def refit_spectrum_2(i):
    result = SpatialFitting.refit_spectrum_phase_2(mp_params[0], mp_data[i], i)
    return result


def calculate_noise_gpy(i):
    result = GaussPyPrepare.calculate_rms_noise(mp_params[0], mp_data[i], i)
    return result


def decompose_spectrum_ts(i):
    result = GaussPyTrainingSet.decompose(mp_params[0], mp_data[i], i)
    return result


def parallel_process(array, function, n_jobs=16, use_kwargs=False, front_num=3):
    """A parallel version of the map function with a progress bar.

    Credit: http://danshiebler.com/2016-09-14-parallel-progress-bar/

    Args:
        array (array-like): An array to iterate over.
        function (function): A python function to apply to the elements of array
        n_jobs (int, default=16): The number of cores to use
        use_kwargs (boolean, default=False): Whether to consider the elements of array as dictionaries of
            keyword arguments to function
        front_num (int, default=3): The number of iterations to run serially before kicking off the parallel job.
            Useful for catching bugs
    Returns:
        [function(array[0]), function(array[1]), ...]
    """
    # We run the first few iterations serially to catch bugs
    if front_num > 0:
        front = [function(**a) if use_kwargs else function(a) for a in array[:front_num]]
    # If we set n_jobs to 1, just run a list comprehension. This is useful for benchmarking and debugging.
    if n_jobs == 1:
        return front + [function(**a) if use_kwargs else function(a) for a in tqdm(array[front_num:])]
    # Assemble the workers
    with ProcessPoolExecutor(max_workers=n_jobs) as pool:
        # Pass the elements of array into function
        if use_kwargs:
            futures = [pool.submit(function, **a) for a in array[front_num:]]
        else:
            futures = [pool.submit(function, a) for a in array[front_num:]]
        kwargs = {
            'total': len(futures),
            'unit': 'it',
            'unit_scale': True,
            'leave': True
        }
        # Print out the progress as tasks complete
        for f in tqdm(as_completed(futures), **kwargs):
            pass
    out = []
    # Get the results from the futures.
    for i, future in tqdm(enumerate(futures)):
        try:
            out.append(future.result())
        except Exception as e:
            out.append(e)
    return front + out


def func(use_ncpus=None, function='noise'):
    # Multiprocessing code
    ncpus = multiprocessing.cpu_count()
    # p = multiprocessing.Pool(ncpus, init_worker)
    if use_ncpus is None:
        use_ncpus = int(ncpus*0.75)
    print('Using {} of {} cpus'.format(use_ncpus, ncpus))
    try:
        if function == 'noise':
            results_list = parallel_process(mp_ilist, calculate_noise, n_jobs=use_ncpus)
        elif function == 'gpy_noise':
            results_list = parallel_process(mp_ilist, calculate_noise_gpy, n_jobs=use_ncpus)
        elif function == 'refit_phase_1':
            results_list = parallel_process(mp_ilist, refit_spectrum_1, n_jobs=use_ncpus)
        elif function == 'refit_phase_2':
            results_list = parallel_process(mp_ilist, refit_spectrum_2, n_jobs=use_ncpus)
        # results_list = p.map(determine_distance, tqdm(ilist))
    except KeyboardInterrupt:
        print("KeyboardInterrupt... quitting.")
        quit()
    return results_list


def func_ts(total, use_ncpus=None):
    # Multiprocessing code
    ncpus = multiprocessing.cpu_count()
    if use_ncpus is None:
        use_ncpus = int(0.75 * ncpus)
    print('using {} out of {} cpus'.format(use_ncpus, ncpus))
    p = multiprocessing.Pool(use_ncpus, init_worker_ts)

    try:
        results_list = []
        counter = 0
        pbar = tqdm(total=total)
        for i, result in enumerate(
                p.imap_unordered(decompose_spectrum_ts, mp_ilist)):
            if result is not None:
                counter += 1
                pbar.update(1)
                results_list.append(result)
            if counter == total:
                break
        pbar.close()

    except KeyboardInterrupt:
        print("KeyboardInterrupt... quitting.")
        p.terminate()
        quit()
    p.close()
    del p
    return results_list


if __name__ == "__main__":
    ''
