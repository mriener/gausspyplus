"""Parallelization routines."""

import multiprocessing
import pickle
import signal
import sys
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

from gausspyplus.gausspy_py3.gp import GaussianDecomposer
from gausspyplus.utils.noise_estimation import determine_noise
from gausspyplus.prepare import GaussPyPrepare
from gausspyplus.spatial_fitting import SpatialFitting
from gausspyplus.training_set import GaussPyTrainingSet
from gausspyplus.finalize import Finalize

# ------------MULTIPROCESSING------------


#  With Python 3.8 the start method for multiprocessing defaults to 'spawn' for
#  MacOS systems. Here we change it back to 'fork' for compatibility reasons.
if sys.version_info[:2] >= (3, 8):
    multiprocessing.set_start_method('fork', force=True)


def init_worker_ts():
    """Worker initializer to ignore Keyboard interrupt."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)


def init_gausspy(*args):
    global agd_object, science_data_path, ilist, agd_data
    if args:
        agd_object, agd_data, ilist = args[0]
    else:
        agd_object, science_data_path, ilist = pickle.load(open('batchdecomp_temp.pickle', 'rb'), encoding='latin1')
        agd_data = pickle.load(open(science_data_path, 'rb'), encoding='latin1')
    if ilist is None:
        ilist = np.arange(len(agd_data['data_list']))


def init(mp_info):
    global mp_ilist, mp_data, mp_params
    mp_data, mp_params = mp_info
    mp_ilist = np.arange(mp_data if isinstance(mp_data, int) else len(mp_data))


def calculate_noise(i):
    xpos = mp_data[i][1]
    ypos = mp_data[i][0]
    spectrum = mp_params[0][:, ypos, xpos]
    return determine_noise(
        spectrum=spectrum,
        max_consecutive_channels=mp_params[1],
        pad_channels=mp_params[2],
        idx=i,
        average_rms=mp_params[3]
    )


def decompose_one(i):
    if agd_data['data_list'][i] is None:
        return None
    if 'signal_ranges' in list(agd_data.keys()):
        signal_ranges = agd_data['signal_ranges'][i]
        noise_spike_ranges = agd_data['noise_spike_ranges'][i]
    else:
        signal_ranges, noise_spike_ranges = None, None

    # TODO: what if idx keyword is missing or None?
    return GaussianDecomposer.decompose(
        self=agd_object,
        xdata=agd_data['x_values'],
        ydata=agd_data['data_list'][i],
        edata=agd_data['error'][i] * np.ones(len(agd_data['x_values'])),
        idx=agd_data['index'][i],
        signal_ranges=signal_ranges,
        noise_spike_ranges=noise_spike_ranges
    )


def refit_spectrum_1(i):
    return SpatialFitting.refit_spectrum_phase_1(self=mp_params[0], index=mp_data[i], i=i)


def refit_spectrum_2(i):
    return SpatialFitting.refit_spectrum_phase_2(self=mp_params[0], index=mp_data[i], i=i)


def calculate_noise_gpy(i):
    return GaussPyPrepare.calculate_rms_noise(self=mp_params[0], index=i)


def decompose_spectrum_ts(i):
    return GaussPyTrainingSet.decompose(self=mp_params[0], index=mp_data[i], i=i)


def make_table(i):
    return Finalize.get_table_rows(self=mp_params[0], idx=mp_data[i], j=i)


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
        for _ in tqdm(as_completed(futures), **kwargs):
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
    if use_ncpus is None:
        use_ncpus = int(ncpus*0.75)
    print(f'Using {use_ncpus} of {ncpus} cpus')
    try:
        if function == 'noise':
            results_list = parallel_process(array=mp_ilist, function=calculate_noise, n_jobs=use_ncpus)
        elif function == 'gausspy_decompose':
            results_list = parallel_process(array=ilist, function=decompose_one, n_jobs=use_ncpus)
        elif function == 'gpy_noise':
            results_list = parallel_process(array=mp_ilist, function=calculate_noise_gpy, n_jobs=use_ncpus)
        elif function == 'refit_phase_1':
            results_list = parallel_process(array=mp_ilist, function=refit_spectrum_1, n_jobs=use_ncpus)
        elif function == 'refit_phase_2':
            results_list = parallel_process(array=mp_ilist, function=refit_spectrum_2, n_jobs=use_ncpus)
        elif function == 'make_table':
            results_list = parallel_process(array=mp_ilist, function=make_table, n_jobs=use_ncpus)
    except KeyboardInterrupt:
        print("KeyboardInterrupt... quitting.")
        quit()
    return results_list


# TODO: alternative way for multiprocessing used for gausspy decomposition -> can be deleted
# def func(use_ncpus=None):
#     # Multiprocessing code
#     ncpus = multiprocessing.cpu_count()
#     if use_ncpus is None:
#         use_ncpus = int(0.75 * ncpus)
#     # p = multiprocessing.Pool(ncpus, init_worker)
#     print(('using {} out of {} cpus'.format(use_ncpus, ncpus)))
#     p = multiprocessing.Pool(use_ncpus, init_worker)
#     kwargs = {
#         'total': len(ilist),
#         'unit': 'it',
#         'unit_scale': True,
#         'leave': True
#     }
#     try:
#         # results_list = p.map(decompose_one, tqdm(ilist))
#         # results_list = tqdm(p.map(decompose_one, ilist), total=len(ilist))
#         results_list = tqdm(p.imap(decompose_one, ilist), **kwargs)
#
#     except KeyboardInterrupt:
#         print("KeyboardInterrupt... quitting.")
#         p.terminate()
#         quit()
#     p.close()
#     del p
#     return results_list


def func_ts(total, use_ncpus=None):
    # Multiprocessing code
    ncpus = multiprocessing.cpu_count()
    if use_ncpus is None:
        use_ncpus = int(0.75 * ncpus)
    print(f'using {use_ncpus} out of {ncpus} cpus')
    p = multiprocessing.Pool(processes=use_ncpus, initializer=init_worker_ts)

    try:
        results_list = []
        counter = 0
        pbar = tqdm(total=total)
        for result in p.imap_unordered(func=decompose_spectrum_ts, iterable=mp_ilist):
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
    pass
