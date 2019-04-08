# @Author: Robert Lindner
# @Date: Nov 10, 2014
# @Filename: ioHDF5.py
# @Last modified by:   riener
# @Last modified time: 2019-03-16T12:00:32+01:00

import h5py
import numpy as np
import pickle
import sys


def flatten(nested_array):
    """
    Flatten list of arrays into one long array
    """
    flat = np.array([item for array in nested_array for item in array])
    return flat


def flatten_index(nested_array):
    """Flatten list of arrays into one long array."""
    index = [i for i in range(len(nested_array)) for item in nested_array[i]]
    return index


def flatten_keys(agd_data, keys):
    out = np.array([])
    for key in keys:
        out = np.concatenate([out, flatten(agd_data[key])])
    out = np.concatenate([out, flatten_index(agd_data[keys[0]])])
    return out


def dict_to_hdf5(hdf5_dataset, dic):
    """Write an array with the given namespace to an hdf5 dataset."""
    for key in list(dic.keys()):
        array = dic[key]['data']
        print(key, ', ', len(array))
        dset = hdf5_dataset.create_dataset(key, (len(array),), dtype=dic[key]['type'], compression='gzip')
        dset[:] = array


def toHDF5(data, filename):
    keys = list(data.keys())
    flat_dict = {}

    if 'data_list' in keys:
        print('Found data_list...')
        data_flat = np.concatenate(data['data_list'])
        data_lens = [len(data['data_list'][i]) for i in range(len(data['data_list']))]
        flat_dict.update({'data_flat': {'data': data_flat, 'type': 'float32'},
                          'data_lens': {'data': data_lens, 'type': 'int'}})

    if 'x_values' in keys:
        x_flat = np.concatenate(data['x_values'])
        flat_dict.update({'x_flat': {'data': x_flat, 'type': 'float32'}})

    if 'errors' in keys:
        errors_flat = np.concatenate(data['errors'])
        flat_dict.update({'errors_flat': {'data': errors_flat, 'type': 'float32'}})

    # Compress Gaussian components
    for tag in ['', '_fit', '_initial']:
        if ('amplitudes'+tag in keys) and ('fwhms'+tag in keys) and ('means'+tag in keys):
            params = flatten_keys(data, ['amplitudes'+tag, 'fwhms'+tag, 'means'+tag])
            flat_dict.update({'params'+tag: {'data': params, 'type': 'float32'}})

    print('Groups to be written to HDF5: ', list(flat_dict.keys()))

    # Create output HDF5 file
    print(filename)
    f = h5py.File(filename, 'w')
    dict_to_hdf5(f, flat_dict)
    f.close()


def reconstruct(flat_list, index_list, n_spectra):
    return [flat_list[index_list == i] for i in range(n_spectra)]


def fromHDF5(filename):
    print()
    print('Loading data from HDF5 file...')
    data = {}
    f = h5py.File(filename)
    fkeys = list(f.keys())
    print('Groups in HDF5 file: ', fkeys)

    data_flat = f['data_flat']
    errors_flat = f['errors_flat']
    x_flat = f['x_flat']
    data_lens = f['data_lens']

    n_spectra = len(data_lens)

    if 'data_flat' in fkeys:
        data['data_list'] = []
        for i in range(len(data_lens)):
            i1 = np.sum(data_lens[0:i])
            i2 = i1 + data_lens[i]
            data['data_list'] = data['data_list'] + [data_flat[i1:i2]]

    if 'x_flat' in fkeys:
        data['x_values'] = []
        for i in range(len(data_lens)):
            i1 = np.sum(data_lens[0:i])
            i2 = i1 + data_lens[i]
            data['x_values'] = data['x_values'] + [x_flat[i1:i2]]

    if 'errors_flat' in fkeys:
        data['errors'] = []
        for i in range(len(data_lens)):
            i1 = np.sum(data_lens[0:i])
            i2 = i1 + data_lens[i]
            data['errors'] = data['errors'] + [errors_flat[i1:i2]]

    # Inflate Gaussian parameter lists
    for tag in ['', '_fit', '_initial']:
        if 'params'+tag in fkeys:
            print('Group: params' + tag, ' found in HDF5 file.')
            n = len(f['params'+tag]) / 4
            index_flat = f['params'+tag][3*n:4*n]
            keys = ['amplitudes' + tag, 'fwhms' + tag, 'means' + tag]
            for i in range(len(keys)):
                print('>>>', keys[i])
                data[keys[i]] = reconstruct(f['params' + tag][i*n:n*(i+1)], index_flat, n_spectra)

    f.close()

    return data


if __name__ == '__main__':
    datapath = '/home/robert/git/paper_diffspec/Pickles/SPONGE_data_decomposed_twophase.pickle'
    data = pickle.load(open(datapath))
    toHDF5(data, 'agd_data.hdf5')

    amplitudes_fit = fromHDF5('agd_data.hdf5')
    print(len(amplitudes_fit))
    print(len(data['amplitudes_fit']))
    print(amplitudes_fit[18] == data['amplitudes_fit'][18])
