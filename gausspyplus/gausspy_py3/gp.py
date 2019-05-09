# @Author: Robert Lindner
# @Date:   2018-12-19T17:30:53+01:00
# @Filename: gp.py
# @Last modified by:   riener
# @Last modified time: 2019-03-29T15:52:07+01:00

import os
import pickle
import multiprocessing
import signal

from matplotlib import pyplot as plt
import numpy as np

from . import AGD_decomposer
from . import gradient_descent
from . import ioHDF5


class GaussianDecomposer(object):
    def __init__(self, filename=None, phase='one'):
        if filename:
            temp = pickle.load(open(filename, 'rb'), encoding='latin1')
            self.p = temp.p
        else:
            self.p = {'alpha1': None, 'alpha2': None, 'training_results': None,
                      'improve_fitting_dict': None, 'use_ncpus': None,
                      'phase': 'one', 'SNR2_thresh': 5., 'SNR_thresh': 5.,
                      'deblend': True, 'mode': 'conv', 'BLFrac': 0.1, 'verbose': False, 'plot': False,
                      'perform_final_fit': True}

    def load_training_data(self, filename):
        self.p['training_data'] = pickle.load(open(filename, 'rb'), encoding='latin1')

    def load_hdf5_data(self, filename):
        return ioHDF5.fromHDF5(filename)

    def dump_hdf5_data(self, data, filename):
        ioHDF5.toHDF5(data, filename)

    def train(self, alpha1_initial=None, alpha2_initial=None, plot=False,
              verbose=False, mode='conv', learning_rate=0.9, eps=0.25, MAD=0.1,
              logger=False):
        """Solve for optimal values of alpha1 (and alpha2) using training data."""
        if (((self.p['phase'] == 'one') and (not alpha1_initial)) or
           ((self.p['phase'] == 'two') and ((not alpha1_initial) or (not alpha1_initial)))):
            print('Must choose initial guesses.')
            print('e.g., train(alpha1_initial=1.0, alpha2_initial=100.)')
            return
        if not self.p['training_data']:
            print('Must first load training data.')
            print('e.g., load_training_data("training_data.pickle")')
            return
        print('Training...')

        self.p['alpha1'], self.p['alpha2'], self.p['training_results'] =\
            gradient_descent.train(alpha1_initial=alpha1_initial,
                                   alpha2_initial=alpha2_initial,
                                   training_data=self.p['training_data'],
                                   phase=self.p['phase'],
                                   SNR_thresh=self.p['SNR_thresh'],
                                   SNR2_thresh=self.p['SNR2_thresh'],
                                   plot=plot, eps=eps,
                                   verbose=verbose, mode=mode,
                                   learning_rate=learning_rate, MAD=MAD,
                                   logger=logger)

    def decompose(self, xdata, ydata, edata, idx=None, signal_ranges=None,
                  noise_spike_ranges=None):
        """Decompose a single spectrum using current parameters."""
        if ((self.p['phase'] == 'one') and (not self.p['alpha1'])):
            print('phase = one, and alpha1 is unset')
            return

        if (self.p['phase'] == 'two') and ((not self.p['alpha1']) or (not self.p['alpha2'])):
            print('phase = two, and either alpha1 or alpha2 is unset')
            return

        if self.p['mode'] != 'conv':
            a1 = 10**self.p['alpha1']
            a2 = 10**self.p['alpha2'] if self.p['phase'] == 'two' else None
        else:
            a1 = self.p['alpha1']
            a2 = self.p['alpha2'] if self.p['phase'] == 'two' else None

        status, results = AGD_decomposer.AGD(
            xdata, ydata, edata, idx=idx, signal_ranges=signal_ranges,
            noise_spike_ranges=noise_spike_ranges, alpha1=a1, alpha2=a2,
            improve_fitting_dict=self.p['improve_fitting_dict'],
            phase=self.p['phase'], mode=self.p['mode'],
            verbose=self.p['verbose'], SNR_thresh=self.p['SNR_thresh'],
            BLFrac=self.p['BLFrac'], SNR2_thresh=self.p['SNR2_thresh'],
            deblend=self.p['deblend'], plot=self.p['plot'], perform_final_fit=self.p['perform_final_fit'])
        return results

    def status(self):
        """Return current values of parameters."""
        print('Current Parameters:')
        print('---' * 10)
        for index, key in enumerate(self.p):
            if key in ['data_list', 'errors', 'x_values', 'amplitudes', 'fwhms',
                       'means', 'amplitudes_fit', 'fwhms_fit', 'means_fit']:
                print('len({0}) = {1}'.format(key, len(self.p[key])))
            else:
                print(key, ' = ', self.p[key])

    def set(self, key, value):
        if key in self.p:
            self.p[key] = value
        else:
            print('Given key does not exist.')

    def save_state(self, filename, clobber=False):
        """Save the current decomposer object, and all associated parameters to a python pickle file."""
        if os.path.isfile(filename):
            if clobber:
                os.remove(filename)
            else:
                print('File exists: ', filename)
                return
        pickle.dump(self, open(filename, 'wb'), protocol=2)

    def batch_decomposition(self, *args, ilist=None, dct=None):
        """Science data should be AGD format ilist is either None or an integer list."""
        from . import batch_decomposition

        if args:
            science_data_path = args[0]
            # Dump information to hard drive to allow multiprocessing
            pickle.dump([self, science_data_path, ilist],
                        open('batchdecomp_temp.pickle', 'wb'), protocol=2)
            batch_decomposition.init()
            result_list = batch_decomposition.func(
                use_ncpus=self.p['use_ncpus'])
        else:
            #  if only a single spectrum is decomposed
            batch_decomposition.init([self, dct, ilist])
            result_list = [batch_decomposition.decompose_one(0)]

        new_keys = ['index_fit', 'amplitudes_fit', 'fwhms_fit', 'means_fit',
                    'index_initial', 'amplitudes_initial', 'fwhms_initial', 'means_initial',
                    'amplitudes_fit_err', 'fwhms_fit_err', 'means_fit_err', 'best_fit_rchi2',
                    'best_fit_aicc', 'N_components', 'N_neg_res_peak', 'N_blended', 'log_gplus', 'pvalue', 'quality_control']

        output_data = dict((key, []) for key in new_keys)

        for i, result in enumerate(result_list):
            # if not isinstance(item, dict):
            #     print(i, item)
            if result is not None:
                # Save best-fit parameters
                idx = result['index']
                ncomps = result['N_components']
                amps = result['best_fit_parameters'][0:ncomps] if ncomps > 0 else []
                fwhms = result['best_fit_parameters'][ncomps:2*ncomps] if ncomps > 0 else []
                offsets = result['best_fit_parameters'][2*ncomps:3*ncomps] if ncomps > 0 else []

                # TODO: rework this back to just i?
                output_data['index_fit'].append(idx)
                output_data['N_components'].append(ncomps)
                output_data['amplitudes_fit'].append(amps)
                output_data['fwhms_fit'].append(fwhms)
                output_data['means_fit'].append(offsets)

                # Save initial guesses if something was found
                ncomps_initial = int(len(result['initial_parameters']) / 3)
                amps_initial = result['initial_parameters'][0:ncomps_initial] if ncomps_initial > 0 else []
                fwhms_initial = result['initial_parameters'][ncomps_initial:2*ncomps_initial] if ncomps_initial > 0 else []
                offsets_initial = result['initial_parameters'][2*ncomps_initial:3*ncomps_initial] if ncomps_initial > 0 else []

                output_data['means_initial'].append(offsets_initial)
                output_data['fwhms_initial'].append(fwhms_initial)
                output_data['amplitudes_initial'].append(amps_initial)
                output_data['index_initial'].append([i for j in range(ncomps_initial)])

                # Final fit errors
                amps_err = result['best_fit_errors'][0:ncomps] if ncomps > 0 else []
                fwhms_err = result['best_fit_errors'][ncomps:2*ncomps] if ncomps > 0 else []
                offsets_err = result['best_fit_errors'][2*ncomps:3*ncomps] if ncomps > 0 else []

                output_data['amplitudes_fit_err'].append(amps_err)
                output_data['fwhms_fit_err'].append(fwhms_err)
                output_data['means_fit_err'].append(offsets_err)

                for key in ['best_fit_rchi2', 'best_fit_aicc', 'N_neg_res_peak',
                            'N_blended', 'log_gplus', 'pvalue',
                            'quality_control']:
                    output_data[key].append(
                        result[key] if key in result else None)

                # # Final goodness of fit criteria
                # rchi2 = result['rchi2'] if 'rchi2' in result else None
                # aicc = result['aicc'] if 'aicc' in result else None
                # Nresidual = result['N_neg_res_peak'] if 'N_neg_res_peak' in result else None
                # Nblended = result['N_blended'] if 'N_blended' in result else None
                # log_gplus = result['log_gplus'] if 'log_gplus' in result else None
                # pvalue = result['pvalue'] if 'pvalue' in result else None
                # quality_control = result['quality_control'] if 'quality_control' in result else None
                #
                # output_data['best_fit_rchi2'].append(rchi2)
                # output_data['best_fit_aicc'].append(aicc)
                # output_data['N_neg_res_peak'].append(Nresidual)
                # output_data['N_blended'].append(Nblended)
                # output_data['log_gplus'].append(log_gplus)
                # output_data['pvalue'].append(pvalue)
                # output_data['quality_control'].append(quality_control)
            else:
                for key in new_keys:
                    output_data[key].append(None)
                output_data['index_fit'][i] = i

        return output_data

    def plot_components(self, data, index, xlabel='x', ylabel='y', xlim=None,
                        ylim=None, guesses=False, plot_true=False):
        # Extract info from data (must contain 'fit' categories)
        # x = data['x_values'][index]
        x = data['x_values'][0]
        y = data['data_list'][index]

        fwhms = data['fwhms_fit'][index]
        amps = data['amplitudes_fit'][index]
        means = data['means_fit'][index]

        fwhms_guess = data['fwhms_initial'][index]
        amps_guess = data['amplitudes_initial'][index]
        means_guess = data['means_initial'][index]

        ncomps = len(amps)

        if 'amplitudes' in data:
            fwhms_true = data['fwhms'][index]
            amps_true = data['amplitudes'][index]
            means_true = data['means'][index]

        plt.plot(x, y, '-k', label='data', lw=1.5)

        # Plot fitted, components
        sum_fit = x * 0.
        for i, amp in enumerate(amps):
            model = amp * np.exp(-(x-means[i])**2/2./(fwhms[i]/2.355)**2)
            model_guess = amps_guess[i] * np.exp(-(x-means_guess[i])**2/2./(fwhms_guess[i]/2.355)**2)
            sum_fit = sum_fit + model
            plt.plot(x, model, '-g', lw=0.5)
            if guesses:
                plt.plot(x, model_guess, '--g', lw=1)
            plt.xlabel(xlabel, fontsize=16)
            plt.ylabel(ylabel, fontsize=16)
            if xlim:
                plt.xlim(*xlim)
            if ylim:
                plt.ylim(*ylim)
        plt.plot(x, sum_fit, '-g', lw=1.0, label='Fit')

        # If available, plot True components
        sum_true = x * 0.
        if ('amplitudes' in data) and plot_true:
            for i, amp in enumerate(amps_true):
                model_true = amps_true[i] * np.exp(-(x-means_true[i])**2/2./(fwhms_true[i]/2.355)**2)
                sum_true = sum_true + model_true
                plt.plot(x, model_true, '-r', lw=0.5)
            plt.plot(x, sum_true, '-r', lw=1.0, label='True')

        plt.title('index = {0}, ncomps = {1}'.format(index, ncomps), fontsize=16)
        plt.legend(loc=0)
        plt.legend(loc=1)
        plt.show()
