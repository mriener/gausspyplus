# @Author: riener
# @Date:   2018-12-19T17:26:54+01:00
# @Filename: training.py
# @Last modified by:   riener
# @Last modified time: 2019-04-08T10:19:11+02:00

import os
import warnings

from .config_file import get_values_from_config_file
from .utils.output import format_warning
warnings.showwarning = format_warning


class GaussPyTraining(object):
    def __init__(self, config_file=''):
        self.path_to_training_set = None
        self.gpy_dirpath = None

        self.two_phase_decomposition = True
        self.snr = 3.
        self.alpha1_initial = None
        self.alpha2_initial = None
        self.snr_thresh = None
        self.snr2_thresh = None

        self.log_output = True
        self.verbose = True

        if config_file:
            get_values_from_config_file(
                self, config_file, config_key='training')

    def initialize(self):
        if self.path_to_training_set is None:
            raise Exception("Need to specify 'path_to_training_set'")

        self.dirname = os.path.dirname(self.path_to_training_set)
        if self.gpy_dirpath is None:
            self.gpy_dirpath = os.path.dirname(self.dirname)
        self.file = os.path.basename(self.path_to_training_set)
        self.filename, self.file_extension = os.path.splitext(self.file)

        if self.snr_thresh is None:
            self.snr_thresh = self.snr
        if self.snr2_thresh is None:
            self.snr2_thresh = self.snr
        if self.alpha1_initial is None:
            self.alpha1_initial = 3.
            warnings.warn(
                'No value for {a} supplied. Setting {a} to {b}.'.format(
                    a='alpha1_initial', b=self.alpha1_initial))
        if self.alpha2_initial is None:
            self.alpha2_initial = 6.
            warnings.warn(
                'No value for {a} supplied. Setting {a} to {b}.'.format(
                    a='alpha2_initial', b=self.alpha2_initial))

    def training(self):
        self.initialize()
        self.gausspy_train_alpha()

    def gausspy_train_alpha(self):
        from .gausspy_py3 import gp as gp

        g = gp.GaussianDecomposer()

        g.load_training_data(self.path_to_training_set)
        g.set('SNR_thresh', self.snr_thresh)
        g.set('SNR2_thresh', self.snr2_thresh)

        if self.log_output:
            log_output = {'dirname': self.gpy_dirpath, 'filename': self.filename}

        if self.two_phase_decomposition:
            g.set('phase', 'two')  # Set GaussPy parameters
            # Train AGD starting with initial guess for alpha
            g.train(alpha1_initial=self.alpha1_initial, alpha2_initial=self.alpha2_initial,
                    log_output=log_output)
        else:
            g.set('phase', 'one')
            g.train(alpha1_initial=self.alpha1_initial, log_output=log_output)
