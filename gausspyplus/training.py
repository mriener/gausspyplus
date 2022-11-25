# TODO: Fix the root cause for the following error message (caused by GaussPy?):
#  WARNING: VisibleDeprecationWarning: Creating an ndarray from ragged nested sequences (which is a list-or-tuple of
#  lists-or-tuples-or ndarrays with different lengths or shapes) is deprecated. If you meant to do this, you must
#  specify 'dtype=object' when creating the ndarray.
import functools
from pathlib import Path

from gausspyplus.config_file import get_values_from_config_file
from gausspyplus.decomposition import gp as gp
from gausspyplus.definitions.checks import BaseChecks
from gausspyplus.utils.output import (
    set_up_logger,
    say,
    make_pretty_header,
)
from gausspyplus.definitions import SettingsDefault, SettingsTraining


class GaussPyTraining(SettingsDefault, SettingsTraining, BaseChecks):
    def __init__(self, config_file=""):
        self.path_to_training_set = None
        self.gpy_dirpath = None

        if config_file:
            get_values_from_config_file(self, config_file, config_key="training")

    @functools.cached_property
    def logger(self):
        if not self.log_output:
            return False
        return set_up_logger(
            parentDirname=(
                self.gpy_dirpath or Path(self.path_to_training_set).parents[1]
            ),
            filename=Path(self.path_to_training_set).stem,
            method="g+_training",
        )

    def training(self):
        self.raise_exception_if_attribute_is_none("path_to_training_set")
        self.set_attribute_if_none("alpha1_initial", 3.0, show_warning=True)
        self.set_attribute_if_none("alpha2_initial", 6.0, show_warning=True)
        say(
            message=make_pretty_header("GaussPy training"),
            verbose=self.verbose,
            logger=self.logger,
        )
        self._gausspy_train_alpha()

    def _gausspy_train_alpha(self):
        say(f"Using training set: {self.path_to_training_set}", logger=self.logger)

        decomposer = gp.GaussianDecomposer()

        decomposer.load_training_data(self.path_to_training_set)
        decomposer.set(
            "SNR_thresh", self.snr if self.snr_thresh is None else self.snr_thresh
        )
        decomposer.set(
            "SNR2_thresh", self.snr if self.snr2_thresh is None else self.snr2_thresh
        )

        if self.two_phase_decomposition:
            decomposer.set("phase", "two")  # Set GaussPy parameters
            # Train AGD starting with initial guess for alpha
            decomposer.train(
                alpha1_initial=self.alpha1_initial,
                alpha2_initial=self.alpha2_initial,
                logger=self.logger,
            )
        else:
            decomposer.set("phase", "one")
            decomposer.train(alpha1_initial=self.alpha1_initial, logger=self.logger)
