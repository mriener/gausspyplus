import os
import sys
from pathlib import Path

ROOT = Path(os.path.realpath("__file__")).parents[1]
sys.path.append(str(ROOT))

from gausspyplus.training.training import GaussPyTraining


def main():
    #  Initialize the 'GaussPyTraining' class and read in the parameter settings from 'gausspy+.ini'.
    train = GaussPyTraining(config_file="gausspy+.ini")

    #  The following lines will override the corresponding parameter settings defined in 'gausspy+.ini'.

    #  Directory in which all files produced by GaussPy+ are saved.
    train.dirpath_gpy = "decomposition_grs"
    #  Filepath to the training set.
    train.path_to_training_set = os.path.join(
        train.dirpath_gpy,
        "gpy_training",
        "grs-test_field-training_set_100_spectra.pickle",
    )
    #  We select the two-phase-decomposition that uses two smoothing parameters.
    train.two_phase_decomposition = True
    #  Initial value for the first smoothing parameter.
    train.alpha1_initial = 2.0
    #  Initial value for the second smoothing parameter.
    train.alpha2_initial = 6.0
    train.verbose = False
    #  Start the training.
    train.training()


if __name__ == "__main__":
    main()
