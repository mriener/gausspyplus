import os
import sys
from pathlib import Path

ROOT = Path(os.path.realpath("__file__")).parents[1]
sys.path.append(str(ROOT))

from gausspyplus.processing.finalize import Finalize


def main():
    #  Initialize the 'Finalize' class and read in the parameter settings from 'gausspy+.ini'.
    finalize = Finalize(config_file="gausspy+.ini")

    #  The following lines will override the corresponding parameter settings defined in 'gausspy+.ini'.

    #  filepath to the pickled dictionary of the prepared data
    finalize.path_to_pickle_file = Path(
        "decomposition_grs", "gpy_prepared", "grs-test_field.pickle"
    )
    #  Filepath to the pickled dictionary of the decomposition results
    finalize.path_to_decomp_file = Path(
        "decomposition_grs", "gpy_decomposed", "grs-test_field_g+_fit_fin_sf-p2.pickle"
    )

    finalize.finalize_dct()
    finalize.make_table()


if __name__ == "__main__":
    main()
