# @Author: Manuel Riener <riener>
# @Date:   04-05-2020
# @Email:  riener@mpia-hd.mpg.de
# @Last modified by:   riener
# @Last modified time: 04-05-2020

import os

from gausspyplus.finalize import Finalize


def main():
    #  Initialize the 'Finalize' class and read in the parameter settings from 'gausspy+.ini'.
    finalize = Finalize(config_file='gausspy+.ini')

    #  The following lines will override the corresponding parameter settings defined in 'gausspy+.ini'.

    #  filepath to the pickled dictionary of the prepared data
    finalize.path_to_pickle_file = os.path.join(
        'decomposition_grs', 'gpy_prepared', 'grs-test_field.pickle')
    #  Filepath to the pickled dictionary of the decomposition results
    finalize.path_to_decomp_file = os.path.join(
        'decomposition_grs', 'gpy_decomposed',
        'grs-test_field_g+_fit_fin_sf-p2.pickle')

    finalize.finalize_dct()
    finalize.make_table()


if __name__ == "__main__":
    main()
