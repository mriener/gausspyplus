from typing import Tuple, List, Optional


def get_flags(n_old: int, n_new: int) -> Tuple[int, int]:
    """Check how the refit affected the number of blended or negative residual features.

    This check will only be performed if the 'self.flag_blended=True' or 'self.flag_neg_res_peak=True'.

    Parameters
    ----------
    fit_results : Dictionary containing the new best fit results after the refit attempt.
    index : Index ('index_fit' keyword) of the spectrum that gets/was refit.
    key : Dictionary keys, either 'N_blended' or 'N_neg_res_peak'.
    flag : User-selected flag criterion, either 'self.flag_blended', or 'self.flag_neg_res_peak'
    updated_fit_results : Only used in phase 2 of the spatially coherent refitting, in case the best fit solution was
        already updated in a previous iteration.

    Returns
    -------
    flag_old : Count of flagged features present in spectrum before refit.
    flag_new : Count of flagged features present in spectrum after refit.

    """
    flag_old, flag_new = 0, 0

    #  flag if old fitting results showed flagged feature
    if n_old > 0:
        flag_old = 1
    #  punish new fit if it contains more of the flagged features
    if n_new > n_old:
        flag_new = flag_old + 1
    #  same flags if the new and old fitting results show the same number of features
    elif n_new == n_old:
        flag_new = flag_old

    return flag_old, flag_new


def get_flags_rchi2(rchi2_old: float, rchi2_new: float, rchi2_limit: float) -> Tuple[int, int]:
    """Check how the reduced chi-square value of a spectrum changed after the refit.

    This check will only be performed if the 'self.flag_rchi2=True'.

    Parameters
    ----------
    fit_results : Dictionary containing the new best fit results after the refit attempt.
    index : Index ('index_fit' keyword) of the spectrum that gets/was refit.
    updated_fit_results : Only used in phase 2 of the spatially coherent refitting, in case the best fit solution was
        already updated in a previous iteration.

    Returns
    -------
    flag_old : Flag value before the refit.
    flag_new : Flag value after the refit.

    """
    flag_old, flag_new = 0, 0

    if rchi2_old > rchi2_limit:
        flag_old += 1
    if rchi2_new > rchi2_limit:
        flag_new += 1

    #  reward new fit if it is closer to rchi2 = 1 and thus likely less "overfit"
    if max(rchi2_old, rchi2_new) < rchi2_limit and abs(rchi2_new - 1) < abs(rchi2_old - 1):
        flag_old += 1

    return flag_old, flag_new


def get_flags_pvalue(pvalue_old: float, pvalue_new: float, min_pvalue: float) -> Tuple[int, int]:
    flag_old, flag_new = 0, 0

    if pvalue_old < min_pvalue:
        flag_old += 1
    if pvalue_new < min_pvalue:
        flag_new += 1

    #  punish fit if pvalue got worse
    if pvalue_new < pvalue_old:
        flag_new += 1

    return flag_old, flag_new


def get_flags_broad(
    fwhms_old: List,
    fwhms_new: List,
    contains_fwhm_flagged_as_broad: bool,
    fwhm_factor: float = 2.0,
    fwhm_separation: float = 4.0,
) -> Tuple[int, int]:
    """Check how the refit affected the number of components flagged as broad.

    This check will only be performed if the 'self.flag_broad=True'.

    Parameters
    ----------
    fit_results : Dictionary containing the new best fit results after the refit attempt.
    index : Index ('index_fit' keyword) of the spectrum that gets/was refit.
    updated_fit_results : Only used in phase 2 of the spatially coherent refitting, in case the best fit solution was
        already updated in a previous iteration.

    Returns
    -------
    flag_old : Flag value before the refit.
    flag_new : Flag value after the refit.

    """
    flag_old, flag_new = 0, 0

    if contains_fwhm_flagged_as_broad:
        flag_old = 1
        # No changes to the fit
        # TODO: Compare these with np.isclose instead
        if max(fwhms_old) == max(fwhms_new):
            flag_new = 1
        # Punish fit if component got even broader
        elif max(fwhms_new) > max(fwhms_old):
            flag_new = 2
    elif len(fwhms_new) > 1:
        #  punish fit if broad component was introduced
        fwhms = sorted(fwhms_new)
        if (fwhms[-1] > fwhm_factor * fwhms[-2]) and (fwhms[-1] - fwhms[-2]) > fwhm_separation:
            flag_new = 1

    return flag_old, flag_new


def get_flags_ncomps(
    ndiff_old: int,
    ndiff_new: int,
    njumps_old: int,
    njumps_new: int,
    max_diff_comps,
    n_max_jump_comps,
) -> Tuple[int, int]:
    """Check how the number of component jumps changed after the refit.

    Parameters
    ----------
    index : Index ('index_fit' keyword) of the spectrum that gets/was refit.
    updated_fit_results : Only used in phase 2 of the spatially coherent refitting, in case the best fit solution was
        already updated in a previous iteration.

    Returns
    -------
    flag_old : Flag value before the refit.
    flag_new : Flag value after the refit.

    """
    flag_old, flag_new = 0, 0

    if (njumps_old > n_max_jump_comps) or (ndiff_old > max_diff_comps):
        flag_old = 1
    if (njumps_new > n_max_jump_comps) or (ndiff_new > max_diff_comps):
        flag_new = 1
    if (njumps_new > njumps_old) or (ndiff_new > ndiff_old):
        flag_new += 1

    return flag_old, flag_new


def get_flags_centroids(
    means_old: List,
    means_new: List,
    interval: Optional[List] = None,
    n_centroids: Optional[int] = None,
) -> Tuple[int, int]:
    """Check how the presence of centroid positions changed after the refit.

    This check is only performed in phase 2 of the spatially coherent refitting.

    Parameters
    ----------
    fit_results : Dictionary containing the new best fit results after the refit attempt.
    index : Index ('index_fit' keyword) of the spectrum that gets/was refit.
    updated_fit_results : Only used in phase 2 of the spatially coherent refitting, in case the best fit solution was
        already updated in a previous iteration.
    interval : List specifying the interval of spectral channels where 'n_centroids' number of centroid positions
        are required.
    n_centroids : Number of centroid positions that should be present in interval.

    Returns
    -------
    flag_old : Flag value before the refit.
    flag_new : Flag value after the refit.

    """
    flag_old, flag_new = 2, 2

    n_centroids_old = sum(interval[0] < x < interval[1] for x in means_old)
    n_centroids_new = sum(interval[0] < x < interval[1] for x in means_new)

    #  reward new fit if it has the required number of centroid positions within 'interval'
    if n_centroids_new == n_centroids:
        flag_new = 0
    #  reward new fit if its number of centroid positions within 'interval' got closer to the required value
    elif abs(n_centroids_new - n_centroids) < abs(n_centroids_old - n_centroids):
        flag_new = 1
    #  punish new fit if its number of centroid positions within 'interval' compared to the required value got worse than in the old fit
    elif abs(n_centroids_new - n_centroids) > abs(n_centroids_old - n_centroids):
        flag_old = 1

    return flag_old, flag_new
