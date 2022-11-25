# Script to train parameters alpha1 and alpha2

import inspect
import multiprocessing
import numpy as np

from gausspyplus.utils.gaussian_functions import CONVERSION_STD_TO_FWHM
from gausspyplus.decomposition import agd_decomposer
import signal

from gausspyplus.utils.output import say


def init_worker():
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def count_ones_in_row(data):
    """Counts number of continuous trailing '1's Used in the convergence criteria"""
    output = np.zeros(len(data))
    for i in range(len(output)):
        if data[i] == 0:
            output[i] = 0
        else:
            total = 1
            counter = 1
            while data[i - counter] == 1:
                total += 1
                if i - counter < 0:
                    break
                counter += 1
            output[i] = total
    return output


def compare_parameters(guess_params, true_params):
    """Figure of merit for comparing guesses to true components.
    guess_params = list of 3xN parameters for the N guessed Gaussians
    = [amp1, amp2, amp3 ..., width1, width2, width3, ... offset1, offset2, offset3]
    true_params  = list of 3xN parameters for the N true Gaussians"""

    # Extract parameters
    n_true = len(true_params) // 3
    n_guess = len(guess_params) // 3
    guess_amps, guess_FWHMs, guess_offsets = np.split(np.array(guess_params), 3)
    true_amps, true_FWHMs, true_offsets = np.split(np.array(true_params), 3)

    truth_matrix = np.zeros([n_true, n_guess], dtype="int")
    # truth_matrix[i,j] = 1 if guess "j" is a correct match to true component "i"

    # Loop through answers and guesses
    for i in range(n_true):
        for j in range(n_guess):
            sigs_away = np.abs(
                (true_offsets[i] - guess_offsets[j])
                / (true_FWHMs[i] / CONVERSION_STD_TO_FWHM)
            )
            if (
                (sigs_away < 1.0)
                and (guess_FWHMs[j] > 0.3 * true_FWHMs[i])  # | Position match
                and (guess_FWHMs[j] < 2.5 * true_FWHMs[i])  # | Width match
                and (guess_amps[j] > 0.0)  # |
                and (guess_amps[j] < 10.0 * true_amps[i])  # | Amplitude match
            ):  # |

                # Check make sure this guess/answer pair in unique
                if not 1 in np.append(truth_matrix[i, :], truth_matrix[:, j]):
                    truth_matrix[i, j] = 1

    # Compute this training example's recall and precision
    n_correct = float(np.sum(np.sum(truth_matrix)))

    return n_correct, n_guess, n_true


def single_training_example(kwargs):
    j = kwargs["j"]
    true_params = np.append(
        kwargs["amps"][j], np.append(kwargs["FWHMs"][j], kwargs["means"][j])
    )

    # Produce initial guesses
    result = AGD_decomposer.AGD(
        vel=kwargs["vel"],
        data=kwargs["data"][j],
        errors=kwargs["errors"][j],
        alpha1=kwargs["alpha1"],
        alpha2=kwargs["alpha2"],
        plot=kwargs["plot"],
        verbose=kwargs["verbose"],
        SNR_thresh=kwargs["SNR_thresh"],
        perform_final_fit=False,  # TODO: Does it make sense to set perform_final_fit to False here?
        phase=kwargs["phase"],
        SNR2_thresh=kwargs["SNR2_thresh"],
    )

    guess_params = result["initial_parameters"]

    return compare_parameters(guess_params, true_params)


def objective_function(
    alpha1,
    alpha2,
    training_data,
    SNR_thresh=5.0,
    SNR2_thresh=0.0,
    phase=None,
    data=None,
    errors=None,
    means=None,
    vel=None,
    FWHMs=None,
    amps=None,
    verbose=False,
    plot=False,
):

    # Obtain dictionary of current-scope keywords/arguments
    frame = inspect.currentframe()
    args, _, _, values = inspect.getargvalues(frame)
    del values["frame"]  # This key not part of function arguments

    # Construct iterator of dictionaries of keywords for multi-processing
    mp_params = iter(
        [
            dict(list(values.items()) + list({"j": j}.items()))
            for j in range(len(training_data["data_list"]))
        ]
    )

    # Multiprocessing code
    ncpus = multiprocessing.cpu_count()
    p = multiprocessing.Pool(ncpus, init_worker)
    if verbose:
        print("N CPUs: ", ncpus)

    try:
        mp_results = p.map(single_training_example, mp_params)

    except KeyboardInterrupt:
        print("KeyboardInterrupt... quitting.")
        p.terminate()
        quit()

    p.close()
    del p

    Nc, Ng, Nt = np.array(mp_results).sum(0)
    accuracy = 2.0 * Nc / (Ng + Nt)  # Cumulative accuracy

    return -np.log(accuracy)


class GradientDescent(object):
    """Bookkeeping object."""

    def __init__(self, iterations):
        self.alpha1_trace = np.zeros(iterations + 1) * np.nan
        self.alpha2_trace = np.zeros(iterations + 1) * np.nan
        self.accuracy_trace = np.zeros(iterations) * np.nan
        self.D_alpha1_trace = np.zeros(iterations) * np.nan
        self.D_alpha2_trace = np.zeros(iterations) * np.nan
        self.alpha1means1 = np.zeros(iterations) * np.nan
        self.alpha1means2 = np.zeros(iterations) * np.nan
        self.alpha2means1 = np.zeros(iterations) * np.nan
        self.alpha2means2 = np.zeros(iterations) * np.nan
        self.fracdiff_alpha1 = np.zeros(iterations) * np.nan
        self.fracdiff_alpha2 = np.zeros(iterations) * np.nan
        self.iter_of_convergence = np.nan


def train(
    objective_function=objective_function,
    training_data=None,
    alpha1_initial=None,
    alpha2_initial=None,
    iterations=500,
    MAD=None,
    eps=None,
    learning_rate=None,
    momentum_value=None,
    window_size=10,
    iterations_for_convergence=10,
    plot=False,
    phase=None,
    SNR2_thresh=0.0,
    SNR_thresh=5.0,
    verbose=False,
    logger=False,
):
    """
    alpha1_initial =
    alpha2_initial =
    iterations =
    MAD = mean absolute difference
    eps = 'epsilson; finite offset for computing derivatives in gradient'
    learning_rate
    p = 'Momentum value'
    window_size = trailing window size to determine convergence,
    iterations_for_convergence = number of continuous iterations within threshold tolerence required to
                                 achieve convergence
    """

    # Default settings for hyperparameters
    if not learning_rate:
        learning_rate = 10.0
    if not eps:
        eps = 1.0
    if not MAD:
        MAD = 0.3
    if not momentum_value:
        momentum_value = 0.8

    thresh = MAD / np.sqrt(window_size)

    if phase == "one":
        momentum_value /= 3.0

    if alpha2_initial is None and phase == "two":
        say("alpha2_initial is required for two-phase decomposition.", logger=logger)
        return None

    if alpha2_initial is not None and phase == "one":
        say("alpha2_intial must be unset for one-phase decomposition.", logger=logger)
        return None

    # Unpack the training data
    data = training_data["data_list"]
    length = len(training_data["data_list"][0])
    errors = [np.ones(length) * error for error in training_data["error"]]
    means = training_data["means"]
    vel = training_data["x_values"]
    FWHMs = training_data["fwhms"]
    amps = training_data["amplitudes"]

    # Initialize book-keeping object
    gd = GradientDescent(iterations)
    gd.alpha1_trace[0] = alpha1_initial
    gd.alpha2_trace[0] = alpha2_initial

    for i in range(iterations):
        alpha1_r, alpha1_c, alpha1_l = (
            gd.alpha1_trace[i] + eps,
            gd.alpha1_trace[i],
            gd.alpha1_trace[i] - eps,
        )
        alpha2_r, alpha2_c, alpha2_l = (
            gd.alpha2_trace[i] + eps,
            gd.alpha2_trace[i],
            gd.alpha2_trace[i] - eps,
        )

        # Calls to objective function
        obj_1r = objective_function(
            alpha1_r,
            alpha2_c,
            training_data,
            phase=phase,
            data=data,
            errors=errors,
            means=means,
            vel=vel,
            FWHMs=FWHMs,
            amps=amps,
            verbose=verbose,
            plot=plot,
            SNR_thresh=SNR_thresh,
            SNR2_thresh=SNR2_thresh,
        )
        if eps == 0.0:
            print("Mean Accuracy: ", np.exp(-obj_1r))  # (Just sampling one position)
            quit()
        obj_1l = objective_function(
            alpha1_l,
            alpha2_c,
            training_data,
            phase=phase,
            data=data,
            errors=errors,
            means=means,
            vel=vel,
            FWHMs=FWHMs,
            amps=amps,
            verbose=verbose,
            plot=plot,
            SNR_thresh=SNR_thresh,
            SNR2_thresh=SNR2_thresh,
        )

        gd.D_alpha1_trace[i] = (obj_1r - obj_1l) / 2.0 / eps
        gd.accuracy_trace[i] = (obj_1r + obj_1l) / 2.0

        if phase == "two":
            # Calls to objective function
            obj_2r = objective_function(
                alpha1_c,
                alpha2_r,
                training_data,
                phase=phase,
                data=data,
                errors=errors,
                means=means,
                vel=vel,
                FWHMs=FWHMs,
                amps=amps,
                verbose=verbose,
                plot=plot,
                SNR_thresh=SNR_thresh,
                SNR2_thresh=SNR2_thresh,
            )
            obj_2l = objective_function(
                alpha1_c,
                alpha2_l,
                training_data,
                phase=phase,
                data=data,
                errors=errors,
                means=means,
                vel=vel,
                FWHMs=FWHMs,
                amps=amps,
                verbose=verbose,
                plot=plot,
                SNR_thresh=SNR_thresh,
                SNR2_thresh=SNR2_thresh,
            )
            gd.D_alpha2_trace[i] = (obj_2r - obj_2l) / 2.0 / eps
            gd.accuracy_trace[i] = (obj_1r + obj_1l + obj_2r + obj_2l) / 4.0

        if i == 0:
            momentum1, momentum2 = 0.0, 0.0
        else:
            momentum1 = momentum_value * (gd.alpha1_trace[i] - gd.alpha1_trace[i - 1])
            momentum2 = momentum_value * (gd.alpha2_trace[i] - gd.alpha2_trace[i - 1])

        gd.alpha1_trace[i + 1] = (
            gd.alpha1_trace[i] - learning_rate * gd.D_alpha1_trace[i] + momentum1
        )
        gd.alpha2_trace[i + 1] = (
            gd.alpha2_trace[i] - learning_rate * gd.D_alpha2_trace[i] + momentum2
        )

        # Sigma_alpha cannot be negative
        if gd.alpha1_trace[i + 1] < 0.0:
            gd.alpha1_trace[i + 1] = 0.0
        if gd.alpha2_trace[i + 1] < 0.0:
            gd.alpha2_trace[i + 1] = 0.0

        say("", logger=logger)
        say(
            f"{gd.alpha1_trace[i]}, {learning_rate}, {gd.D_alpha1_trace[i]}, {momentum1}",
            logger=logger,
        )
        say(
            "iter {0}: F1={1:4.1f}%, alpha=[{2}, {3}], momentum_value=[{4:4.2f}, {5:4.2f}]".format(
                i,
                100 * np.exp(-gd.accuracy_trace[i]),
                np.round(gd.alpha1_trace[i], 2),
                np.round(gd.alpha2_trace[i], 2),
                np.round(momentum1, 2),
                np.round(momentum2, 2),
            ),
            logger=logger,
            end=" ",
        )

        #    if False: (use this to avoid convergence testing)
        if i <= 2 * window_size:
            say(
                f" (Convergence testing begins in {int(2 * window_size - i)} iterations)",
                logger=logger,
            )
        else:
            gd.alpha1means1[i] = np.mean(gd.alpha1_trace[i - window_size : i])
            gd.alpha1means2[i] = np.mean(
                gd.alpha1_trace[i - 2 * window_size : i - window_size]
            )
            gd.alpha2means1[i] = np.mean(gd.alpha2_trace[i - window_size : i])
            gd.alpha2means2[i] = np.mean(
                gd.alpha2_trace[i - 2 * window_size : i - window_size]
            )

            gd.fracdiff_alpha1[i] = np.abs(gd.alpha1means1[i] - gd.alpha1means2[i])
            gd.fracdiff_alpha2[i] = np.abs(gd.alpha2means1[i] - gd.alpha2means2[i])

            if phase == "two":
                converge_logic = (gd.fracdiff_alpha1 < thresh) & (
                    gd.fracdiff_alpha2 < thresh
                )
            elif phase == "one":
                converge_logic = gd.fracdiff_alpha1 < thresh

            c = count_ones_in_row(converge_logic)
            say(
                f"  ({gd.fracdiff_alpha1[i]:4.2F},{gd.fracdiff_alpha2[i]:4.2F} < {thresh:4.2F} for {int(c[i])} iters [{iterations_for_convergence} required])",
                logger=logger,
            )

            if np.any(c > iterations_for_convergence):
                i_converge = np.min(np.argwhere(c > iterations_for_convergence))
                gd.iter_of_convergence = i_converge
                say(
                    f"Stable convergence achieved at iteration: {i_converge}",
                    logger=logger,
                )
                break

    # Return best-fit alphas, and bookkeeping object
    return gd.alpha1means1[i], gd.alpha2means1[i], gd
