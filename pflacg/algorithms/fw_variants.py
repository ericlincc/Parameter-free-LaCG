# codeing=utf-8
"""Contains code for parameter-free locally accelerated conditional gradient."""

from copy import deepcopy
import logging
import time
import numpy as np

from pflacg.algorithms._abstract_algorithm import _AbstractAlgorithm

from pflacg.algorithms._algorithms_utils import step_size, DISPLAY_DECIMALS, new_vertex_fail_fast, delete_vertex_index, calculate_stepsize


logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s :: %(asctime)s :: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOGGER = logging.getLogger()


class FrankWolfe(_AbstractAlgorithm):
    """
    Implementation of Frank-Wolfe/Conditional Gradients algorithms with line search

    Implementation of Adaptive Frank-Wolfe/Conditional Gradients algorithms
    AdaAFW, AdaPFW and AdaFW from Pedregosa et al.

    Pedregosa, Fabian, et al. "Linearly convergent Frank-Wolfe with backtracking line-search."
    International Conference on Artificial Intelligence and Statistics. PMLR, 2020.

    The tau parameter controls the rate at which we increase the smoothness estimate,
    when the smoothness estimate is found to be too small.
    The eta parameter controls the (potential) shrinking of the estimate that happens
    when we start an iteration (in the hopes of locally adapting to the local smoothness.)


    """

    # TODO: Add comment above referencing the papers for ALL algorithms.

    def __init__(
        self, fw_variant, step_type, tau=2.0, eta=0.9, smoothness_estimate=1.0e-4
    ):
        """
        Parameters
        ----------
        fw_variant: str
            Variant for the algorithm to run:
                -"FW": Vanilla Frank-Wolfe
                -"AFW": Away-step Frank-Wolfe
                -"PFW": Pairwise-step Frank-Wolfe
                -"DIPFW": Decomposition-invariant FW
        step_type: str
            Variant for the algorithm to run:
                -"line_search": Exact linesearch.
                -"adaptive_short_step": Short step that minimizes smoothness.
        """
        self.fw_variant = fw_variant
        assert (
            fw_variant == "AFW"
            or fw_variant == "PFW"
            or fw_variant == "FW"
            or fw_variant == "DIPFW"
        ), "Wrong variant supplied to the adaptive algorithm"
        assert (
            step_type == "line_search" or step_type == "adaptive_short_step"
        ), "Wrong step size strategy supplied to the algorithm"
        if fw_variant == "DIPFW":
            assert (
                step_type == "line_search"
            ), "DIPFW is only parameter-free with line search"
        self.step_size_param = {
            "type_step": step_type,
            "L_estimate": smoothness_estimate,
            "tau": tau,
            "eta": eta,
        }
        
        return

    def run(
        self,
        objective_function,
        feasible_region,
        exit_criterion,
        initial_point=None,
        active_set=None,
        lambdas=None,
    ):

        if (
            initial_point is None
            or active_set is None
            or lambdas is None
        ):
            x = feasible_region.initial_point.copy()
            if self.fw_variant != "FW":
                active_set = [x]
                lambdas = [1.0]
        else:
            x = initial_point.copy()
            if self.fw_variant != "FW":
                active_set = deepcopy(active_set)
                lambdas = deepcopy(lambdas)

        start_time = time.time()
        grad = objective_function.evaluate_grad(x)

        iteration = 0
        duration = 0.0
        f_val = objective_function.evaluate(x)
        dual_gap = np.dot(grad, x - feasible_region.lp_oracle(grad))
        run_status = (iteration, duration, f_val, dual_gap)
        LOGGER.info(
            "Running " + str(self.fw_variant) + "({5}): "
            "iteration = {1:.{0}f}, duration = {2:.{0}f}, f_val = {3:.{0}f}, dual_gap = {4:.{0}f}".format(
                DISPLAY_DECIMALS, *run_status, self.fw_variant
            )
        )

        run_history = [run_status]

        while not exit_criterion.has_met_exit_criterion(run_status):
            if self.fw_variant == "AFW":
                x, dual_gap = away_step_fw(
                    objective_function,
                    feasible_region,
                    x,
                    active_set,
                    lambdas,
                    self.step_size_param,
                )
            if self.fw_variant == "PFW":
                x, dual_gap = pairwise_step_fw(
                    objective_function,
                    feasible_region,
                    x,
                    active_set,
                    lambdas,
                    self.step_size_param,
                )
            if self.fw_variant == "FW":
                x, dual_gap = step_fw(
                    objective_function,
                    feasible_region,
                    x,
                    self.step_size_param,
                )
            if self.fw_variant == "DIPFW":
                x, dual_gap = DIPFW(objective_function, feasible_region, x, self.step_size_param)

            iteration += 1
            duration = time.time() - start_time
            f_val = objective_function.evaluate(x)
            run_status = (
                iteration,
                duration,
                f_val,
                dual_gap,
            )
            LOGGER.info(
                "Running " + str(self.fw_variant) + ": "
                "iteration = {1}, duration = {2:.{0}f}, f_val = {3:.{0}f}, dual_gap = {4:.{0}f}".format(
                    DISPLAY_DECIMALS, *run_status
                )
            )
            run_history.append(run_status)
        return run_history

def step_fw(objective_function, feasible_region, x, step_size_param):
    grad = objective_function.evaluate_grad(x)
    v = feasible_region.lp_oracle(grad)
    d = v - x
    alphaMax = 1.0
    alpha = step_size(
        objective_function, x, d, grad, alphaMax, step_size_param
    )
    return x + alpha * d, grad.dot(x - v)

def away_step_fw(objective_function, feasible_region, x, active_set, lambdas, step_size_param):
    assert np.all(np.asarray(lambdas) > 0.0), "Invalid lambda values in AFW."
    grad = objective_function.evaluate_grad(x)
    v = feasible_region.lp_oracle(grad)
    a, indexMax = feasible_region.away_oracle(grad, active_set)
    # Choose FW direction, can overwrite index.
    FWGap = grad.dot(x - v)
    if FWGap == 0.0:
        return x, FWGap
    if FWGap > np.dot(grad, a - x):
        d = v - x
        alphaMax = 1.0
        alpha = step_size(
            objective_function, x, d, grad, alphaMax, step_size_param
        )
        if alpha != alphaMax:
            flag, index = new_vertex_fail_fast(v, active_set)
            lambdas[:] = [i * (1 - alpha) for i in lambdas]
            if flag:
                active_set.append(v)
                lambdas.append(alpha)
            else:
                # Update existing weights
                lambdas[index] += alpha
        # Max step length away step, only one vertex now.
        else:
            active_set[:] = [v]
            lambdas[:] = [alphaMax]
    else:
        d = x - a
        alphaMax = lambdas[indexMax] / (1.0 - lambdas[indexMax])
        alpha = step_size(
            objective_function, x, d, grad, alphaMax, step_size_param
        )
        lambdas[:] = [i * (1 + alpha) for i in lambdas]
        # Max step, need to delete a vertex.
        if alpha != alphaMax:
            lambdas[indexMax] -= alpha
        else:
            delete_vertex_index(indexMax, active_set, lambdas)
    return x + alpha * d, FWGap

def pairwise_step_fw( objective_function, feasible_region, x, active_set, lambdas, step_size_param):
    grad = objective_function.evaluate_grad(x)
    v = feasible_region.lp_oracle(grad)
    a, index = feasible_region.away_oracle(grad, active_set)
    # Find the weight of the extreme point a in the decomposition.
    alphaMax = lambdas[index]
    # Update weight of away vertex.
    d = v - a
    alpha = step_size(
        objective_function, x, d, grad, alphaMax, step_size_param
    )
    lambdas[index] -= alpha
    if alpha == alphaMax:
        delete_vertex_index(index, active_set, lambdas)
    # Update the FW vertex
    flag, index = new_vertex_fail_fast(v, active_set)
    if flag:
        active_set.append(v)
        lambdas.append(alpha)
    else:
        lambdas[index] += alpha
    return x + alpha * d, grad.dot(x - v)

def DIPFW(objective_function, feasible_region, x, step_size_param):
    grad = objective_function.evaluate_grad(x)
    v = feasible_region.lp_oracle(grad)
    grad_aux = grad.copy()
    for i in range(len(grad_aux)):
        if x[i] == 0.0:
            grad_aux[i] = -1.0e15
    a = feasible_region.lp_oracle(-grad_aux)
    d = v - a
    alphaMax = calculate_stepsize(x, d)
    assert step_size_param["type_step"] == "line_search", "DIPFW only accepts exact linesearch."
    alpha = step_size(
        objective_function, x, d, grad, alphaMax, step_size_param
    )
    return x + alpha * d, grad.dot(x - v)