# codeing=utf-8
"""Contains code for parameter-free locally accelerated conditional gradient."""

from copy import deepcopy
import logging
import time

import numpy as np

from pflacg.algorithms._abstract_algorithm import _AbstractAlgorithm
from pflacg.algorithms._algorithms_utils import (
    Point,
    step_size,
    DISPLAY_DECIMALS,
    calculate_stepsize,
)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s :: %(asctime)s :: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
LOGGER = logging.getLogger()


class FrankWolfeSimplex(_AbstractAlgorithm):
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
        self,
        fw_variant,
        step_type,
        tau=2.0,
        eta=0.9,
        smoothness_estimate=1.0e-4,
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
            fw_variant == "AFW" or fw_variant == "PFW" or fw_variant == "FW" or fw_variant == "DIPFW"
        ), "Wrong variant supplied to the adaptive algorithm"
        assert (
            step_type == "line_search" or step_type == "adaptive_short_step"
        ), "Wrong step size strategy supplied to the algorithm"
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
        x_initial=None,
        save_and_output_results=True,
        global_iter=None,
    ):

        if x_initial is None:
            x = feasible_region.initial_point.copy()
        else:
            x = x_initial

        start_time = time.time()
        grad = objective_function.evaluate_grad(x)

        iteration = 0
        duration = 0.0
        f_val = objective_function.evaluate(x)
        v = feasible_region.lp_oracle(grad)
        if self.fw_variant == "FW":
            strong_wolfe_gap = 0.0
        else:
            a, index_max = feasible_region.away_oracle_fast(grad, x)
            strong_wolfe_gap = grad.dot(a - v)

        dual_gap = grad.dot(x - v)

        run_status = (iteration, duration, f_val, dual_gap, strong_wolfe_gap)
        if save_and_output_results:
            LOGGER.info(
                "Running " + str(self.fw_variant) + "({5}): "
                "iteration = {1:.{0}f}, duration = {2:.{0}f}, f_val = {3:.{0}f}, dual_gap = {4:.{0}f}, strong_wolfe_gap = {5:.{0}f}".format(
                    DISPLAY_DECIMALS, *run_status, self.fw_variant
                )
            )
            run_history = [run_status]

        while True:
            x_prev = x
            if self.fw_variant == "AFW":
                x, dual_gap_prev, strong_wolfe_gap_prev = away_step_fw_simplex(
                    objective_function,
                    feasible_region,
                    x,
                    self.step_size_param,
                )
            if self.fw_variant == "PFW":
                x, dual_gap_prev, strong_wolfe_gap_prev = pairwise_step_fw_simplex(
                    objective_function,
                    feasible_region,
                    x,
                    self.step_size_param,
                )
            if self.fw_variant == "FW":
                x, dual_gap_prev, strong_wolfe_gap_prev = step_fw_simplex(
                    objective_function,
                    feasible_region,
                    x,
                    self.step_size_param,
                )
            if self.fw_variant == "DIPFW":
                x, dual_gap_prev, strong_wolfe_gap_prev = dipfw_simplex(
                    objective_function,
                    feasible_region,
                    x,
                    self.step_size_param,
                )
            iteration += 1
            duration = time.time() - start_time
            f_val = objective_function.evaluate(x)
            run_status = (
                iteration,
                duration,
                f_val,
                dual_gap_prev,
                strong_wolfe_gap_prev,
            )
            if exit_criterion.has_met_exit_criterion(run_status):
                break
            if global_iter:
                # Increment global iteration count
                with global_iter.get_lock():
                    global_iter.value += 1
            if save_and_output_results:
                if dual_gap_prev is None or strong_wolfe_gap_prev is None:
                    LOGGER.info(
                        "Running " + str(self.fw_variant) + ": "
                        "iteration = {1}, duration = {2:.{0}f}, "
                        "f_val = {3:.{0}f}, dual_gap = None, strong_wolfe_gap = None".format(
                            DISPLAY_DECIMALS, *run_status
                        )
                    )
                else:
                    LOGGER.info(
                        "Running " + str(self.fw_variant) + ": "
                        "iteration = {1}, duration = {2:.{0}f}, "
                        "f_val = {3:.{0}f}, dual_gap = {4:.{0}f}, strong_wolfe_gap = {5:.{0}f}".format(
                            DISPLAY_DECIMALS, *run_status
                        )
                    )
                run_history.append(run_status)
        if save_and_output_results:
            return run_history
        else:
            return x_prev, dual_gap_prev, strong_wolfe_gap_prev

def step_fw_simplex(objective_function, feasible_region, x, step_size_param):
    grad = objective_function.evaluate_grad(x)
    v = feasible_region.lp_oracle(grad)
    a, index_max = feasible_region.away_oracle_fast(grad, x)
    wolfe_gap = grad.dot(x - v)
    strong_wolfe_gap = grad.dot(a - v)
    d = v - x
    alpha_max = 1.0
    alpha = step_size(
        objective_function,
        x,
        d,
        grad,
        alpha_max,
        step_size_param,
    )
    return x + alpha * d, wolfe_gap, strong_wolfe_gap

def away_step_fw_simplex(objective_function, feasible_region, x, step_size_param):
    grad = objective_function.evaluate_grad(x)
    v = feasible_region.lp_oracle(grad)
    a, index_max = feasible_region.away_oracle_fast(grad, x)
    wolfe_gap = grad.dot(x - v)
    strong_wolfe_gap = grad.dot(a - v)
    if 2.0 * wolfe_gap > strong_wolfe_gap:
        d = v - x
        alpha_max = 1.0
        alpha = step_size(
            objective_function,
            x,
            d,
            grad,
            alpha_max,
            step_size_param,
        )
    else:
        d = x - a
        alpha_max = x[index_max] / (1.0 - x[index_max])
        alpha = step_size(
            objective_function,
            x,
            d,
            grad,
            alpha_max,
            step_size_param,
        )
    return x + alpha * d, wolfe_gap, strong_wolfe_gap


def pairwise_step_fw_simplex(objective_function, feasible_region, x, step_size_param):
    grad = objective_function.evaluate_grad(x)
    v = feasible_region.lp_oracle(grad)
    wolfe_gap = grad.dot(x - v)
    a, index_max = feasible_region.away_oracle_fast(grad, x)
    strong_wolfe_gap = grad.dot(a - v)
    # Find the weight of the extreme point a in the decomposition.
    alpha_max = x[index_max]
    alpha = step_size(
        objective_function,
        x,
        v - a,
        grad,
        alpha_max,
        step_size_param,
    )
    return x + alpha * (v - a), wolfe_gap, strong_wolfe_gap


def dipfw_simplex(objective_function, feasible_region, x, step_size_param):
    grad = objective_function.evaluate_grad(x)
    v = feasible_region.lp_oracle(grad)
    a, index_max = feasible_region.away_oracle_fast(grad, x)
    grad_aux = grad.copy()
    wolfe_gap = grad.dot(x - v)
    strong_wolfe_gap = grad.dot(a - v)
    for i in range(len(grad_aux)):
        if x[i] == 0.0:
            grad_aux[i] = -1.0e15
    a = feasible_region.lp_oracle(-grad_aux)
    d = v - a
    alpha_max = calculate_stepsize(x, d)
    assert (
        step_size_param["type_step"] == "line_search"
    ), "DIPFW only accepts exact linesearch."
    alpha = step_size(
        objective_function,
        x,
        d,
        grad,
        alpha_max,
        step_size_param,
    )
    return x + alpha * d, wolfe_gap, strong_wolfe_gap


class ConditionalGradientSliding:
    def __init__(self):
        self.iteration = 0
    def run(self, objective_function, feasible_region, exit_criterion, x_initial=None, save_and_output_results=True, global_iter=None):
        
        if x_initial is None:
            x = feasible_region.initial_point.copy()
        else:
            x = x_initial
        start_time = time.time()
        grad = objective_function.evaluate_grad(x)
        duration = 0.0
        f_val = objective_function.evaluate(x)
        v = feasible_region.lp_oracle(grad)
        dual_gap = grad.dot(x - v)
        a, index_max = feasible_region.away_oracle_fast(grad, x)
        strong_wolfe_gap = grad.dot(a - v)
        run_status = (self.iteration, duration, f_val, dual_gap, strong_wolfe_gap)
        if save_and_output_results:
            LOGGER.info(
                "Running CGS "
                "iteration = {1:.{0}f}, duration = {2:.{0}f}, f_val = {3:.{0}f}, dual_gap = {4:.{0}f}, strong_wolfe_gap = {5:.{0}f}".format(
                    DISPLAY_DECIMALS, *run_status,
                )
            )
            run_history = [run_status]
        
        
        N = int(np.ceil(2*np.sqrt(6.0*objective_function.largest_eigenvalue/objective_function.smallest_eigenvalue)))
        s = 1.0
        while(True): 
            x = self.CGSubroutine(objective_function, feasible_region, x, dual_gap, N, s)
            s += 1.0
            grad = objective_function.evaluate_grad(x)
            v = feasible_region.lp_oracle(grad)
            dual_gap_prev = grad.dot(x - v)
            a, index_max = feasible_region.away_oracle_fast(grad, x)
            strong_wolfe_gap_prev = grad.dot(a - v)
            duration = time.time() - start_time
            f_val = objective_function.evaluate(x)
            run_status = (
                self.iteration,
                duration,
                f_val,
                dual_gap_prev,
                strong_wolfe_gap_prev,
            )
            if exit_criterion.has_met_exit_criterion(run_status):
                break
            
            if global_iter:
                # Increment global iteration count
                with global_iter.get_lock():
                    global_iter.value += 1
            LOGGER.info(
                "Running CGS "
                "iteration = {1}, duration = {2:.{0}f}, "
                "f_val = {3:.{0}f}, dual_gap =  {4:.{0}f}, strong_wolfe_gap =  {5:.{0}f}".format(
                    DISPLAY_DECIMALS, *run_status
                )
            )
            run_history.append(run_status)
        if save_and_output_results:
            return run_history
        else:
            return x, dual_gap_prev, strong_wolfe_gap_prev
            
    def CGSubroutine(self, function, feasible_region, x0, delta0, N, s):
        L = function.largest_eigenvalue
        Mu = function.smallest_eigenvalue
        y = x0.copy()
        x = x0.copy()
        for k in range(1, N + 1):
            gamma = 2.0/(k + 1.0)
            nu = 8.0*L*delta0*np.power(2, -s)/(Mu*N*k)
            beta = 2.0*L/k
            z = (1 - gamma)*y + gamma*x
            x = self.CGSuProjection(function.evaluate_grad(z), x, beta, nu, feasible_region)
            y = (1 - gamma)*y + gamma*x
        return y
    
    def CGSuProjection(self, g, u, beta, nu, feasible_region):
        t = 1
        u_t = u
        while(True):
            grad = g + beta*(u_t - u)
            v = feasible_region.lp_oracle(grad)      
            self.iteration += 1
            V = np.dot(g + beta*(u_t - u), u_t - v)
            if(V <= nu):
                return u_t 
            else:
                d = v - u_t
                alphaOpt = -np.dot(grad, d)/(beta*np.dot(d,d))
                alpha = min(1, alphaOpt)
                u_t = (1 - alpha)*u_t + alpha*v
                t += 1