# codeing=utf-8
"""Contains code for parameter-free locally accelerated conditional gradient."""

from copy import deepcopy
import logging
import time
import numpy as np

from pflacg.algorithms._abstract_algorithm import _AbstractAlgorithm


logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s :: %(asctime)s :: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOGGER = logging.getLogger()


class ParameterFreeLaCG(_AbstractAlgorithm):
    def __init__(self, **kwargs):
        pass

    def run(
        self,
        objective_function,
        feasible_region,
        exit_criterion,
        initial_point=None,
        initial_active_set=None
        async=False,
    ):
        pass


class FractionalAwayStepFW(_AbstractAlgorithm):
    def __init__(self, ratio=0.5, **kwargs):
        pass

    def run(self, **kwargs):
        pass


class wACC(_AbstractAlgorithm):
    def __init__(self, **kwargs):
        pass

    def run(self, **kwargs):
        pass


class AdaptiveFW(_AbstractAlgorithm):
    """
    Implementation of Adaptive Frank-Wolfe/Conditional Gradients algorithms
    AdaAFW, AdaPFW and AdaFW from Pedregosa et al.
    
    Pedregosa, Fabian, et al. "Linearly convergent Frank-Wolfe with backtracking line-search." 
    International Conference on Artificial Intelligence and Statistics. PMLR, 2020.
    
    The tau parameter controils the rate at which we increase the smoothness estimate, when the smoothness estimate is found to be too small.
    The eta parameter controls the (potential) shrinking of the estimate that happens when we start an iteration (in the hopes of locally adapting to the local smoothness.)
    
    """


    def __init__(self, fw_variant, L_estimate, tau, eta):
        assert tau > 1.0 and eta <= 1.0, "Input parameters for the adaptive algorithms are incorrect."
        assert fw_variant == "AdaAFW" or fw_variant == "AdaPFW" or fw_variant == "AdaFW", "Wrong variant supplied to the adaptive algorithm"
        self.fw_variant = fw_variant
        self.step_size_parameters = {"L_estimate": L_estimate, "tau": tau, "eta": eta}
        return

    def backtracking_step_size(function, d, x, grad, alpha_max):
        M = self.step_size_parameters["L_estimate"] * self.step_size_parameters["eta"]
        d_norm_squared = np.dot(d, d)
        g_t = np.dot(-grad, d)
        alpha = min(g_t / (M * d_norm_squared), alpha_max)
        while (
            function.f(x + alpha * d)
            > function.f(x) - alpha * g_t + 0.5 * M * d_norm_squared * alpha * alpha
        ):
            M *= self.step_size_parameters["tau"]
            alpha = min(g_t / (M * d_norm_squared), alpha_max)
        self.step_size_parameters["L_estimate"] = M
        return alpha

    @staticmethod
    def step_fw(objective_function, feasible_region, x):
        grad = objective_function.evaluate_grad(x)
        v = feasible_region.lp_oracle(grad)
        # Choose FW direction, can overwrite index.
        d = v - x
        alphaMax = 1.0
        optStep = backtracking_step_size(objective_function, d, x, grad, alphaMax)
        return x + alpha * d, np.dot(grad, x - v)

    @staticmethod
    def away_step_fw(
        objective_function, feasible_region, x, active_set, lambdas
    ):
        assert np.all(np.asarray(lambdas) > 0.0), "Invalid lambda values in AFW."
        grad = objective_function.evaluate_grad(x)
        v = feasible_region.lp_oracle(grad)
        a, indexMax = feasible_region.away_oracle(grad, active_set)
        # Choose FW direction, can overwrite index.
        FWGap = np.dot(grad, x - v)
        if FWGap == 0.0:
            return x, FWGap
        if FWGap > np.dot(grad, a - x):
            d = v - x
            alphaMax = 1.0
            optStep = backtracking_step_size(objective_function, d, x, grad, alphaMax)
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
            optStep = backtracking_step_size(objective_function, d, x, grad, alphaMax)
            lambdas[:] = [i * (1 + alpha) for i in lambdas]
            # Max step, need to delete a vertex.
            if alpha != alphaMax:
                lambdas[indexMax] -= alpha
            else:
                delete_vertex_index(indexMax, active_set, lambdas)
        return x + alpha * d, FWGap

    @staticmethod
    def pairwise_step_fw(
        objective_function, feasible_region, x, active_set, lambdas
    ):
        grad = objective_function.evaluate_grad(x)
        v = feasible_region.lp_oracle(grad)
        a, index = feasible_region.away_oracle(grad, active_set)
        # Find the weight of the extreme point a in the decomposition.
        alphaMax = lambdas[index]
        # Update weight of away vertex.
        d = v - a
        optStep = backtracking_step_size(objective_function, d, x, grad, alphaMax)
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
        return x + alpha * d, np.dot(grad, x - v)

    def run(
        self,
        objective_function,
        feasible_region,
        exit_criterion,
        initial_point=None,
        active_set=None,
        initial_barycentric_coordinates=None,
    ):
        # Setting initial point
        if initial_point is None or active_set is None or initial_barycentric_coordinates is None:
            x = feasible_region.initial_point.copy()
            active_set = [x]
            lambdas = [1.0]
        else:
            x = initial_point.copy()
            active_set = deepcopy(active_set)
            lambdas = deepcopy(initial_barycentric_coordinates)

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
            if self.fw_variant == "AdaAFW":
                x, dual_gap = self.away_step_fw(
                    objective_function,
                    feasible_region,
                    x,
                    active_set,
                    lambdas,
                )
            if self.fw_variant == "AdaPFW":
                x, dual_gap = self.pairwise_step_fw(
                    objective_function,
                    feasible_region,
                    x,
                    active_set,
                    lambdas,
                )
            if self.fw_variant == "AdaFW":
                x, dual_gap = self.step_fw(
                    objective_function,
                    feasible_region,
                    x,
                )

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