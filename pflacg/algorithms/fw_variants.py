# codeing=utf-8
"""Contains code for parameter-free locally accelerated conditional gradient."""

from copy import deepcopy
import logging
import time
import numpy as np

from pflacg.algorithms._abstract_algorithm import _AbstractAlgorithm

from pflacg.algorithms._algorithms_utils import step_size, DISPLAY_DECIMALS, new_vertex_fail_fast, delete_vertex_index, calculate_stepsize

from pflacg.algorithms._algorithms_utils import Point, max_min_vertex_backup, max_min_vertex_quick_exit_backup


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
            or fw_variant == "lazy"
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
        point_initial=None,
        save_and_output_results=True,
    ):

        if (
            point_initial is None
        ):
            vertex = feasible_region.initial_point.copy()
            x = Point(vertex, (1.0,), (vertex,))
        else:
            x = point_initial

        start_time = time.time()
        grad = objective_function.evaluate_grad(x.cartesian_coordinates)

        iteration = 0
        duration = 0.0
        f_val = objective_function.evaluate(x.cartesian_coordinates)
        v = feasible_region.lp_oracle(grad)
        if(self.fw_variant == "FW" or self.fw_variant == "DIPFW"):
            strong_wolfe_gap = 0.0
        else:
            a, indexMax = feasible_region.away_oracle(grad, x)
            strong_wolfe_gap = grad.dot(a.cartesian_coordinates - v)
            

            
        dual_gap = grad.dot(x.cartesian_coordinates - v)
        
        if self.fw_variant == "lazy" or self.fw_variant == "lazy quick exit":
            phiVal = [dual_gap]
        
        run_status = (iteration, duration, f_val, dual_gap, strong_wolfe_gap)
        if(save_and_output_results):
            LOGGER.info(
                "Running " + str(self.fw_variant) + "({5}): "
                "iteration = {1:.{0}f}, duration = {2:.{0}f}, f_val = {3:.{0}f}, dual_gap = {4:.{0}f}, strong_wolfe_gap = {5:.{0}f}".format(
                    DISPLAY_DECIMALS, *run_status, self.fw_variant
                )
            )
            run_history = [run_status]


            
        while not exit_criterion.has_met_exit_criterion(run_status):
            if self.fw_variant == "AFW":
                x, dual_gap, strong_wolfe_gap = away_step_fw(
                    objective_function,
                    feasible_region,
                    x,
                    self.step_size_param,
                )
            if self.fw_variant == "PFW":
                x, dual_gap, strong_wolfe_gap = pairwise_step_fw(
                    objective_function,
                    feasible_region,
                    x,
                    self.step_size_param,
                )
            if self.fw_variant == "FW":
                x, dual_gap, strong_wolfe_gap = step_fw(
                    objective_function,
                    feasible_region,
                    x,
                    self.step_size_param,
                )
            if self.fw_variant == "lazy":
                x, dual_gap, strong_wolfe_gap = FW_away_lazy(
                    objective_function,
                    feasible_region,
                    x,
                    self.step_size_param,
                    phiVal,
                )
            # if self.fw_variant == "lazy quick exit":
            #     x, dual_gap, strong_wolfe_gap = FW_away_lazy_quick_exit(
            #         objective_function,
            #         feasible_region,
            #         x,
            #         self.step_size_param,
            #         phiVal,
            #     )
            if self.fw_variant == "DIPFW":
                x, dual_gap, strong_wolfe_gap = DIPFW(objective_function, feasible_region, x, self.step_size_param)
            iteration += 1
            duration = time.time() - start_time
            f_val = objective_function.evaluate(x.cartesian_coordinates)
            run_status = (
                iteration,
                duration,
                f_val,
                dual_gap,
                strong_wolfe_gap,
            )
            if(save_and_output_results):
                LOGGER.info(
                    "Running " + str(self.fw_variant) + ": "
                    "iteration = {1}, duration = {2:.{0}f}, f_val = {3:.{0}f}, dual_gap = {4:.{0}f}, strong_wolfe_gap = {5:.{0}f}".format(
                        DISPLAY_DECIMALS, *run_status
                    )
                )
                run_history.append(run_status)
        if(save_and_output_results):
            return run_history
        else:
            return x

#Note that the VANILLA FW algorithm only uses the cartesian coordinates
#and does not use the active set or the barycentric coordinates for anything.
def step_fw(objective_function, feasible_region, point_x, step_size_param):
    grad = objective_function.evaluate_grad(point_x.cartesian_coordinates)
    v = feasible_region.lp_oracle(grad)
    FWGap = grad.dot(point_x.cartesian_coordinates - v)
    d = v - point_x.cartesian_coordinates
    alphaMax = 1.0
    alpha = step_size(
        objective_function, point_x.cartesian_coordinates, d, grad, alphaMax, step_size_param
    )
    if alpha != alphaMax:
        flag, point_v = point_x.is_vertex_in_support(v)
        if flag == False:
            new_barycentric_coordinates = list(point_x.barycentric_coordinates)
            new_barycentric_coordinates.append(0.0)
            point_x = Point(point_x.cartesian_coordinates, tuple(new_barycentric_coordinates), point_v.support)
        return point_x + alpha*(point_v - point_x), FWGap, 0.0
    else:
        return Point(v, (1.0,), (v,)), FWGap,  0.0

def away_step_fw(objective_function, feasible_region, point_x, step_size_param):
    grad = objective_function.evaluate_grad(point_x.cartesian_coordinates)
    v = feasible_region.lp_oracle(grad)
    point_a, indexMax = feasible_region.away_oracle(grad, point_x)
    FWGap = grad.dot(point_x.cartesian_coordinates - v)
    StrongFWGap = grad.dot(point_a.cartesian_coordinates - v)
    if FWGap > grad.dot(point_a.cartesian_coordinates - point_x.cartesian_coordinates):
        alphaMax = 1.0
        alpha = step_size(
            objective_function, point_x.cartesian_coordinates, v - point_x.cartesian_coordinates, grad, alphaMax, step_size_param
        )
        if alpha != alphaMax:
            flag, point_v = point_x.is_vertex_in_support(v)
            if flag == False:
                new_barycentric_coordinates = list(point_x.barycentric_coordinates)
                new_barycentric_coordinates.append(0.0)
                point_x = Point(point_x.cartesian_coordinates, tuple(new_barycentric_coordinates), point_v.support)
            return point_x + alpha*(point_v - point_x), FWGap, StrongFWGap
        else:
            return Point(v, (1.0,), (v,)), FWGap, StrongFWGap
    else:
        alphaMax = point_x.barycentric_coordinates[indexMax] / (1.0 - point_x.barycentric_coordinates[indexMax])
        alpha = step_size(
            objective_function, point_x.cartesian_coordinates, point_x.cartesian_coordinates - point_a.cartesian_coordinates, grad, alphaMax, step_size_param
        )
        point_x = point_x + alpha * (point_x - point_a)
        if alpha == alphaMax:
            point_x = point_x.delete_vertex_in_support(indexMax)
        return point_x, FWGap, StrongFWGap

def pairwise_step_fw( objective_function, feasible_region, point_x, step_size_param):
    grad = objective_function.evaluate_grad(point_x.cartesian_coordinates)
    v = feasible_region.lp_oracle(grad)
    FW_gap = grad.dot(point_x.cartesian_coordinates - v)
    point_a, indexMax = feasible_region.away_oracle(grad, point_x)
    StrongWolfe_gap = grad.dot(point_a.cartesian_coordinates - v)
    # Find the weight of the extreme point a in the decomposition.
    alphaMax = point_x.barycentric_coordinates[indexMax]
    alpha = step_size(
        objective_function, point_x.cartesian_coordinates, v - point_a.cartesian_coordinates, grad, alphaMax, step_size_param
    )
    flag, point_v = point_x.is_vertex_in_support(v)
    if flag == False:
        new_barycentric_coordinates = list(point_x.barycentric_coordinates)
        new_barycentric_coordinates.append(0.0)
        point_x = Point(point_x.cartesian_coordinates, tuple(new_barycentric_coordinates), point_v.support)
        new_barycentric_coordinates = list(point_a.barycentric_coordinates)
        new_barycentric_coordinates.append(0.0)
        point_a = Point(point_a.cartesian_coordinates, tuple(new_barycentric_coordinates), point_v.support)
    point_x = point_x + alpha * (point_v - point_a)
    if alpha == alphaMax:
        point_x = point_x.delete_vertex_in_support(indexMax)
    return point_x, FW_gap, StrongWolfe_gap

def DIPFW(objective_function, feasible_region, point_x, step_size_param):
    grad = objective_function.evaluate_grad(point_x.cartesian_coordinates)
    v = feasible_region.lp_oracle(grad)
    grad_aux = grad.copy()
    for i in range(len(grad_aux)):
        if point_x.cartesian_coordinates[i] == 0.0:
            grad_aux[i] = -1.0e15
    a = feasible_region.lp_oracle(-grad_aux)
    d = v - a
    alphaMax = calculate_stepsize(point_x.cartesian_coordinates, d)
    assert step_size_param["type_step"] == "line_search", "DIPFW only accepts exact linesearch."
    alpha = step_size(
        objective_function, point_x.cartesian_coordinates, d, grad, alphaMax, step_size_param
    )
    new_cartesian = point_x.cartesian_coordinates + alpha * d
    return Point(new_cartesian, (1.0,), (new_cartesian,)), grad.dot(point_x.cartesian_coordinates - v), 0.0

def FW_away_lazy(objective_function, feasible_region, point_x, step_size_param, phiVal, K = 2.0):
    grad = objective_function.evaluate_grad(point_x.cartesian_coordinates)
    point_a, indexMax, point_v, indexMin = point_x.max_min_vertex(grad)
    # Use old FW vertex.
    if (
        np.dot(grad, point_x.cartesian_coordinates - point_v.cartesian_coordinates) >= np.dot(grad,point_a.cartesian_coordinates - point_x.cartesian_coordinates)
        and np.dot(grad,point_x.cartesian_coordinates - point_v.cartesian_coordinates) >= phiVal[0] / K
    ):
        alphaMax = 1.0
        alpha = step_size(
            objective_function, point_x.cartesian_coordinates, point_v.cartesian_coordinates - point_x.cartesian_coordinates, grad, alphaMax, step_size_param
        )
        if alpha != alphaMax:
            return point_x + alpha*(point_v - point_x), grad.dot(point_x.cartesian_coordinates - point_v.cartesian_coordinates), 0.0
        else:
            return point_v, grad.dot(point_x.cartesian_coordinates - point_v.cartesian_coordinates),  0.0
    else:
        if (
            np.dot(grad, point_a.cartesian_coordinates - point_x.cartesian_coordinates) > np.dot(grad, point_x.cartesian_coordinates - point_v.cartesian_coordinates)
            and np.dot(grad, point_a.cartesian_coordinates - point_x.cartesian_coordinates) >= phiVal[0] / K
        ):
            alphaMax = point_x.barycentric_coordinates[indexMax] / (1.0 - point_x.barycentric_coordinates[indexMax])
            alpha = step_size(
                objective_function, point_x.cartesian_coordinates, point_x.cartesian_coordinates - point_a.cartesian_coordinates, grad, alphaMax, step_size_param
            )
            point_x = point_x + alpha * (point_x - point_a)
            if alpha == alphaMax:
                point_x = point_x.delete_vertex_in_support(indexMax)
            return point_x, grad.dot(point_x.cartesian_coordinates - point_v.cartesian_coordinates), 0.0
        else:
            v = feasible_region.lp_oracle(grad)
            if np.dot(grad, point_x.cartesian_coordinates - v) >= phiVal[0] / K:
                flag, point_v = point_x.is_vertex_in_support(v)
                alphaMax = 1.0
                alpha = step_size(
                    objective_function, point_x.cartesian_coordinates, point_v.cartesian_coordinates - point_x.cartesian_coordinates, grad, alphaMax, step_size_param
                )
                if(flag == False):
                    new_barycentric_coordinates = list(np.zeros(len(point_x.barycentric_coordinates)))
                    new_barycentric_coordinates.append(0.0)
                    point_x = Point(point_x.cartesian_coordinates, tuple(new_barycentric_coordinates), point_v.support)
   
                if alpha != alphaMax:
                    return point_x + alpha*(point_v - point_x), grad.dot(point_x.cartesian_coordinates - point_v.cartesian_coordinates), 0.0
                else:
                    return Point(v, (1.0,), (v,)), grad.dot(point_x.cartesian_coordinates - point_v.cartesian_coordinates), 0.0
            else:
                phiVal[0] = min(grad.dot(point_x.cartesian_coordinates - v), phiVal[0] / 2.0)
                return point_x, grad.dot(point_x.cartesian_coordinates - v), 0.0

class FrankWolfe_backup(_AbstractAlgorithm):
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
            or fw_variant == "lazy"
            or fw_variant == "lazy quick exit"
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
        save_and_output_results=True,
    ):

        if (
            initial_point is None
            or active_set is None
            or lambdas is None
        ):
            x = feasible_region.initial_point.copy()
            active_set = [x]
            lambdas = [1.0]
        else:
            x = initial_point.copy()
            active_set = deepcopy(active_set)
            lambdas = deepcopy(lambdas)
        
        start_time = time.time()
        grad = objective_function.evaluate_grad(x)

        iteration = 0
        duration = 0.0
        f_val = objective_function.evaluate(x)
        v = feasible_region.lp_oracle(grad)
        if(self.fw_variant == "FW" or self.fw_variant == "DIPFW"):
            strong_wolfe_gap = 0.0
        else:
            a, indexMax = feasible_region.away_oracle_old(grad, active_set)
            strong_wolfe_gap = grad.dot(a - v)
        dual_gap = grad.dot(x - v)
        
        if self.fw_variant == "lazy" or self.fw_variant == "lazy quick exit":
            phiVal = [dual_gap]
        
        run_status = (iteration, duration, f_val, dual_gap, strong_wolfe_gap)
        if(save_and_output_results):
            LOGGER.info(
                "Running " + str(self.fw_variant) + "({5}): "
                "iteration = {1:.{0}f}, duration = {2:.{0}f}, f_val = {3:.{0}f}, dual_gap = {4:.{0}f}, strong_wolfe_gap = {5:.{0}f}".format(
                    DISPLAY_DECIMALS, *run_status, self.fw_variant
                )
            )
            run_history = [run_status]
            

        while not exit_criterion.has_met_exit_criterion(run_status):
            if self.fw_variant == "AFW":
                x, dual_gap, strong_wolfe_gap = away_step_fw_backup(
                    objective_function,
                    feasible_region,
                    x,
                    active_set,
                    lambdas,
                    self.step_size_param,
                )
            if self.fw_variant == "PFW":
                x, dual_gap, strong_wolfe_gap = pairwise_step_fw_backup(
                    objective_function,
                    feasible_region,
                    x,
                    active_set,
                    lambdas,
                    self.step_size_param,
                )
            if self.fw_variant == "FW":
                x, dual_gap, strong_wolfe_gap = step_fw_backup(
                    objective_function,
                    feasible_region,
                    x,
                    active_set,
                    lambdas,
                    self.step_size_param,
                )
                
            if self.fw_variant == "lazy":
                x, dual_gap, strong_wolfe_gap = FW_away_lazy_backup(
                    objective_function,
                    feasible_region,
                    x,
                    active_set,
                    lambdas,
                    self.step_size_param,
                    phiVal,
                )
            if self.fw_variant == "lazy quick exit":
                x, dual_gap, strong_wolfe_gap = FW_away_lazy_quick_exit_backup(
                    objective_function,
                    feasible_region,
                    x,
                    active_set,
                    lambdas,
                    self.step_size_param,
                    phiVal,
                )
            if self.fw_variant == "DIPFW":
                x, dual_gap, strong_wolfe_gap = DIPFW_backup(objective_function, feasible_region, x, self.step_size_param)

            iteration += 1
            duration = time.time() - start_time
            f_val = objective_function.evaluate(x)
            run_status = (
                iteration,
                duration,
                f_val,
                dual_gap,
                strong_wolfe_gap,
            )
            if(save_and_output_results):
                LOGGER.info(
                    "Running " + str(self.fw_variant) + ": "
                    "iteration = {1}, duration = {2:.{0}f}, f_val = {3:.{0}f}, dual_gap = {4:.{0}f}, strong_wolfe_gap = {5:.{0}f}".format(
                        DISPLAY_DECIMALS, *run_status
                    )
                )
                run_history.append(run_status)
        if(save_and_output_results):
            return run_history
        else:
            return x, active_set, lambdas

def step_fw_backup(objective_function, feasible_region, x, active_set, lambdas, step_size_param):
    grad = objective_function.evaluate_grad(x)
    v = feasible_region.lp_oracle(grad)
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
    return x + alpha * d, grad.dot(x - v), 0.0

def away_step_fw_backup(objective_function, feasible_region, x, active_set, lambdas, step_size_param):
    assert np.all(np.asarray(lambdas) > 0.0), "Invalid lambda values in AFW."
    grad = objective_function.evaluate_grad(x)
    v = feasible_region.lp_oracle(grad)
    a, indexMax = feasible_region.away_oracle_old(grad, active_set)
    # Choose FW direction, can overwrite index.
    FWGap = grad.dot(x - v)
    StrongFWGap = grad.dot(a - v)
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
    return x + alpha * d, FWGap, StrongFWGap

def pairwise_step_fw_backup( objective_function, feasible_region, x, active_set, lambdas, step_size_param):
    grad = objective_function.evaluate_grad(x)
    v = feasible_region.lp_oracle(grad)
    FW_gap = grad.dot(x - v)
    a, index = feasible_region.away_oracle_old(grad, active_set)
    StrongWolfe_gap = grad.dot(a - v)
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
    return x + alpha * d, FW_gap, StrongWolfe_gap

def DIPFW_backup(objective_function, feasible_region, x, step_size_param):
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
    return x + alpha * d, grad.dot(x - v), 0.0

def FW_away_lazy_backup(objective_function, feasible_region, x, active_set, lambdas, step_size_param, phiVal, K = 2.0):
    grad = objective_function.evaluate_grad(x)
    a, indexMax, v, indexMin = max_min_vertex_backup(grad, active_set)
    # Use old FW vertex.
    if (
        np.dot(grad, x - v) >= np.dot(grad, a - x)
        and np.dot(grad, x - v) >= phiVal[0] / K
    ):
        d = v - x
        alphaMax = 1.0
        # alpha = step_size(objective_function, x, d, grad, i, step_size_param)
        alpha = step_size(
            objective_function, x, d, grad, alphaMax, step_size_param
        )
        
        if alpha != alphaMax:
            lambdas[:] = [i * (1 - alpha) for i in lambdas]
            lambdas[indexMin] += alpha
        else:
            active_set[:] = [v]
            lambdas[:] = [alphaMax]
    else:
        if (
            np.dot(grad, a - x) > np.dot(grad, x - v)
            and np.dot(grad, a - x) >= phiVal[0] / K
        ):
            d = x - a
            alphaMax = lambdas[indexMax] / (
                1.0 - lambdas[indexMax]
            )
            alpha = step_size(
                objective_function, x, d, grad, alphaMax, step_size_param
            )
            
            # alpha = step_size(function, x, d, grad, i, step_size_param)
            lambdas[:] = [i * (1 + alpha) for i in lambdas]
            # Max step, need to delete a vertex.
            if alpha != alphaMax:
                lambdas[indexMax] -= alpha
            else:
                del active_set[indexMax]
                del lambdas[indexMax]
        else:
            v = feasible_region.lp_oracle(grad)
            if np.dot(grad, x - v) >= phiVal[0] / K:
                d = v - x
                
                
                alphaMax = 1.0
                # alpha = step_size(objective_function, x, d, grad, i, step_size_param)
                alpha = step_size(
                    objective_function, x, d, grad, alphaMax, step_size_param
                )
                
                if alpha != alphaMax:
                    lambdas[:] = [i * (1 - alpha) for i in lambdas]
                    active_set.append(v)
                    lambdas.append(alpha)
                else:
                    active_set[:] = [v]
                    lambdas[:] = [alphaMax]
            else:
                phiVal[0] = min(np.dot(grad, x - v), phiVal[0] / 2.0)
                alpha = 0.0
                d = np.zeros(len(x))
    return x + alpha * d, grad.dot(x - v), grad.dot(a - v)


def FW_away_lazy_quick_exit_backup(objective_function, feasible_region, x, active_set, lambdas, step_size_param, phiVal, K = 2.0):
    grad = objective_function.grad(x)
    a, indexMax, v, indexMin = max_min_vertex_quick_exit_backup(
        feasible_region, grad, x, active_set, phiVal, K
    )
    if v is not None and np.dot(grad, x - v) >= phiVal / K:
        d = v - x
        alphaMax = 1.0
        alpha = step_size(
            objective_function, x, d, grad, alphaMax, step_size_param
                )
        # alpha = step_size(function, x, d, grad, i, step_size_param)
        if alpha != alphaMax:
            lambdas[:] = [i * (1 - alpha) for i in lambdas]
            if indexMin is not None:
                lambdas[indexMin] += alpha
            else:
                active_set.append(v)
                lambdas.append(alpha)
        else:
            active_set[:] = [v]
            lambdas[:] = [alphaMax]
    else:
        if a is not None and np.dot(grad, a - x) >= phiVal / K:
            d = x - a
            alphaMax = lambdas[indexMax] / (
                1.0 - lambdas[indexMax]
            )
            alpha = step_size(
                objective_function, x, d, grad, alphaMax, step_size_param
                )
            # alpha = step_size(function, x, d, grad, i, step_size_param)
            lambdas[:] = [i * (1 + alpha) for i in lambdas]
            if alpha != alphaMax:
                lambdas[indexMax] -= alpha
            else:
                del active_set[indexMax]
                del lambdas[indexMax]
        else:
            phiVal = min(np.dot(grad, x - v), phiVal / 2.0)
            alpha = 0.0
            d = np.zeros(len(x))
    return x + alpha * d, grad.dot(x - v), grad.dot(a - v)