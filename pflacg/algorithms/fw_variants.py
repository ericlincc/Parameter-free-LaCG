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
        self,
        fw_variant,
        step_type,
        tau=2.0,
        eta=0.9,
        smoothness_estimate=1.0e-4,
        sampling_frequency=5,
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
        self.sampling_frequency = sampling_frequency
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
        global_iter=None,
    ):

        if point_initial is None:
            vertex = feasible_region.initial_point.copy()
            point_x = Point(vertex, (1.0,), (vertex,))
        else:
            point_x = point_initial

        start_time = time.time()
        grad = objective_function.evaluate_grad(point_x.cartesian_coordinates)

        iteration = 0
        duration = 0.0
        f_val = objective_function.evaluate(point_x.cartesian_coordinates)
        v = feasible_region.lp_oracle(grad)
        if self.fw_variant == "FW" or self.fw_variant == "DIPFW":
            strong_wolfe_gap = 0.0
        else:
            a, index_max = feasible_region.away_oracle(grad, point_x)
            strong_wolfe_gap = grad.dot(a.cartesian_coordinates - v)

        dual_gap = grad.dot(point_x.cartesian_coordinates - v)

        if self.fw_variant == "lazy" or self.fw_variant == "lazy quick exit":
            phi_val = [dual_gap]

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
            point_x_prev = point_x
            if self.fw_variant == "AFW":
                point_x, dual_gap_prev, strong_wolfe_gap_prev = away_step_fw(
                    objective_function,
                    feasible_region,
                    point_x,
                    self.step_size_param,
                )
            if self.fw_variant == "PFW":
                point_x, dual_gap_prev, strong_wolfe_gap_prev = pairwise_step_fw(
                    objective_function,
                    feasible_region,
                    point_x,
                    self.step_size_param,
                )
            if self.fw_variant == "FW":
                point_x, dual_gap_prev, strong_wolfe_gap_prev = step_fw(
                    objective_function,
                    feasible_region,
                    point_x,
                    self.step_size_param,
                )
                if (
                    iteration % self.sampling_frequency == 0
                    and strong_wolfe_gap_prev is None
                ):
                    grad = objective_function.evaluate_grad(
                        point_x_prev.cartesian_coordinates
                    )
                    v = feasible_region.lp_oracle(grad)
                    point_a, indexMax = feasible_region.away_oracle(grad, point_x_prev)
                    dual_gap_prev = grad.dot(point_x_prev.cartesian_coordinates - v)
                    strong_wolfe_gap_prev = grad.dot(point_a.cartesian_coordinates - v)
            if self.fw_variant == "lazy":
                point_x, dual_gap_prev, strong_wolfe_gap_prev = fw_away_lazy(
                    objective_function,
                    feasible_region,
                    point_x,
                    self.step_size_param,
                    phi_val,
                )
                if iteration % self.sampling_frequency == 0 and (
                    strong_wolfe_gap_prev is None or dual_gap_prev is None
                ):
                    grad = objective_function.evaluate_grad(
                        point_x_prev.cartesian_coordinates
                    )
                    v = feasible_region.lp_oracle(grad)
                    point_a, indexMax = feasible_region.away_oracle(grad, point_x_prev)
                    dual_gap_prev = grad.dot(point_x_prev.cartesian_coordinates - v)
                    strong_wolfe_gap_prev = grad.dot(point_a.cartesian_coordinates - v)
            if self.fw_variant == "DIPFW":
                point_x, dual_gap_prev, strong_wolfe_gap_prev = dipfw(
                    objective_function, feasible_region, point_x, self.step_size_param
                )
            iteration += 1
            duration = time.time() - start_time
            f_val = objective_function.evaluate(point_x.cartesian_coordinates)
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
                        ) + f"\t active set size = {len(point_x.support)}"
                    )
                run_history.append(run_status)
        print("Size of the active set", len(point_x_prev.support))  
        if save_and_output_results:
            return run_history, point_x_prev
        else:
            return point_x_prev, dual_gap_prev, strong_wolfe_gap_prev


# Note that the VANILLA FW algorithm only uses the cartesian coordinates
# and does not use the active set or the barycentric coordinates for anything.
def step_fw(objective_function, feasible_region, point_x, step_size_param):
    grad = objective_function.evaluate_grad(point_x.cartesian_coordinates)
    v = feasible_region.lp_oracle(grad)
    wolfe_gap = grad.dot(point_x.cartesian_coordinates - v)
    d = v - point_x.cartesian_coordinates
    alpha_max = 1.0
    alpha = step_size(
        objective_function,
        point_x.cartesian_coordinates,
        d,
        grad,
        alpha_max,
        step_size_param,
    )
    if alpha != alpha_max:
        flag, point_v = point_x.is_vertex_in_support(v)
        if flag == False:
            new_barycentric_coordinates = list(point_x.barycentric_coordinates)
            new_barycentric_coordinates.append(0.0)
            point_x = Point(
                point_x.cartesian_coordinates,
                tuple(new_barycentric_coordinates),
                point_v.support,
            )
        return point_x + alpha * (point_v - point_x), wolfe_gap, 0.0
    else:
        return Point(v, (1.0,), (v,)), wolfe_gap, 0.0


def away_step_fw(objective_function, feasible_region, point_x, step_size_param):
    grad = objective_function.evaluate_grad(point_x.cartesian_coordinates)
    v = feasible_region.lp_oracle(grad)
    point_a, index_max = feasible_region.away_oracle(grad, point_x)
    wolfe_gap = grad.dot(point_x.cartesian_coordinates - v)
    strong_wolfe_gap = grad.dot(point_a.cartesian_coordinates - v)
    if wolfe_gap > grad.dot(
        point_a.cartesian_coordinates - point_x.cartesian_coordinates
    ):
        alpha_max = 1.0
        alpha = step_size(
            objective_function,
            point_x.cartesian_coordinates,
            v - point_x.cartesian_coordinates,
            grad,
            alpha_max,
            step_size_param,
        )
        if alpha != alpha_max:
            flag, point_v = point_x.is_vertex_in_support(v)
            if flag == False:
                new_barycentric_coordinates = list(point_x.barycentric_coordinates)
                new_barycentric_coordinates.append(0.0)
                point_x = Point(
                    point_x.cartesian_coordinates,
                    tuple(new_barycentric_coordinates),
                    point_v.support,
                )
            return point_x + alpha * (point_v - point_x), wolfe_gap, strong_wolfe_gap
        else:
            return (
                Point(v, (1.0,), (v,)),
                wolfe_gap,
                strong_wolfe_gap,
            )  # TODO: Can we use point_v here instead?
    else:
        alpha_max = point_x.barycentric_coordinates[index_max] / (
            1.0 - point_x.barycentric_coordinates[index_max]
        )
        alpha = step_size(
            objective_function,
            point_x.cartesian_coordinates,
            point_x.cartesian_coordinates - point_a.cartesian_coordinates,
            grad,
            alpha_max,
            step_size_param,
        )
        point_x = point_x + alpha * (point_x - point_a)
        if alpha == alpha_max:
            point_x = point_x.delete_vertex_in_support(index_max)
        return point_x, wolfe_gap, strong_wolfe_gap


def pairwise_step_fw(objective_function, feasible_region, point_x, step_size_param):
    grad = objective_function.evaluate_grad(point_x.cartesian_coordinates)
    v = feasible_region.lp_oracle(grad)
    wolfe_gap = grad.dot(point_x.cartesian_coordinates - v)
    point_a, index_max = feasible_region.away_oracle(grad, point_x)
    strong_wolfe_gap = grad.dot(point_a.cartesian_coordinates - v)
    # Find the weight of the extreme point a in the decomposition.
    alpha_max = point_x.barycentric_coordinates[index_max]
    alpha = step_size(
        objective_function,
        point_x.cartesian_coordinates,
        v - point_a.cartesian_coordinates,
        grad,
        alpha_max,
        step_size_param,
    )
    flag, point_v = point_x.is_vertex_in_support(v)
    if flag == False:
        new_barycentric_coordinates = list(point_x.barycentric_coordinates)
        new_barycentric_coordinates.append(0.0)
        point_x = Point(
            point_x.cartesian_coordinates,
            tuple(new_barycentric_coordinates),
            point_v.support,
        )
        new_barycentric_coordinates = list(point_a.barycentric_coordinates)
        new_barycentric_coordinates.append(0.0)
        point_a = Point(
            point_a.cartesian_coordinates,
            tuple(new_barycentric_coordinates),
            point_v.support,
        )
    point_x = point_x + alpha * (point_v - point_a)
    if alpha == alpha_max:
        point_x = point_x.delete_vertex_in_support(index_max)
    return point_x, wolfe_gap, strong_wolfe_gap


def dipfw(objective_function, feasible_region, point_x, step_size_param):
    grad = objective_function.evaluate_grad(point_x.cartesian_coordinates)
    v = feasible_region.lp_oracle(grad)
    grad_aux = grad.copy()
    for i in range(len(grad_aux)):
        if point_x.cartesian_coordinates[i] == 0.0:
            grad_aux[i] = -1.0e15
    a = feasible_region.lp_oracle(-grad_aux)
    d = v - a
    alpha_max = calculate_stepsize(point_x.cartesian_coordinates, d)
    assert (
        step_size_param["type_step"] == "line_search"
    ), "DIPFW only accepts exact linesearch."
    alpha = step_size(
        objective_function,
        point_x.cartesian_coordinates,
        d,
        grad,
        alpha_max,
        step_size_param,
    )
    new_cartesian = point_x.cartesian_coordinates + alpha * d
    return (
        Point(new_cartesian, (1.0,), (new_cartesian,)),
        grad.dot(point_x.cartesian_coordinates - v),
        0.0,
    )


def fw_away_lazy(
    objective_function, feasible_region, point_x, step_size_param, phi_val, K=2.0
):
    grad = objective_function.evaluate_grad(point_x.cartesian_coordinates)
    point_a, index_max, point_v, index_min = point_x.max_min_vertex(grad)
    # Use old FW vertex.
    if (
        np.dot(grad, point_x.cartesian_coordinates - point_v.cartesian_coordinates)
        >= np.dot(grad, point_a.cartesian_coordinates - point_x.cartesian_coordinates)
        and np.dot(grad, point_x.cartesian_coordinates - point_v.cartesian_coordinates)
        >= phi_val[0] / K
    ):
        alpha_max = 1.0
        alpha = step_size(
            objective_function,
            point_x.cartesian_coordinates,
            point_v.cartesian_coordinates - point_x.cartesian_coordinates,
            grad,
            alpha_max,
            step_size_param,
        )
        if alpha != alpha_max:
            return (
                point_x + alpha * (point_v - point_x),
                None,
                None,
            )
        else:
            return (
                point_v,
                None,
                None,
            )
    else:
        if (
            np.dot(grad, point_a.cartesian_coordinates - point_x.cartesian_coordinates)
            > np.dot(
                grad, point_x.cartesian_coordinates - point_v.cartesian_coordinates
            )
            and np.dot(
                grad, point_a.cartesian_coordinates - point_x.cartesian_coordinates
            )
            >= phi_val[0] / K
        ):
            alpha_max = point_x.barycentric_coordinates[index_max] / (
                1.0 - point_x.barycentric_coordinates[index_max]
            )
            alpha = step_size(
                objective_function,
                point_x.cartesian_coordinates,
                point_x.cartesian_coordinates - point_a.cartesian_coordinates,
                grad,
                alpha_max,
                step_size_param,
            )
            point_x = point_x + alpha * (point_x - point_a)
            if alpha == alpha_max:
                point_x = point_x.delete_vertex_in_support(index_max)
            return (
                point_x,
                None,
                None,
            )
        else:
            v = feasible_region.lp_oracle(grad)
            strong_wolfe_gap = grad.dot(point_a.cartesian_coordinates - v)
            wolfe_gap = grad.dot(point_x.cartesian_coordinates - v)
            if np.dot(grad, point_x.cartesian_coordinates - v) >= phi_val[0] / K:
                flag, point_v = point_x.is_vertex_in_support(v)
                alpha_max = 1.0
                alpha = step_size(
                    objective_function,
                    point_x.cartesian_coordinates,
                    point_v.cartesian_coordinates - point_x.cartesian_coordinates,
                    grad,
                    alpha_max,
                    step_size_param,
                )
                if flag == False:
                    new_barycentric_coordinates = list(point_x.barycentric_coordinates)
                    new_barycentric_coordinates.append(0.0)
                    point_x = Point(
                        point_x.cartesian_coordinates,
                        tuple(new_barycentric_coordinates),
                        point_v.support,
                    )
                if alpha != alpha_max:
                    return (
                        point_x + alpha * (point_v - point_x),
                        wolfe_gap,
                        strong_wolfe_gap,
                    )
                else:
                    return (
                        Point(v, (1.0,), (v,)),
                        wolfe_gap,
                        strong_wolfe_gap,
                    )
            else:
                phi_val[0] = min(
                    grad.dot(point_x.cartesian_coordinates - v), phi_val[0] / 2.0
                )
                return point_x, wolfe_gap, strong_wolfe_gap
