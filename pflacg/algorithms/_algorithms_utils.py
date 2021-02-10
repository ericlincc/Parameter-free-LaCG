# coding=utf-8
"""Utility functions and classes for the algorithm module."""


import logging
import sys
import time

import numpy as np


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s :: %(asctime)s :: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
LOGGER = logging.getLogger()


DISPLAY_DECIMALS = 10


# Helper classes
class ExitCriterion:
    """Stores parameters to determine the exit criterion."""

    def __init__(
        self,
        criterion_type,
        criterion_value,
        criterion_reference=0.0,
        max_time=np.inf,
        max_iter=np.inf,
    ):
        """
        Parameters
        ----------
        criterion_type: str
            Type of criterion to determine the exit condition.
            Choice of "PG", Primal Gap, DG" (Duality Gap), "IT" (#ITerations).
        criterion_value: float
            Threshold to determine when criterion is met.
        criterion_reference: float
            For "PG" only.
        max_time: float
            Maximum execution time allowed.
        max_iter: int
            Maximum number of iterations allowed.
        """

        valid_criterion_type = ["PG", "DG", "IT", "SWG"]
        criterion_value_threshold = 0.0

        # Safety check
        if criterion_type not in valid_criterion_type:
            raise ValueError(
                "Invalid criterion_type: {0}. One of {1}".format(
                    criterion_type, ",".join(valid_criterion_type)
                )
            )
        if criterion_value <= criterion_value_threshold:
            raise ValueError("criterion_value must be greater than 0")

        self.criterion_type = criterion_type
        self.criterion_value = criterion_value
        self.criterion_reference = criterion_reference
        self.max_time = max_time
        self.max_iter = max_iter

    def has_met_exit_criterion(self, run_status):
        """
        Returns true if exit criterion has been met, given run_status.

        Parameters
        ----------
        run_status: tuple of (duration, iteration, f_val, dual_gap, strong_wolfe_gap)
            Values representating algorithm status and progress.

        Returns
        -------
        boolean
        """

        iteration, duration, f_val, dual_gap, strong_wolfe_gap = run_status

        if duration > self.max_time:
            return True
        if iteration > self.max_iter:
            return True

        if self.criterion_type == "PG":
            #            print("Value: "  + str(self.criterion_reference))
            primal_gap = f_val - self.criterion_reference
            #            LOGGER.info("Primal gap: {0}".format(primal_gap))
            return primal_gap < self.criterion_value
        elif self.criterion_type == "DG":
            #            LOGGER.info("Wolfe gap: {0}".format(dual_gap))
            if dual_gap is None:
                return False
            else:
                return dual_gap < self.criterion_value
        elif self.criterion_type == "SWG":
            if strong_wolfe_gap is None:
                return False
            else:
                return strong_wolfe_gap < self.criterion_value
        elif self.criterion_type == "IT":
            return iteration >= self.criterion_value
        else:
            raise ValueError("Invalid criterion_type: {0}".format(self.criterion_type))


# Helper functions


def AwayOracle_reduced(grad, active_set_point):
    aux = np.multiply(grad, np.sign(active_set_point))
    indices = np.where(active_set_point > 0.0)[0]
    v = np.zeros(len(active_set_point), dtype=float)
    indexMax = indices[np.argmax(aux[indices])]
    v[indexMax] = 1.0
    return v, indexMax


def LPOracle_reduced(grad, active_set_point):
    aux = np.multiply(grad, np.sign(active_set_point))
    indices = np.where(active_set_point > 0.0)[0]
    v = np.zeros(len(active_set_point), dtype=float)
    indexMin = indices[np.argmin(aux[indices])]
    v[indexMin] = 1.0
    return v


def compute_strong_wolfe_gap_simplex_reduced(x, objective_function, active_set_point):
    grad = objective_function.evaluate_grad(x)
    v = LPOracle_reduced(grad, active_set_point)
    a, _ = AwayOracle_reduced(grad, x)
    strong_wolfe_gap = np.dot(grad, a - v)
    wolfe_gap = np.dot(grad, x - v)
    return strong_wolfe_gap, wolfe_gap


def compute_wolfe_gap_simplex_reduced(x, objective_function, active_set_point):
    grad = objective_function.evaluate_grad(x)
    v = LPOracle_reduced(grad, active_set_point)
    wolfe_gap = np.dot(grad, x - v)
    return wolfe_gap


def argmin_quadratic_over_active_set_simplex(
    quadratic_coefficient,
    linear_vector,
    active_set_point,
):
    LOGGER.info("Calling argmin")
    indices = np.where(active_set_point > 0.0)[0]
    aux = project_simplex(-linear_vector[indices] / (2.0 * quadratic_coefficient))
    output_point = np.zeros(len(active_set_point))
    output_point[indices] = aux
    return output_point


def project_simplex(x):
    (n,) = x.shape
    if x.sum() == 1.0 and np.alltrue(x >= 0):
        return x
    v = x - np.max(x)
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.count_nonzero(u * np.arange(1, n + 1) > (cssv - 1.0)) - 1
    theta = float(cssv[rho] - 1.0) / (rho + 1)
    w = (v - theta).clip(min=0)
    return w


def step_size(function, x, d, grad, alpha_max, step_size_param):
    """Pick a stepsize."""

    if step_size_param["type_step"] == "line_search":
        alpha = function.line_search(grad, d, x)
    if step_size_param["type_step"] == "adaptive_short_step":
        alpha, L_estimate = backtracking_step_size(
            function,
            d,
            x,
            grad,
            step_size_param["L_estimate"],
            alpha_max,
            tau=step_size_param["tau"],
            eta=step_size_param["eta"],
        )
        step_size_param["L_estimate"] = L_estimate
    return min(alpha, alpha_max)


def backtracking_step_size(function, d, x, grad, L, alpha_max, tau, eta):
    M = L * eta
    d_norm_squared = np.dot(d, d)
    g_t = np.dot(-grad, d)
    alpha = min(g_t / (M * d_norm_squared), alpha_max)
    while (
        function.evaluate(x + alpha * d)
        > function.evaluate(x) - alpha * g_t + 0.5 * M * d_norm_squared * alpha * alpha
    ):
        M *= tau
        alpha = min(g_t / (M * d_norm_squared), alpha_max)
    return alpha, M


def calculate_stepsize(x, d):
    """ Used in the DICG algorithm."""
    assert not np.any(x < 0.0), "There is a negative coordinate."
    index = np.where(x == 0)[0]
    if np.any(d[index] < 0.0):
        return 0.0
    index = np.where(x > 0)[0]
    coeff = np.zeros(len(x))
    for i in index:
        if d[i] < 0.0:
            coeff[i] = -x[i] / d[i]
    val = coeff[coeff > 0]
    if len(val) == 0:
        return 0.0
    else:
        return min(val)
