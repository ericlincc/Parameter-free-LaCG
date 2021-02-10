# coding=utf-8
"""Utility functions and helper classes for the algorithm module."""


import logging
import sys
import time

import numpy as np

from pflacg.algorithms.project_onto_active_set_jit import (
    accelerated_projected_gradient_descent_over_simplex_jit,
)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s :: %(asctime)s :: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
LOGGER = logging.getLogger()


DISPLAY_DECIMALS = 10


# Helper classes


class Point:
    """Immutable abstraction of a point with respect to its support."""

    def __init__(
        self,
        cartesian_coordinates,
        barycentric_coordinates,
        support,
    ):
        """
        Parameters
        ----------
        cartesian_coordinates: numpy.ndarray
        barycentric_coordinates: tuple(float)
        support: tuple(numpy.ndarray)
        """

        if not len(barycentric_coordinates) == len(support):
            raise ValueError("Lengths of barycentric_coordinates and support not equal")
        if not isinstance(barycentric_coordinates, tuple):
            barycentric_coordinates = tuple(barycentric_coordinates)
        if not isinstance(support, tuple):
            support = tuple(support)

        self.cartesian_coordinates = cartesian_coordinates
        self.barycentric_coordinates = barycentric_coordinates
        self.support = support

    def __add__(self, P):
        """Overloading addition."""

        # Checking if addition is valid
        if not isinstance(P, Point):
            raise TypeError("Cannot add non-Point object with a Point object")
        if not len(self.support) == len(P.support):
            raise ValueError("Cannot add two Points with different supports")
        for vertex1, vertex2 in zip(self.support, P.support):
            if not id(vertex1) == id(vertex2):
                raise ValueError("Cannot add two Points with different supports")

        return Point(
            self.cartesian_coordinates + P.cartesian_coordinates,
            tuple(
                [
                    b1 + b2
                    for b1, b2 in zip(
                        self.barycentric_coordinates, P.barycentric_coordinates
                    )
                ]
            ),
            self.support,
        )

    def __sub__(self, P):
        """Overloading substraction."""

        # Checking if substraction is valid
        if not isinstance(P, Point):
            raise TypeError("Cannot add non-Point object with a Point object")
        if not len(self.support) == len(P.support):
            raise ValueError("Cannot add two Points with different support")
        for vertex1, vertex2 in zip(self.support, P.support):
            if not id(vertex1) == id(vertex2):
                raise ValueError("Cannot add two Points with different support")

        return Point(
            self.cartesian_coordinates - P.cartesian_coordinates,
            tuple(
                [
                    b1 - b2
                    for b1, b2 in zip(
                        self.barycentric_coordinates, P.barycentric_coordinates
                    )
                ]
            ),
            self.support,
        )

    def __mul__(self, t):
        """Overloading multiplication with an int/float."""

        return Point(
            self.cartesian_coordinates * t,
            tuple([i * t for i in self.barycentric_coordinates]),
            self.support,
        )

    __rsub__ = __sub__
    __radd__ = __add__
    __rmul__ = __mul__

    def is_vertex_in_support(self, new_vertex):
        """
        Checks if new_vertex is in the support. If it is, then it returns
        a representation of new_vertex as a Point using the current support.
        Otherwise if it is not in the support it returns a representation of
        new_vertex as a Point using an expanded support (current support plus new_vertex).

        Parameters
        ----------
        new_vertex: np.ndarray
            New vertex to be checked.

        Returns
        -------
        boolean, Point
        """
        for i in range(len(self.support)):
            # if np.array_equal(self.support[i], new_vertex):
            if np.allclose(self.support[i], new_vertex):
                barycentric = np.zeros(len(self.support))
                barycentric[i] = 1.0
                return True, Point(self.support[i], tuple(barycentric), self.support)
        barycentric = np.zeros(len(self.support) + 1)
        barycentric[-1] = 1.0
        new_list = list(self.support)
        new_list.append(new_vertex)
        return False, Point(new_vertex, tuple(barycentric), tuple(new_list))

    def delete_vertex_in_support(self, index):
        barycentric = list(self.barycentric_coordinates)
        support = list(self.support)
        del barycentric[index]
        del support[index]
        return Point(self.cartesian_coordinates, tuple(barycentric), tuple(support))

    def max_min_vertices(self, grad):
        """ Finds the step with the maximum and minimum inner product with the gradient."""

        max_prod = grad.dot(self.support[0])
        min_prod = grad.dot(self.support[0])
        max_ind = 0
        min_ind = 0
        for i in range(len(self.support)):
            if grad.dot(self.support[i]) > max_prod:
                max_prod = grad.dot(self.support[i])
                max_ind = i
            else:
                if grad.dot(self.support[i]) < min_prod:
                    min_prod = grad.dot(self.support[i])
                    min_ind = i
        barycentric_max = np.zeros(len(self.support))
        barycentric_max[max_ind] = 1.0
        barycentric_min = np.zeros(len(self.support))
        barycentric_min[min_ind] = 1.0
        return (
            Point(self.support[max_ind], tuple(barycentric_max), self.support),
            max_ind,
            Point(self.support[min_ind], tuple(barycentric_min), self.support),
            min_ind,
        )


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


def compute_wolfe_gap(point_x, objective_function, feasible_region):
    """Compute the Wolfe gap given a point."""

    grad = objective_function.evaluate_grad(point_x.cartesian_coordinates)
    v = feasible_region.lp_oracle(grad)
    wolfe_gap = grad.dot(point_x.cartesian_coordinates - v)
    return wolfe_gap


def compute_strong_wolfe_gap(point_x, objective_function, feasible_region):
    """Compute w(x, S) given a point with its proper support."""

    grad = objective_function.evaluate_grad(point_x.cartesian_coordinates)
    v = feasible_region.lp_oracle(grad)
    point_a, _ = feasible_region.away_oracle(grad, point_x)
    strong_wolfe_gap = np.dot(grad, point_a.cartesian_coordinates - v)
    return strong_wolfe_gap


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


def argmin_quadratic_over_active_set(
    quadratic_coefficient,
    linear_vector,
    active_set,
    point_reference,
    tolerance_type,
    tolerance,
    base_quadratic=None,
):
    """
    Solves the minimization problem for a quadratic over the convex hull formed by the
    active set, using Nesterov's AGD with projections onto simplex. See Appendix D.2
    for more details.

    The quadratic it tries to solve must be of this form:
    f (u) = <linear_vector,.u> + quadratic_coefficient * ||u||_2^2

    Parameters
    ----------
    quadratic_coefficient: float
        The coefficient of the quadratic subproblem.
    linear_vector: np.ndarray
        The linear vector of the quadratic subproblem.
    active_set: tuple(np.ndarray)
        The tuple of active vertices.
    point_reference: np.ndarray
        Reference point for computing the gradient mapping if
        tolerance_type="gradient mapping".
    tolerance_type: string
        Either "dual gap" or "gradient mapping".
    tolerance: float
        The accuracy to which we need to solve this subproblem to.
    base_quadratic: np.ndarray
        The matrix M^T M where M is the stacked arrays of vertices

    Returns
    -------
    Point
        The approx argmin of the subproblem satisfying the accuracy requirement.
    """

    if tolerance_type not in ["dual gap", "gradient mapping"]:
        raise ValueError("tolerance_type must be either dual_gap or gradient_mapping")

    matrix = np.vstack(active_set)
    if base_quadratic is not None:
        quadratic = (2 * quadratic_coefficient) * base_quadratic
    else:
        quadratic = (2 * quadratic_coefficient) * matrix.dot(matrix.T)
    linear = matrix.dot(linear_vector)
    constant = 0.0
    barycentric_coordinates = accelerated_projected_gradient_descent_over_simplex_jit(
        quadratic=quadratic,
        linear=linear,
        constant=constant,
        active_set=matrix,
        initial_x=np.array(point_reference.barycentric_coordinates),
        reference_x=np.array(point_reference.cartesian_coordinates),
        tolerance_type=tolerance_type,
        tolerance=tolerance,
    )
    if not barycentric_coordinates.any():
        LOGGER.info("Projection step is getting stuck. Terminating..")
        sys.exit()
    cartesian_coordinates = np.zeros(len(active_set[0]))
    for i in range(len(active_set)):
        cartesian_coordinates += barycentric_coordinates[i] * active_set[i]
    return Point(cartesian_coordinates, tuple(barycentric_coordinates), active_set)
