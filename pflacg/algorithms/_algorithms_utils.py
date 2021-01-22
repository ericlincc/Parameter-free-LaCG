# codeing=utf-8
"""Utility functions and classes for the algorithm module."""


import logging
import time

import numpy as np
from scipy.sparse import csc_matrix


from pflacg.algorithms.project_onto_active_set_jit import (
    accelerated_projected_gradient_descent_over_simplex_jit,
)


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
            raise ValueError("Cannot add two Points with different support")
        for vertex1, vertex2 in zip(self.support, P.support):
            if not id(vertex1) == id(vertex2):
                raise ValueError("Cannot add two Points with different support")

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

    # Checks if new_vertex is in the support. If it is, then it returns
    # a representation of new_vertex as a Point using the current support.
    # Otherwise if it is not in the support it returns a representation of
    # new_vertex as a Point using an expanded support (current support plus new_vertex).
    def is_vertex_in_support(self, new_vertex):
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

    def max_min_vertex(self, grad):
        maxProd = grad.dot(self.support[0])
        minProd = grad.dot(self.support[0])
        maxInd = 0
        minInd = 0
        for i in range(len(self.support)):
            if grad.dot(self.support[i]) > maxProd:
                maxProd = grad.dot(self.support[i])
                maxInd = i
            else:
                if grad.dot(self.support[i]) < minProd:
                    minProd = grad.dot(self.support[i])
                    minInd = i
        barycentric_max = np.zeros(len(self.support))
        barycentric_max[maxInd] = 1.0
        barycentric_min = np.zeros(len(self.support))
        barycentric_min[minInd] = 1.0
        return (
            Point(self.support[maxInd], tuple(barycentric_max), self.support),
            maxInd,
            Point(self.support[minInd], tuple(barycentric_min), self.support),
            minInd,
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
        run_status: tuple of (duration, iteration, f_val, dual_gap)
            Values representating algorithm status and progress.
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
    grad = objective_function.evaluate_grad(point_x.cartesian_coordinates)
    v = feasible_region.lp_oracle(grad)
    wolfe_gap = grad.dot(point_x.cartesian_coordinates - v)
    return wolfe_gap


def compute_strong_wolfe_gap(point_x, objective_function, feasible_region):
    grad = objective_function.evaluate_grad(point_x.cartesian_coordinates)
    v = feasible_region.lp_oracle(grad)
    point_a, _ = feasible_region.away_oracle(grad, point_x)
    strong_wolfe_gap = np.dot(grad, point_a.cartesian_coordinates - v)
    return strong_wolfe_gap


def line_search(self, grad, d, x):
    return -np.dot(grad, d) / np.dot(d, self.M.dot(d))


# Pick a stepsize.
def step_size(function, x, d, grad, alpha_max, step_size_param):
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


# TODO: Are we using this?
# Provides an initial estimate for the smoothness parameter.
def smoothnessEstimate(x0, function):
    L = 1.0e-3
    while function.f(x0 - function.grad(x0) / L) > function.f(x0):
        L *= 1.5
    return L


def new_vertex_fail_fast(vertex, active_set):
    """ Find if x is in the active set."""
    for i in range(len(active_set)):
        # Compare succesive indices.
        for j in range(len(active_set[i])):
            if active_set[i][j] != vertex[j]:
                break
        if j == len(active_set[i]) - 1:
            return False, i
    return True, np.nan


def delete_vertex_index(index, active_set, lambdas):
    """Deletes the extremepoint from the list active_set, and from lambdas. """
    del active_set[index]
    del lambdas[index]
    return


def max_min_vertices(grad, active_set):
    """ Finds the step with the maximum and minimum inner product with the gradient."""
    maxProd = np.dot(active_set[0], grad)
    minProd = np.dot(active_set[0], grad)
    maxInd = 0
    minInd = 0
    for i in range(len(active_set)):
        if np.dot(active_set[i], grad) > maxProd:
            maxProd = np.dot(active_set[i], grad)
            maxInd = i
        else:
            if np.dot(active_set[i], grad) < minProd:
                minProd = np.dot(active_set[i], grad)
                minInd = i
    return active_set[maxInd], maxInd, active_set[minInd], minInd


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


# TODO: Optimise and clean up these projection methods
# TODO: Need more consistent variable naming below
def argmin_quadratic_over_active_set(
    quadratic_coefficient,
    linear_vector,
    active_set,
    reference_point,
    tolerance_type,
    tolerance,
    time_limit=np.inf,
    max_steps=np.inf,
    use_numba=True,
):

    LOGGER.info("Calling argmin")

    if tolerance_type not in ["dual gap", "gradient mapping"]:
        raise ValueError("tolerance_type must be either dual_gap or gradient_mapping")

    if use_numba:
        matrix = np.vstack(active_set)
        quadratic = 2 * quadratic_coefficient * matrix.dot(matrix.T)
        linear = matrix.dot(linear_vector)
        constant = 0.0
        barycentric_coordinates = (
            accelerated_projected_gradient_descent_over_simplex_jit(
                quadratic=quadratic,
                linear=linear,
                constant=constant,
                active_set=matrix,
                initial_x=np.array(
                    reference_point.barycentric_coordinates
                ),  # TODO: make sure that this is an np array
                reference_x=np.array(reference_point.cartesian_coordinates),
                tolerance_type=tolerance_type,
                tolerance=tolerance,
            )
        )
        if not barycentric_coordinates.any():
            raise Exception("projection step is getting stuck")
        cartesian_coordinates = np.zeros(len(active_set[0]))
        for i in range(len(active_set)):
            cartesian_coordinates += barycentric_coordinates[i] * active_set[i]
        return Point(cartesian_coordinates, tuple(barycentric_coordinates), active_set)

    if tolerance_type == "dual gap":
        stopping_criterion = StoppingCriterion(
            tolerance=tolerance,
            reference_point=None,
            coefficient=None,
        )
    elif tolerance_type == "gradient mapping":
        stopping_criterion = StoppingCriterion(
            tolerance=None,
            reference_point=reference_point,
            coefficient=tolerance,
        )
    else:
        raise ValueError("Invalid tolerance_type.")
    x, active_set, barycentric_coordinates = project_onto_active_set(
        quadratic_coefficient,
        linear_vector,
        active_set,
        reference_point.barycentric_coordinates,
        stopping_criterion,
    )

    return Point(x, tuple(barycentric_coordinates), tuple(active_set))


def project_onto_active_set(
    quadratic_coefficient,
    linear_vector,
    active_set,
    barycentric_coordinates,
    stopping_criterion,
    time_limit=np.inf,
    max_steps=np.inf,
):
    """
    Minimizes the objective function:
        f(u) = quadratic_coefficient*u^Tu + linear_vector^Tu + constant
    over the vertices in active set until the stopping criterion is satisfied.

    Parameters
    ----------
    quadratic_coefficient : float.
        Coefficient that accompanies the quadratic term in objective function.
    linear_vector: numpy array
        Numpy array that accompanies linear term in objective function
    active_set : list of numpy arrays
        Contains the vertices over which we will minimize.
    barycentric_coordinates : list of float
        Contains the weights associated with the active set.
    stopping_criterion : class
        Class that contains a function "evaluate" that determines if the exit
        criteria has been met.
    time_limit : float
        Maximum time for which the algorithm will be run.
    max_steps : int
        Maximum number of iterations on the projection algorithm.

    Returns
    -------
    x : numpy array
        Outputted solution with primal gap below the target tolerance
    new_active_set: list numpy array
        Contains the new active set for the point x.
    new_barycentric_coordinates: list float
        Contains the barycentric coordinates for x w.r.t. the new_active_set
    """
    matrix = np.vstack(active_set)
    quadratic = 2 * quadratic_coefficient * matrix.dot(matrix.T)
    linear = matrix.dot(linear_vector)

    # Create objective function and feasible region.
    from pflacg.experiments.feasible_regions import ProbabilitySimplexPolytope

    feas_reg = ProbabilitySimplexPolytope(len(active_set))
    projection_objective_fun = projection_objective_function(quadratic, linear, 0.0)
    x, x_barycentric_coordinates = accelerated_projected_gradient_descent(
        projection_objective_fun,
        feas_reg,
        active_set,
        stopping_criterion,
        np.array(barycentric_coordinates),
        time_limit=time_limit,
        max_iteration=max_steps,
    )
    return x, active_set, x_barycentric_coordinates


class StoppingCriterion:
    """
    Stopping criterion for the projection

    Parameters
    ----------
    tolerance : float.
        If the tolerance is not none, it will stop when the FW gap is below
        this given value
    reference_point: Point
        If tolerance is None, will check if FW gap is smaller than the
        gradient mapping with respect to this point.
    coefficient : float
        Coefficient for use with reference_point.
    """

    def __init__(self, tolerance=None, reference_point=None, coefficient=None):
        self.tolerance = tolerance
        self.reference_point = reference_point
        self.coefficient = coefficient

    def evaluate(self, x, wolfe_gap):
        if self.tolerance is not None:
            return self.tolerance < wolfe_gap
        else:
            active_set = self.reference_point.support
            w = np.zeros(len(active_set[0]))
            for i in range(len(active_set)):
                w += x[i] * active_set[i]
            return (
                self.coefficient
                * np.linalg.norm(w - self.reference_point.cartesian_coordinates) ** 2
                < wolfe_gap
            )


def accelerated_projected_gradient_descent(
    f,
    feasible_region,
    active_set,
    stopping_criterion,
    initial_x,
    time_limit=60,
    max_iteration=100,
):
    # TODO: The below description is not to date.
    """
    Run Nesterov's accelerated projected gradient descent.

    References
    ----------
    Nesterov, Y. (2018). Lectures on convex optimization (Vol. 137).
    Berlin, Germany: Springer. (Constant scheme II, Page 93)

    Parameters
    ----------
    x0 : numpy array.
        Initial point.
    function: function being minimized
        Function that we will minimize. Gradients are computed through a
        function.grad(x) function that returns the gradient at x as a
        numpy array.
    feasible_region : feasible region function.
        Returns projection oracle of a point x onto the feasible region,
        which are computed through the function feasible_region.project(x).
        Additionally, a LMO is used to compute the Frank-Wolfe gap (used as a
        stopping criterion) through the function
        feasible_region.linear_optimization_oracle(grad) function, which
        minimizes <x, grad> over the feasible region.
    tolerance : float
        Frank-Wolfe accuracy to which the solution is outputted.

    Returns
    -------
    x : numpy array
        Outputted solution with primal gap below the target tolerance
    """

    # Quantities we want to output.
    L = f.largest_eigenvalue()
    mu = f.smallest_eigenvalue()
    q = mu / L
    x = initial_x
    y = initial_x
    if (mu < 1.0e-3) or q == 1:
        alpha = 0
    else:
        alpha = np.sqrt(q)
    grad = f.evaluate_grad(x)
    wolfe_gap = grad.dot(x - feasible_region.lp_oracle(grad))
    time_ref = time.time()
    it_count = 0

    while stopping_criterion.evaluate(x, wolfe_gap):
        x_ = x
        x = feasible_region.projection(y - 1 / L * f.evaluate_grad(y))
        if (mu < 1.0e-3) or q == 1:
            alpha_ = alpha
            alpha = 0.5 * (1 + np.sqrt(1 + 4 * alpha_ * alpha_))
            beta = (alpha_ - 1.0) / alpha
        else:
            root = np.roots([1, alpha ** 2 - q, -(alpha ** 2)])
            root = root[(root >= 0.0) & (root < 1.0)]
            assert len(root) != 0, "Root does not meet desired criteria.\n"
            _alpha = alpha
            alpha = root[0]
            beta = _alpha * (1 - _alpha) / (_alpha ** 2 - alpha)
        y = x + beta * (x - x_)
        grad = f.evaluate_grad(x)
        _wolfe_gap = wolfe_gap
        wolfe_gap = grad.dot(x - feasible_region.lp_oracle(grad))
        it_count += 1
        if time.time() - time_ref > time_limit or it_count > max_iteration:
            break
        if np.isclose(wolfe_gap, _wolfe_gap) and wolfe_gap < 1e-13:
            raise Exception("projection step is getting stuck")
    w = np.zeros(len(active_set[0]))
    for i in range(len(active_set)):
        w += x[i] * active_set[i]
    return w, x.tolist()


class projection_objective_function:
    import numpy as np
    from scipy.sparse import issparse

    def __init__(self, Q, b, constant):
        self.Q = Q.copy()  # TODO: Probably doesn't need to copy this Q
        self.b = b.copy()  # TODO: Probably doesn't need to copy this b
        self.constant = constant

        from scipy.linalg import eigvalsh

        w = eigvalsh(self.Q)
        self.L = np.max(w)
        self.Mu = np.min(w)
        return

    # Evaluate function.
    def evaluate(self, x):
        return self.b.dot(x) + 0.5 * x.T.dot(self.Q.dot(x)) + self.constant

    # Evaluate gradient.
    def evaluate_grad(self, x):
        return self.b + self.Q.dot(x)

    # Line Search.
    def line_search(self, grad, d, x, maxStep=None):
        alpha = -d.dot(grad) / d.T.dot(self.Q.dot(d))
        if maxStep is None:
            return min(alpha, 1.0)
        else:
            return min(alpha, maxStep)

    def largest_eigenvalue(self):
        return self.L

    def smallest_eigenvalue(self):
        return self.Mu
