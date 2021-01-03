# codeing=utf-8
"""Utility functions and classes for the algorithm module."""


import logging
from scipy.sparse import csc_matrix
import numpy as np
import time


logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s :: %(asctime)s :: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOGGER = logging.getLogger()


DISPLAY_DECIMALS = 10


class ExitCriterion:
    """Stores parameters to determine the exit criterion."""

    def __init__(
        self, criterion_type, criterion_value, criterion_reference=0.0, max_time=1800, max_iter=1000
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
            return dual_gap < self.criterion_value
        elif self.criterion_type == "SWG":
            return strong_wolfe_gap < self.criterion_value
        elif self.criterion_type == "IT":
            return iteration >= self.criterion_value
        else:
            raise ValueError("Invalid criterion_type: {0}".format(self.criterion_type))

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


# def step_size(function, d, grad, x, type_step="EL", maxStep=None):
#     """ Stepsize selection for the algorithm."""
#     if type_step == "SS":
#         return -np.dot(grad, d) / (function.largest_eigenvalue * np.dot(d, d))
#     else:
#         # Exact Linesearch.
#         return function.line_search(grad, d, x)


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

def project_onto_active_set(
    quadratic_coefficient,
    linear_vector,
    constant,
    active_set,
    barycentric_coordinates,
    stopping_criterion,
    threshold=0.0,
    time_limit=np.inf,
    max_steps = np.inf,
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
    constant : float
        Constant term in the objective function (it is not involved in any of
                                                 the computations).
    active_set : list of numpy arrays
        Contains the vertices over which we will minimize.
    barycentric_coordinates : list of float
        Contains the weights associated with the active set.
    stopping_criterion : class
        Class that contains a function "evaluate" that determines if the exit 
        criteria has been met.
    threshold : float
        Threshold for the barycentric coordinates. Any value in the barycentric
        coordinates below this value will be set to zero and the corresponding
        weight will be assigned to the other vertices.
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
    quadratic = quadratic_coefficient*matrix.dot(matrix.T)
    linear = matrix.dot(linear_vector)

    # Create objective function and feasible region.
    from pflacg.experiments.feasible_regions import ProbabilitySimplexPolytope
    feas_reg = ProbabilitySimplexPolytope(len(active_set))
    projection_objective_fun = projection_objective_function(quadratic, linear, constant)
    x, polished_barycentric_coordinates, gap_values1 = accelerated_projected_gradient_descent(
        projection_objective_fun,
        feas_reg,
        active_set,
        stopping_criterion,
        barycentric_coordinates,
        time_limit = time_limit,
        max_iteration = max_steps
        )
    return remove_vertives(active_set, polished_barycentric_coordinates, threshold)

class stopping_criterion:
    """
    Stopping criterion for the projection

    Parameters
    ----------
    tolerance : float.
        If the tolerance is not none, it will stop when the FW gap is below 
        this given value
    reference_point: numpy array
        If tolerance is None, will check if FW gap is smaller than the 
        gradient mapping with respect to this point.
    coefficient : float
        Coefficient for use with reference_point.
    """
    def __init__(self, tolerance = None, reference_point = None, coefficient = None):
        self.tolerance = tolerance
        self.reference_point = reference_point
        self.coefficient = coefficient
        
    def evaluate(self, x, FW_gap):
        if(self.tolerance is not None):
            return self.tolerance < FW_gap
        else:
            return self.coefficient*np.linalg.norm(x - self.reference_point)**2 < FW_gap


def remove_vertives(active_set, barycentric_coordinates, threshold):
    new_active_set = []
    new_barycentric_coordinates = []
    for i in range(len(active_set)):
        if barycentric_coordinates[i] > threshold:
            new_active_set.append(active_set[i])
            new_barycentric_coordinates.append(barycentric_coordinates[i])
    aux = sum(new_barycentric_coordinates)
    new_barycentric_coordinates = [
        x + (1.0 - aux) / len(new_barycentric_coordinates)
        for x in new_barycentric_coordinates
    ]
    x = np.zeros(active_set[0].shape)
    for i in range(len(new_active_set)):
        x += new_barycentric_coordinates[i] * new_active_set[i]
    return x, new_active_set, new_barycentric_coordinates

def accelerated_projected_gradient_descent(
    f,
    feasible_region,
    active_set,
    stopping_criterion,
    alpha0,
    time_limit=60,
    max_iteration=100,
):
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
    from collections import deque
    # Quantities we want to output.
    L = f.largest_eigenvalue()
    mu = f.smallest_eigenvalue()
    initial_point = np.asarray(alpha0)
    x = deque([initial_point], maxlen=2)
    y = initial_point
    q = mu / L
    if(mu < 1.0e-3):
        alpha = deque([0], maxlen=2)
    else:
        alpha = deque([np.sqrt(q)], maxlen=2)
    grad = f.gradient(x[-1])
    FWGap = grad.dot(x[-1] - feasible_region.lp_oracle(grad))
    time_ref = time.time()
    it_count = 0
    gap_values = [FWGap]
    while stopping_criterion.evaluate(x[-1], FWGap):
        print("FW gap: ", FWGap)
        x.append(feasible_region.projection(y - 1 / L * f.gradient(y)))
        if(mu < 1.0e-3):
            alpha.append(0.5 * (1 + np.sqrt(1 + 4 * alpha[-1] * alpha[-1])))
            beta = (alpha[-2] - 1.0) / alpha[-1]
        else:
            root = np.roots([1, alpha[-1] ** 2 - q, -alpha[-1] ** 2])
            root = root[(root >= 0.0) & (root < 1.0)]
            assert len(root) != 0, "Root does not meet desired criteria.\n"
            alpha.append(root[0])
            beta = alpha[-2] * (1 - alpha[-2]) / (alpha[-2] ** 2 - alpha[-1])
        y = x[-1] + beta * (x[-1] - x[-2])
        grad = f.gradient(x[-1])
        FWGap = grad.dot(x[-1] - feasible_region.lp_oracle(grad))
        it_count += 1
        if time.time() - time_ref > time_limit or it_count > max_iteration:
            break
        gap_values.append(FWGap)
    w = np.zeros(len(active_set[0]))
    for i in range(len(active_set)):
        w += x[-1][i] * active_set[i]
    return w, x[-1].tolist(), gap_values

class projection_objective_function:
    import numpy as np
    from scipy.sparse import issparse

    def __init__(self, Q, b, constant):
        self.Q = Q.copy()
        self.b = b.copy()
        self.constant = constant

        from scipy.linalg import eigvalsh

        w = eigvalsh(self.Q)
        self.L = np.max(w)
        self.Mu = np.min(w)
        return

    # Evaluate function.
    def f(self, x):
        return self.b.dot(x) + 0.5 * x.T.dot(self.Q.dot(x)) + self.constant

    # Evaluate gradient.
    def gradient(self, x):
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