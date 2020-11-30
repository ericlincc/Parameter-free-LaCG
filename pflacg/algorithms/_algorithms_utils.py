# codeing=utf-8
"""Utility functions and classes for the algorithm module."""


import logging
from scipy.sparse import csc_matrix
import numpy as np


logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s :: %(asctime)s :: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOGGER = logging.getLogger()


DISPLAY_DECIMALS = 4


class ExitCriterion:
    """Stores parameters to determine the exit criterion."""

    def __init__(
        self, criterion_type, criterion_value, criterion_reference=0.0, max_time=1800
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

        valid_criterion_type = ["PG", "DG", "IT"]
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

    def has_met_exit_criterion(self, run_status):
        """
        Returns true if exit criterion has been met, given run_status.

        Parameters
        ----------
        run_status: tuple of (duration, iteration, f_val, dual_gap)
            Values representating algorithm status and progress.
        """

        iteration, duration, f_val, dual_gap = run_status

        if duration > self.max_time:
            return True

        if self.criterion_type == "PG":
            #            print("Value: "  + str(self.criterion_reference))
            primal_gap = f_val - self.criterion_reference
            #            LOGGER.info("Primal gap: {0}".format(primal_gap))
            return primal_gap < self.criterion_value
        elif self.criterion_type == "DG":
            #            LOGGER.info("Wolfe gap: {0}".format(dual_gap))
            return dual_gap < self.criterion_value
        elif self.criterion_type == "IT":
            return iteration >= self.criterion_value
        else:
            raise ValueError("Invalid criterion_type: {0}".format(self.criterion_type))


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


def step_size(function, d, grad, x, type_step="EL", maxStep=None):
    """ Stepsize selection for the algorithm."""
    if type_step == "SS":
        return -np.dot(grad, d) / (function.largest_eigenvalue * np.dot(d, d))
    else:
        # Exact Linesearch.
        return function.line_search(grad, d, x)


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


# Delete some vertices and add the weight to the remaining vertices.
def cullActiveSet(lambdas, activeSet):
    weightRemoved = 0.0
    indexes = np.where(np.asarray(lambdas) < 1.0e-9)[0]
    if len(indexes) != 0:
        for i in range(len(indexes)):
            weightRemoved += lambdas[indexes[-1 - i]]
            del lambdas[indexes[-1 - i]]
            del activeSet[indexes[-1 - i]]
    # Redistribute the weight among the rest of the vertices.
    N = len(activeSet)
    lambdas[:] = [x + weightRemoved / N for x in lambdas]
    return


# Function used in NAGD for LaCG
class funcSimplexLambdaNormalizedEigen:
    # Assemble the matrix from the active set.
    def __init__(self, activeSet, z, A, L, Mu):
        from scipy.sparse.linalg import eigsh

        self.len = len(activeSet)
        Mat = np.zeros((len(activeSet[0]), self.len))
        self.c = Mu * A + L - Mu
        self.b = np.zeros(len(activeSet))
        for i in range(0, self.len):
            Mat[:, i] = activeSet[i]
            self.b[i] = -np.dot(z, activeSet[i])
        self.b /= self.c
        self.M = np.dot(np.transpose(Mat), Mat)
        # Create a sparse matrix from the data.
        self.M = csc_matrix(self.M)
        if self.M.shape == (1, 1):
            self.L = 1.0
            self.Mu = 1.0
        else:
            self.L = 1.0
            self.Mu = 1.0
        #            self.L = eigsh(self.M, 1, which="LM", return_eigenvectors=False)[0]
        #            self.Mu = eigsh(
        #                self.M, 1, sigma=1.0e-10, which="LM", return_eigenvectors=False
        #            )[0]
        return

    def evaluate(self, x):
        return 0.5 * np.dot(x.T, self.M.dot(x)) + np.dot(self.b, x)

    def evaluate_grad(self, x):
        return self.M.dot(x) + self.b

    # Line Search.
    def line_search(self, grad, d):
        return -np.dot(grad, d) / np.dot(d, np.dot(self.M, d))

    def returnM(self):
        return self.M

    @property
    def largest_eigenvalue(self):
        return self.L

    @property
    def smallest_eigenvalue(self):
        return self.Mu

    def FWGap(self, x):
        grad = self.evaluate_grad(x)
        v = np.zeros(len(x))
        minVert = np.argmin(grad)
        v[minVert] = 1.0
        return np.dot(grad, x - v)

    def returnlen(self):
        return self.len

    def update(self, activeSet, z, A, L, Mu):
        self.c = Mu * A + L - Mu
        self.b = np.zeros(len(activeSet))
        for i in range(0, len(activeSet)):
            self.b[i] = -np.dot(z, activeSet[i])
        self.b /= self.c
        return


"""# Simplex Problem Subsolvers
Reference: Nesterov, Yurii. "Introductory lectures on convex programming volume i: Basic course." Lecture notes 3.4 (1998): 5.
"""


def NAGD_SmoothCvx(f, activeSet, tolerance, alpha0):
    if len(activeSet) == 1:
        return activeSet[0].copy(), [1.0]
    from collections import deque

    # Quantities we want to output.
    L = f.largest_eigenvalue
    if len(activeSet) != len(alpha0):
        initPoint = np.ones(len(activeSet)) / len(activeSet)
        x = deque([initPoint], maxlen=2)
        y = deque([initPoint], maxlen=2)
    else:
        x = deque([np.asarray(alpha0)], maxlen=2)
        y = deque([np.asarray(alpha0)], maxlen=2)
    lambdas = deque([0], maxlen=2)
    while f.FWGap(x[-1]) > tolerance:
        x.append(project_onto_simplex(y[-1] - 1 / L * f.evaluate_grad(y[-1])))
        lambdas.append(0.5 * (1 + np.sqrt(1 + 4 * lambdas[-1] * lambdas[-1])))
        step = (1.0 - lambdas[-2]) / lambdas[-1]
        y.append((1.0 - step) * x[-1] + step * x[-2])
    w = np.zeros(len(activeSet[0]))
    for i in range(len(activeSet)):
        w += x[-1][i] * activeSet[i]
    return w, x[-1].tolist()


# NAGD for the Smooth and strongly convex case.
def NAGD_SmoothStrCvx(f, activeSet, tolerance, alpha0):
    if len(activeSet) == 1:
        return activeSet[0].copy(), [1.0]
    from collections import deque

    # Quantities we want to output.
    L = f.largest_eigenvalue
    mu = f.smallest_eigenvalue
    q = mu / L
    if len(activeSet) != len(alpha0):
        initPoint = np.ones(len(activeSet)) / len(activeSet)
        x = deque([initPoint], maxlen=2)
        y = deque([initPoint], maxlen=2)
    else:
        x = deque([np.asarray(alpha0)], maxlen=2)
        y = deque([np.asarray(alpha0)], maxlen=2)
    alpha = deque([np.sqrt(q)], maxlen=2)
    itCount = 0
    while f.FWGap(x[-1]) > tolerance:
        x.append(project_onto_simplex(y[-1] - 1 / L * f.evaluate_grad(y[-1])))
        root = np.roots([1, alpha[-1] ** 2, -alpha[-1] ** 2 - q * alpha[-1]])
        root = root[(root >= 0.0) & (root <= 1.0)]
        assert len(root) != 0, "Root does not meet desired criteria.\n"
        alpha.append(root[0])
        beta = alpha[-2] * (1 - alpha[-2]) / (alpha[-2] ** 2 + alpha[-1])
        y.append(x[-1] + beta * (x[-1] - x[-2]))
        itCount += 1
    w = np.zeros(len(activeSet[0]))
    for i in range(len(activeSet)):
        w += x[-1][i] * activeSet[i]
    return w, x[-1].tolist()


# Performs projections onto the simplex.
def project_onto_simplex(vect, s=1):
    assert s > 0, "Radius s must be strictly positive (%d <= 0)" % s
    (n,) = vect.shape  # will raise ValueError if v is not 1-D
    if vect.sum() == s and np.alltrue(vect >= 0):
        return vect
    v = vect - np.max(vect)
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.count_nonzero(u * np.arange(1, n + 1) > (cssv - s)) - 1
    theta = float(cssv[rho] - s) / (rho + 1)
    w = (v - theta).clip(min=0)
    return w
