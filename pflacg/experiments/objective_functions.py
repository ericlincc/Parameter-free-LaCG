# codeing=utf-8
"""This module contains objective functions for the experiements."""


from abc import ABC, abstractmethod
import logging

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.sparse.linalg import splu, eigs


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s :: %(asctime)s :: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
LOGGER = logging.getLogger()


# Helper functions


def calculate_max_and_min_eigenvalues(M):
    """
    Computes the maximum and minimum eigenvalues of a given matrix M.

    Parameters
    ----------
    M: np.ndarray or sparse.csc_matrix

    Returns
    -------
    tuple
        max_eigenvalue, min_eigenvalue
    """
    from scipy.linalg import eigvalsh

    dim = len(M)
    L = eigvalsh(M, eigvals=(dim - 1, dim - 1))[0]
    Mu = eigvalsh(M, eigvals=(0, 0))[0]
    return L, Mu


def random_psd_generator_sparse(dim, sparsity):
    """Random PSD matrix with a given sparsity."""

    mask = np.random.rand(dim, dim) > (1 - sparsity)
    mat = np.random.normal(size=(dim, dim))
    Aux = np.multiply(mat, mask)
    return np.dot(Aux.T, Aux) + np.identity(dim)


def random_psd_generator(dim, Mu, L):
    """TODO: Add description."""

    eigenval = np.zeros(dim)
    eigenval[0] = Mu
    eigenval[-1] = L
    eigenval[1:-1] = np.random.uniform(Mu, L, dim - 2)
    M = np.zeros((dim, dim))
    A = rvs(dim)
    for i in range(dim):
        M += eigenval[i] * np.outer(A[i], A[i])
    return M


# Core functions


class _AbstractObjectiveFunction(ABC):
    """An abstract class to serving function calls."""

    def __init__(self, *args, **kwargs):
        """Initialise abstract function class."""
        pass

    @property
    @abstractmethod
    def dim(self):
        pass

    @property
    @abstractmethod
    def smallest_eigenvalue(self):
        pass

    @property
    @abstractmethod
    def largest_eigenvalue(self):
        pass

    @abstractmethod
    def line_search(self, grad, d):
        pass

    @abstractmethod
    def evaluate(self, x):
        """An abstract method to evaluate function value."""
        pass

    @abstractmethod
    def evaluate_grad(self, x):
        pass


class Quadratic(_AbstractObjectiveFunction):
    """Given matrix M and vector b, f(x) = 0.5 x^T M * x + b^T x."""

    def __init__(self, dim, M, b):
        self._dim = dim
        self.M = M
        self.b = b
        self.L, self.Mu = calculate_max_and_min_eigenvalues(M)

    @property
    def dim(self):
        return self._dim

    @property
    def smallest_eigenvalue(self):
        return self.Mu

    @property
    def largest_eigenvalue(self):
        return self.L

    def line_search(self, grad, d, x):
        return -np.dot(grad, d) / np.dot(d, self.M.dot(d))

    def evaluate(self, x):
        return 0.5 * np.dot(x, self.M.dot(x)) + np.dot(self.b, x)

    def evaluate_grad(self, x):
        return self.M.dot(x) + self.b

    def evaluate_smoothness_inequality(self, x, y):
        x_diff_norm = (x - y) / np.linalg.norm(x - y)
        return 0.5 * np.dot(x_diff_norm, self.M.dot(x_diff_norm))


class QuadraticSparse(_AbstractObjectiveFunction):
    """Given matrix M and vector b, f(x) = 0.5 x^T M * x + b^T x."""

    def __init__(self, dim, M_sparse, b):
        self._dim = dim
        if type(M_sparse).__name__ in ("csr_matrix", "csc_matrix"):
            self.M_sparse = M_sparse
        else:
            raise TypeError("M_sparse should be a csr_matrix.")
        self.b = b
        self.L = np.real(
            eigs(self.M_sparse, k=1, which="LR", return_eigenvectors=False)[0]
        )
        self.Mu = np.real(
            eigs(self.M_sparse, k=1, which="SR", return_eigenvectors=False)[0]
        )

    @property
    def dim(self):
        return self._dim

    @property
    def smallest_eigenvalue(self):
        return self.Mu

    @property
    def largest_eigenvalue(self):
        return self.L

    def line_search(self, grad, d, x):
        return -np.dot(grad, d) / np.dot(d, self.M_sparse.dot(d))

    def evaluate(self, x):
        return 0.5 * np.dot(x, self.M_sparse.dot(x)) + np.dot(self.b, x)

    def evaluate_grad(self, x):
        return self.M_sparse.dot(x) + self.b

    def evaluate_smoothness_inequality(self, x, y):
        x_diff_norm = (x - y) / np.linalg.norm(x - y)
        return 0.5 * np.dot(x_diff_norm, self.M_sparse.dot(x_diff_norm))


class HuberLoss(_AbstractObjectiveFunction):
    """If distance(x - ref) <= radius, then Quadratic, else Linear."""

    def __init__(self, dim, reference_point, radius):
        print("Going in here.")
        self._dim = dim
        #        assert dim == len(reference_point), "Invalid reference in Huber loss."
        self.ref = reference_point
        self.ref = np.ones(self._dim)
        self.ref /= np.linalg.norm(self.ref)
        assert radius > 0.0, "Invalid radius value in Huber loss."
        self.rad = radius

    @property
    def dim(self):
        return self._dim

    @property
    def smallest_eigenvalue(self):
        # Function is globally convex, but locally sharp around reference.
        return 0.0

    @property
    def largest_eigenvalue(self):
        # Function is 1-smooth inside "ball".
        return 1.0

    # Line search procedure is the same regardless of the regions.
    def line_search(self, grad, d, x):
        return -np.dot(x - self.ref, d) / np.dot(d, d)

    def evaluate(self, x):
        dist = np.linalg.norm(x - self.ref)
        if dist <= self.rad:
            return 0.5 * dist * dist
        else:
            return self.rad * (dist - 0.5 * self.rad)

    def evaluate_grad(self, x):
        """Evaluate subgradient."""

        dist = np.linalg.norm(x - self.ref)
        if dist <= self.rad:
            return x - self.ref
        else:
            return self.rad / dist * (x - self.ref)

    def evaluate_smoothness_inequalities(self, x, y):
        f_diff = self.evaluate(y) - self.evaluate(x)
        grad_x = self.evaluate_grad(x)
        y_x = y - x
        return (f_diff - np.dot(grad_x, y_x)) / np.dot(y_x, y_x)


class QuadraticDiagonal(_AbstractObjectiveFunction):
    """TODO: Add docstring."""

    def __init__(self, size, x_opt, Mu=1.0, L=2.0):
        self.len = size
        self.matdim = int(np.sqrt(size))
        self.eigenval = np.zeros(size)
        self.eigenval[0] = Mu
        self.eigenval[-1] = L
        self.eigenval[1:-1] = np.random.uniform(Mu, L, size - 2)
        self.L = L
        self.Mu = Mu
        self.x_opt = x_opt
        self.b = -np.multiply(self.x_opt, self.eigenval)
        self.inv_hess = None

    @property
    def dim(self):
        return self.len

    @property
    def largest_eigenvalue(self):
        return self.L

    @property
    def smallest_eigenvalue(self):
        return self.Mu

    def line_search(self, grad, d, x):
        return -np.dot(grad, d) / np.dot(d, np.multiply(self.eigenval, d))

    def evaluate(self, x):
        return 0.5 * np.dot(x, np.multiply(self.eigenval, x)) + np.dot(self.b, x)

    def evaluate_grad(self, x):
        return np.multiply(x, self.eigenval) + self.b
        # Evaluate the inverse of the Hessian

    def evaluate_smoothness_inequality(self, x, y):
        x_diff_norm = (x - y) / np.linalg.norm(x - y)
        return 0.5 * np.dot(x_diff_norm, np.multiply(self.eigenval, x_diff_norm))


class RegularizedObjectiveFunction(_AbstractObjectiveFunction):
    """Regularize an objective function with a quadratic function.
    f_{delta} (x) = f (x) + sigma * ||x - x_0||_2^2 / 2
    """

    def __init__(self, objective_function, sigma, reference_point):
        self.objective_function = objective_function
        self.sigma = sigma
        self.reference_point = reference_point

        if not self.objective_function.dim == self.reference_point.shape[0]:
            raise ValueError(
                "Dimension of reference_point does not equal that of objective_function"
            )
        self._dim = objective_function.dim

    @property
    def dim(self):
        return self._dim

    @property
    def smallest_eigenvalue(self):
        return self.objective_function.smallest_eigenvalue + self.sigma

    @property
    def largest_eigenvalue(self):
        return self.objective_function.largest_eigenvalue + self.sigma

    def line_search(self, grad, d):
        pass

    def evaluate(self, x):
        x_diff = x - self.reference_point
        return (
            self.objective_function.evaluate(x)
            + self.sigma * np.linalg.norm(x_diff) ** 2 / 2
        )

    def evaluate_grad(self, x):
        x_diff = x - self.reference_point
        return self.objective_function.evaluate_grad(x) + self.sigma * x_diff

    def evaluate_smoothness_inequality(self, x, y):
        return (
            self.objective_function.evaluate_smoothness_inequality(x, y)
            + 0.5 * self.sigma
        )


class GraphicalLasso(_AbstractObjectiveFunction):
    """TODO: Add description."""

    def __init__(self, n, S, lambaVal, delta=0.0):
        self.dim = n
        self.S = S
        self.lambdaVal = lambaVal
        self.delta = 0.0
        return

    # Evaluate function.
    def evaluate(self, X):
        val = X.reshape((self.dim, self.dim))
        return (
            -self.logdetFun(val + self.delta * np.identity(self.dim))
            + np.matrix.trace(np.matmul(self.S, val))
            + 0.5 * self.lambdaVal * np.sum(np.dot(X, X))
        )

    # Evaluate gradient.
    def evaluate_grad(self, X):
        val = X.reshape((self.dim, self.dim))
        return (
            -np.linalg.inv(val + self.delta * np.identity(self.dim)) + self.S
        ).flatten() + self.lambdaVal * X

    # Line Search.
    def line_search(self, grad, d, x, maxStep=None):
        options = {"xatol": 1e-12, "maxiter": 5000000, "disp": 0}

        def InnerFunction(t):  # Hidden from outer code
            return self.fEval(x + t * d)

        if maxStep is None:
            res = minimize_scalar(
                InnerFunction, bounds=(0, 1), method="bounded", options=options
            )
        else:
            res = minimize_scalar(
                InnerFunction, bounds=(0, maxStep), method="bounded", options=options
            )
        return res.x

    def logarithmic_determinant(self, X):
        lu = splu(X)
        diagL = lu.L.diagonal().astype(np.complex128)
        diagU = lu.U.diagonal().astype(np.complex128)
        logdet = np.log(diagL).sum() + np.log(diagU).sum()
        return logdet.real

    def evaluate_smoothness_inequalities(self, x, y):
        f_diff = self.evaluate(y) - self.evaluate(x)
        grad_x = self.evaluate_grad(x)
        y_x = y - x
        return (f_diff - np.dot(grad_x, y_x)) / np.dot(y_x, y_x)


class LogisticRegression(_AbstractObjectiveFunction):
    """TODO: Add description."""

    def __init__(self, n, numSamples, samples, labels, mu=0.0):
        self.samples = samples.copy()
        self.labels = labels.copy()
        self.numSamples = numSamples
        self.dim = n
        self.mu = mu
        return

    def evaluate(self, x):
        aux = np.sum(
            np.logaddexp(
                np.zeros(self.numSamples),
                np.multiply(self.samples.dot(-x), self.labels),
            )
        )
        return aux / self.numSamples + self.mu * np.dot(x, x) / 2.0

    def evaluate_grad(self, x):
        aux = -self.labels / (
            1.0 + np.exp(np.multiply(self.samples.dot(x), self.labels))
        )
        vectors = self.samples.T.multiply(aux).sum(axis=1)
        return np.squeeze(np.asarray(vectors)) / self.numSamples + self.mu * x

    # Line Search.
    def line_search(self, grad, d, x, maxStep=None):
        options = {"xatol": 1e-12, "maxiter": 50000, "disp": 0}

        def InnerFunction(t):  # Hidden from outer code
            return self.fEval(x + t * d)

        if maxStep is None:
            res = minimize_scalar(
                InnerFunction, bounds=(0, 1), method="bounded", options=options
            )
        else:
            res = minimize_scalar(
                InnerFunction, bounds=(0, maxStep), method="bounded", options=options
            )
        return res.x

    def evaluate_smoothness_inequalities(self, x, y):
        f_diff = self.evaluate(y) - self.evaluate(x)
        grad_x = self.evaluate_grad(x)
        y_x = y - x
        return (f_diff - np.dot(grad_x, y_x)) / np.dot(y_x, y_x)
