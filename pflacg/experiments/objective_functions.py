# codeing=utf-8
"""This module contains objective functions for the experiements."""

from abc import ABC, abstractmethod
import numpy as np
from scipy.sparse import csc_matrix


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
    L = eigvalsh(M, eigvals=(dim - 1, dim - 1))[0]  # TODO: what does eigvals here do?
    Mu = eigvalsh(M, eigvals=(0, 0))[0]
    return L, Mu


def random_psd_generator_sparse(dim, sparsity):
    """Random PSD matrix with a given sparsity."""

    mask = np.random.rand(dim, dim) > (1 - sparsity)
    mat = np.random.normal(size=(dim, dim))
    Aux = np.multiply(mat, mask)
    return np.dot(Aux.T, Aux) + np.identity(dim)


def random_psd_generator(dim, Mu, L):
    # TODO: Alex, can you add docstring here describing what this does?
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
    def update_self(self, *args, **kwargs):
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

    def update_self(self):
        pass

    def evaluate(self, x):
        return 0.5 * np.dot(x, self.M.dot(x)) + np.dot(self.b, x)

    def evaluate_grad(self, x):
        return self.M.dot(x) + self.b


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

    def update_self(self):
        pass


class RegularizedObjectiveFunction(_AbstractObjectiveFunction):
    """Regularize an objective function with a quadratic function.
    f_{delta} (x) = f (x) + sigma * ||x - x_0||_2^2 / 2
    """

    def __init__(self, objective_function, sigma, reference_point):
        self.objective_function = objective_function
        self.sigma = sigma
        self.reference_point = reference_point

        if not self.objective_function.dim == self.reference_point.shape[0]:
            raise ValueError("Dimension of reference_point does not equal that of objective_function")
        self._dim = objective_function.dim

    @property
    def dim(self):
        return self._dim

    @property
    def smallest_eigenvalue(self):
        return self.objective_function.smallest_eigenvalue + sigma

    @property
    def largest_eigenvalue(self):
        return self.objective_function.largest_eigenvalue + sigma

    def line_search(self, grad, d):
        pass

    def update_self(self, *args, **kwargs):
        pass

    def evaluate(self, x):
        _x = x - self.reference_point
        return self.objective_function.evaluate(x) + self.sigma / 2.0 * np.dot(_x, _x)

    def evaluate_grad(self, x):
        _x = x - self.reference_point
        return self.objective_function.evaluate_grad(x) + self.sigma * _x
