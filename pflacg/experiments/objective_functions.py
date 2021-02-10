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
