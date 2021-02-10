# codeing=utf-8
"""This module contains feasible region classes for the experiements."""

from abc import ABC, abstractmethod
import logging
import math

import numpy as np
from scipy.sparse.linalg import eigsh
from scipy.sparse import csc_matrix


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s :: %(asctime)s :: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
LOGGER = logging.getLogger()


class _AbstractFeasibleRegion(ABC):
    """An abstract class to construct feasible region objects."""

    def __init__(self, *args, **kwargs):
        """Initialise abstract feasible region class."""
        pass

    @property
    def initial_point(self):
        raise NotImplementedError(
            "Initial point has not been set for this feasible region!"
        )

    @property
    def initial_active_set(self):
        raise NotImplementedError(
            "Initial active set has not been set for this feasible region!"
        )

    @abstractmethod
    def lp_oracle(self, d):
        """
        Compute the linear oracle.

        Parameters
        ----------
        d : np.ndarray
            The direction.

        Returns
        -------
        np.ndarray
        """

        pass

    @abstractmethod
    def away_oracle(self, d, point_x):
        """
        Compute the away oracle.

        Parameters
        ----------
        d: np.ndarray
            The direction.
        point_x: Point
            Point x with its proper support.

        Returns
        -------
        Point
        """

        pass

    def projection(self, x, accuracy):
        raise NotImplementedError(
            "Projection has not been implemented for this feasible region!"
        )


class ProbabilitySimplexPolytope(_AbstractFeasibleRegion):
    def __init__(self, dim):
        self.dim = dim

    @property
    def initial_point(self):
        v = np.zeros(self.dim)
        v[0] = 1.0
        return v

    @property
    def initial_active_set(self):
        return [self.initial_point()]

    def lp_oracle(self, x):
        v = np.zeros(len(x), dtype=float)
        v[np.argmin(x)] = 1.0
        return v

    # Input is the vector over which we calculate the inner product.
    def away_oracle(self, grad, x):
        aux = np.multiply(grad, np.sign(x))
        indices = np.where(x > 0.0)[0]
        v = np.zeros(len(x), dtype=float)
        indexMax = indices[np.argmax(aux[indices])]
        v[indexMax] = 1.0
        return v, indexMax

    def projection(self, x):
        (n,) = x.shape  # will raise ValueError if v is not 1-D
        if x.sum() == 1.0 and np.alltrue(x >= 0):
            return x
        v = x - np.max(x)
        u = np.sort(v)[::-1]
        cssv = np.cumsum(u)
        rho = np.count_nonzero(u * np.arange(1, n + 1) > (cssv - 1.0)) - 1
        theta = float(cssv[rho] - 1.0) / (rho + 1)
        w = (v - theta).clip(min=0)
        return w
