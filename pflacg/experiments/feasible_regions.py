# codeing=utf-8
"""This module contains feasible region classes for the experiements."""

from abc import ABC, abstractmethod
import logging
import math

import numpy as np

from pflacg.experiments.experiments_helper import max_vertex_old, max_vertex


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
        raise NotImplementedError("Initial point has not been set for this feasible region!")

    @property
    def initial_active_set(self):
        raise NotImplementedError("Initial active set has not been set for this feasible region!")

    @abstractmethod
    def lp_oracle(self, d):
        pass

    @abstractmethod
    def away_oracle(self, d, active_vertices):
        pass

    def projection(self, x, accuracy):
        raise NotImplementedError("Projection has not been implemented for this feasible region!")


class ConvexHull(_AbstractFeasibleRegion):
    """Convex hull given a set of vertice."""

    def __init__(self, vertices):
        self.vertices = vertices

    def lp_oracle(self, d):
        return max_vertex(-d, self.vertices)

    def away_oracle(self, d, active_vertices):
        return max_vertex(d, active_vertices)
    
    def away_oracle_old(self, d, active_vertices):
        return max_vertex_old(d, active_vertices)

    def projection(self, x, accuracy):
        pass

class BirkhoffPolytope(_AbstractFeasibleRegion):
    def __init__(self, dim):
        self.dim = dim
        self.mat_dim = int(np.sqrt(dim))

    @property
    def initial_point(self):
        return np.identity(self.mat_dim).flatten()

    @property
    def initial_active_set(self):
        return [self.initial_point()]

    def lp_oracle(self, x):
        from scipy.optimize import linear_sum_assignment

        objective = x.reshape((self.mat_dim, self.mat_dim))
        matching = linear_sum_assignment(objective)
        solution = np.zeros((self.mat_dim, self.mat_dim))
        solution[matching] = 1
        return solution.reshape(self.dim)

    def away_oracle(self, grad, active_vertex):
        return max_vertex(grad, active_vertex)

    def away_oracle_old(self, grad, active_vertex):
        return max_vertex_old(grad, active_vertex)

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

    #     #This is a faster implementation of the away oracle without having to loop through active set.
    #    def away_oracle(self, grad, x):
    #        aux = np.multiply(grad, np.sign(x))
    #        indices = np.where(x > 0.0)[0]
    #        v = np.zeros(len(x), dtype=float)
    #        index_max = indices[np.argmax(aux[indices])]
    #        v[index_max] = 1.0
    #        return v, index_max

    def away_oracle(self, grad, active_vertex):
        return max_vertex(grad, active_vertex)
    
    def away_oracle_old(self, grad, active_vertex):
        return max_vertex_old(grad, active_vertex)


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


class L1UnitBallPolytope(_AbstractFeasibleRegion):
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
        max_ind = np.argmax(np.abs(x))
        v[max_ind] = -1.0 * np.sign(x[max_ind])
        return v

    def away_oracle(self, grad, active_vertex):
        return max_vertex(grad, active_vertex)
    
    def away_oracle_old(self, grad, active_vertex):
        return max_vertex_old(grad, active_vertex)

    def projection(self, x):
        u = np.abs(x)
        if u.sum() <= 1.0:
            return x
        w = self.projectionSimplex(u)
        w *= np.sign(x)
        return w

    def projectionSimplex(self, x):
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


class L2UnitBallPolytope(_AbstractFeasibleRegion):
    def __init__(self, dim):
        self.dim = dim

    @property
    def initial_point(self):
        v = np.ones(self.dim)
        return v / np.linalg.norm(v)

    @property
    def initial_active_set(self):
        return [self.initial_point()]

    def lp_oracle(self, x):
        return -x / np.linalg.norm(x)

    def away_oracle(self, grad, active_vertex):
        return max_vertex(grad, active_vertex)
    
    def away_oracle_old(self, grad, active_vertex):
        return max_vertex_old(grad, active_vertex)
    
    def projection(self, x):
        return x / np.linalg.norm(x)
