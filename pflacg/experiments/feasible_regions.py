# codeing=utf-8
"""This module contains feasible region classes for the experiements."""

from abc import ABC, abstractmethod
import logging
import math

import numpy as np

from pflacg.experiments.experiments_helper import max_vertex


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

    def projection(self, x, accuracy):
        pass
