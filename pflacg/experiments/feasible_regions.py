# codeing=utf-8
"""This module contains feasible region classes for the experiements."""

from abc import ABC, abstractmethod
import logging
import math

import numpy as np
from scipy.sparse.linalg import eigsh

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


class ConvexHull(_AbstractFeasibleRegion):
    """Convex hull given a set of vertice."""

    def __init__(self, vertices):
        self.vertices = vertices

    def lp_oracle(self, d):
        val, index = d.dot(self.vertices[0]), 0
        for _index, vertex in enumerate(self.vertices):
            _val = d.dot(vertex)
            if _val < val:
                val, index = _val, _index
        return self.vertices[index]

    def away_oracle(self, d, point_x):
        return max_vertex(d, point_x.support)

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

    def lp_oracle(self, d):
        from scipy.optimize import linear_sum_assignment

        objective = d.reshape((self.mat_dim, self.mat_dim))
        matching = linear_sum_assignment(objective)
        solution = np.zeros((self.mat_dim, self.mat_dim))
        solution[matching] = 1
        return solution.reshape(self.dim)

    def away_oracle(self, grad, point_x):
        return max_vertex(grad, point_x.support)


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

    #Input is the vector over which we calculate the inner product.
    def away_oracle_fast(self, grad, x):
        aux = np.multiply(grad, np.sign(x))
        indices = np.where(x > 0.0)[0]
        v = np.zeros(len(x), dtype = float)
        indexMax = indices[np.argmax(aux[indices])]
        v[indexMax] = 1.0
        return v, indexMax

    def away_oracle(self, grad, point_x):
        return max_vertex(grad, point_x.support)

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

    def away_oracle(self, grad, point_x):
        return max_vertex(grad, point_x.support)

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

    def away_oracle(self, grad, point_x):
        return max_vertex(grad, point_x.support)

    def projection(self, x):
        return x / np.linalg.norm(x)


class spectrahedron(_AbstractFeasibleRegion):
    def __init__(self, dim):
        self.dim = dim
        self.matdim = int(np.sqrt(dim))

    def lp_oracle(self, X):
        objective = X.reshape((self.matdim, self.matdim))
        w, v = eigsh(-objective, 1, which="LA", maxiter=100000)
        return (np.outer(v, v)).reshape(self.dim)

    @property
    def initial_point(self):
        return (np.identity(self.matdim) / self.matdim).flatten()

    @property
    def initial_active_set(self):
        return [self.initial_point()]

    def away_oracle(self, grad, point_x):
        return max_vertex(grad, point_x.support)


import networkx as nx

# Generate a valid DAG such that we can solve the shortest path problem.
def generateRandomGraph(n, p):
    DG = nx.gnr_graph(n, p)
    return DG


# Graph with a source and a sink, and a number of layers specified by layers
# and a number of nodes per layer equal to nodesPerLayer.
def generateStructuredGraph(layers, nodesPerLayer):
    m = layers
    s = nodesPerLayer
    DG = nx.DiGraph()
    DG.add_nodes_from(range(0, m * s + 1))
    # Add first edges between source
    DG.add_edges_from([(0, x + 1) for x in range(s)])
    # Add all the edges in the subsequent layers.
    for i in range(m - 1):
        DG.add_edges_from(
            [(x + 1 + s * i, y + 1 + s * (i + 1)) for x in range(s) for y in range(s)]
        )
    DG.add_edges_from([(x + 1 + s * (m - 1), m * s + 1) for x in range(s)])
    return DG


"""
If typegraph = "Structured":
    param1 = number of layers
    param2 = number of nodes per layer.
    
Otherwise:
Growing network with redirection (GNR) digraph
    param1 = number of nodes
    param2 = The redirection probability.
"""


class flow_polytope(_AbstractFeasibleRegion):
    """Shortest path problem on a DAG."""

    def __init__(self, param1, param2, typeGraph="Structured"):
        # Generate the type of graph that we want
        if typeGraph == "Structured":
            self.graph = generateStructuredGraph(param1, param2)
        else:
            self.graph = generateRandomGraph(param1, param2)
        # Sort the graph in topological order
        self.topologicalSort = list(nx.topological_sort(self.graph))
        self.dictIndices = self.constructDictionaryIndices(self.graph)
        self.dim = self.graph.number_of_edges()
        return

    @property
    def initial_point(self):
        return self.lp_oracle(np.ones(self.dim))

    @property
    def initial_active_set(self):
        return [self.lp_oracle(np.ones(self.dim))]

    def lp_oracle(self, weight):
        d = math.inf * np.ones(nx.number_of_nodes(self.graph))
        d[self.topologicalSort[0]] = 0.0
        p = -np.ones(nx.number_of_nodes(self.graph), dtype=int)
        for u in self.topologicalSort:
            for v in self.graph.neighbors(u):
                self.relax(u, v, d, weight, p)

        pathAlg = [self.topologicalSort[-1]]
        while pathAlg[-1] != self.topologicalSort[0]:
            pathAlg.append(p[pathAlg[-1]])
        pathAlg.reverse()
        # Reconstruc the vertex.
        outputVect = np.zeros(nx.number_of_edges(self.graph))
        for i in range(len(pathAlg) - 1):
            outputVect[self.dictIndices[(pathAlg[i], pathAlg[i + 1])]] = 1.0
        return outputVect

    def relax(self, i, j, dVect, wVect, pVect):
        if dVect[j] > dVect[i] + wVect[self.dictIndices[(i, j)]]:
            dVect[j] = dVect[i] + wVect[self.dictIndices[(i, j)]]
            pVect[j] = i
        return

    # Function that returns the values of the weights.
    def func(self, u, v, wVect):
        return self.weight[self.dictIndices[(v, u)]]

    # Given a DAG, returns a mapping from the edges to indices from 0 to N
    # where N represents the number of Edges.
    def constructDictionaryIndices(self, graph):
        # Construct a dictionary of the indices
        dictionary = {}
        itCount = 0
        for i in graph.edges:
            dictionary[i] = itCount
            itCount += 1
        return dictionary

        def dim(self):
            return self.dimension

    def plot(self):
        import matplotlib.pyplot as plt

        nx.draw(self.graph)
        plt.show()

    def returnEdges(self):
        return self.graph.edges()

    def topologicalOrdering(self):
        return self.topologicalSort

    def away_oracle(self, grad, point_x):
        return max_vertex(grad, point_x.support)

    def dimension(self):
        return self.dim
