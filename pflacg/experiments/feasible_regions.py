# codeing=utf-8
"""This module contains feasible region classes for the experiements."""

from abc import ABC, abstractmethod
import logging
import math

import numpy as np
from scipy.sparse.linalg import eigsh

from pflacg.experiments.experiments_helper import max_vertex
from scipy.optimize import linprog
from cvxopt import matrix, sparse, solvers


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


class ConvexHull(_AbstractFeasibleRegion):
    """Convex hull given a set of vertice."""

    def __init__(self, vertices):
        self.vertices = vertices

    @property
    def initial_point(self):
        return self.vertices[0]

    @property
    def initial_active_set(self):
        return [self.vertices[0]]

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

    #     #This is a faster implementation of the away oracle without having to loop through active set.
    #    def away_oracle(self, grad, x):
    #        aux = np.multiply(grad, np.sign(x))
    #        indices = np.where(x > 0.0)[0]
    #        v = np.zeros(len(x), dtype=float)
    #        index_max = indices[np.argmax(aux[indices])]
    #        v[index_max] = 1.0
    #        return v, index_max

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


class ConstrainedL1BallPolytope(_AbstractFeasibleRegion):
    def __init__(
        self,
        l1_regularization,
        dim,
        const_matrix_ineq=None,
        const_vector_ineq=None,
        const_matrix_eq=None,
        const_vector_eq=None,
        solver_type="cvxopt",
        scipy_solver="revised simplex",
        sparse_solver=False,
    ):
        self.dim = dim
        self.l1_regularization = l1_regularization
        self.solver_type = solver_type
        assert solver_type == "cvxopt" or solver_type == "scipy", "Wrong solver type"
        if solver_type == "cvxopt":
            solvers.options["show_progress"] = False
        else:
            self.scipy_solver = scipy_solver
        if sparse_solver:
            assert (
                solver_type == "cvxopt"
            ), "scipy solver cannot handle sparse matrices."
        simplex_dimensionality = int(2 * dim)
        if const_matrix_ineq is not None and const_vector_ineq is not None:
            num_ineq_constraints, dim_ineq_constraints = const_matrix_ineq.shape
            assert (
                dim_ineq_constraints == self.dim
            ), "Dimension of the inequality constraints does not match the dimensionality of the problem."
            self.G = np.vstack(
                (
                    np.hstack((const_matrix_ineq, -const_matrix_ineq)),
                    -np.identity(simplex_dimensionality),
                )
            )
            self.h = np.append(const_vector_ineq, np.zeros(simplex_dimensionality))
            if solver_type == "cvxopt":
                self.G = matrix(
                    self.G,
                    (
                        simplex_dimensionality + num_ineq_constraints,
                        simplex_dimensionality,
                    ),
                )
                if sparse_solver:
                    self.G = sparse(self.G)
                self.h = matrix(
                    self.h, (simplex_dimensionality + num_ineq_constraints, 1)
                )
        else:
            self.G = -np.identity(simplex_dimensionality)
            self.h = np.zeros(simplex_dimensionality)
            if solver_type == "cvxopt":
                self.G = matrix(
                    self.G,
                )
                self.h = matrix(self.h, (simplex_dimensionality, 1))
                if sparse_solver:
                    self.G = sparse(self.G)

        if const_matrix_eq is not None and const_vector_eq is not None:
            num_eq_constraints, dim_eq_constraints = const_matrix_eq.shape
            assert (
                dim_eq_constraints == self.dim
            ), "Dimension of the equality constraints does not match the dimensionality of the problem."
            self.A = np.vstack(
                (
                    np.hstack((const_matrix_eq, -const_matrix_eq)),
                    np.ones(simplex_dimensionality),
                )
            )
            self.b = np.append(const_vector_eq, self.l1_regularization).tolist()
            if solver_type == "cvxopt":
                self.A = matrix(
                    self.A, (1 + num_eq_constraints, simplex_dimensionality)
                )
                self.b = matrix(self.b, (1 + len(const_vector_eq), 1), "d")
                if sparse_solver:
                    self.A = sparse(self.A)
        else:
            self.A = np.ones(simplex_dimensionality)
            self.b = self.l1_regularization
            if solver_type == "cvxopt":
                self.A = matrix(self.A, (1, simplex_dimensionality))
                self.b = matrix(self.b)
            else:
                self.A = np.ones(simplex_dimensionality).reshape(
                    (simplex_dimensionality, 1)
                )
                self.b = np.asarray(self.b).reshape((1,))

    @property
    def initial_point(self):
        c = np.ones(self.dim)
        return self.lp_oracle(c)

    @property
    def initial_active_set(self):
        return [self.initial_point()]

    def lp_oracle(self, x):
        cost_vector = np.hstack((x, -x))
        if self.solver_type == "cvxopt":
            sol = solvers.lp(
                matrix(cost_vector),
                self.G,
                self.h,
                self.A,
                self.b,
                solver="cvxopt_glpk",
            )
            assert sol["status"] == "optimal", "Algorithm did not converge."
            optimum = np.array(sol["x"])
            return (
                optimum[: int(len(optimum) / 2)] - optimum[int(len(optimum) / 2) :]
            ).flatten()
        else:
            res = linprog(
                cost_vector,
                A_ub=self.G,
                b_ub=self.h,
                A_eq=self.A.T,
                b_eq=self.b,
                method=self.scipy_solver,
                bounds=(-np.inf, np.inf),
            )
            assert res.status == 0, "LP oracle did not return succesfully."
            optimum = np.array(res.x)
            return (
                optimum[: int(len(optimum) / 2)] - optimum[int(len(optimum) / 2) :]
            ).flatten()

    def away_oracle(self, grad, point_x):
        return max_vertex(grad, point_x.support)


class GeneralPolytope(_AbstractFeasibleRegion):
    def __init__(
        self,
        dim,
        const_matrix_ineq=None,
        const_vector_ineq=None,
        const_matrix_eq=None,
        const_vector_eq=None,
        solver_type="cvxopt",
        scipy_solver="revised simplex",
        sparse_solver=False,
    ):
        self.dim = dim
        self.solver_type = solver_type
        assert solver_type == "cvxopt" or solver_type == "scipy", "Wrong solver type"
        if solver_type == "cvxopt":
            solvers.options["show_progress"] = False
        else:
            self.scipy_solver = scipy_solver
        if sparse_solver:
            assert (
                solver_type == "cvxopt"
            ), "scipy solver cannot handle sparse matrices."
        if const_matrix_ineq is not None and const_vector_ineq is not None:
            num_ineq_constraints, dim_ineq_constraints = const_matrix_ineq.shape
            assert (
                dim_ineq_constraints == self.dim
            ), "Dimension of the inequality constraints does not match the dimensionality of the problem."
            self.G = const_matrix_ineq
            self.h = const_vector_ineq
            if solver_type == "cvxopt":
                self.G = matrix(self.G, (num_ineq_constraints, dim_ineq_constraints))
                if sparse_solver:
                    self.G = sparse(self.G)
                self.h = matrix(self.h, (num_ineq_constraints, 1))
        else:
            self.G = None
            self.h = None

        if const_matrix_eq is not None and const_vector_eq is not None:
            num_eq_constraints, dim_eq_constraints = const_matrix_eq.shape
            assert (
                dim_eq_constraints == self.dim
            ), "Dimension of the equality constraints does not match the dimensionality of the problem."
            self.A = const_matrix_eq
            self.b = const_vector_eq
            if solver_type == "cvxopt":
                self.A = matrix(self.A, (num_eq_constraints, dim_eq_constraints))
                self.b = matrix(self.b, (num_eq_constraints, 1), "d")
                if sparse_solver:
                    self.A = sparse(self.A)
        else:
            self.A = None
            self.b = None

    @property
    def initial_point(self):
        c = np.ones(self.dim)
        return self.lp_oracle(c)

    @property
    def initial_active_set(self):
        return [self.initial_point()]

    def lp_oracle(self, x):
        if self.solver_type == "cvxopt":
            if self.G is not None and self.h is not None:
                if self.A is not None and self.b is not None:
                    sol = solvers.lp(
                        matrix(x),
                        G=self.G,
                        h=self.h,
                        A=self.A,
                        b=self.b,
                        solver="cvxopt_glpk",
                    )
                else:
                    sol = solvers.lp(
                        matrix(x), G=self.G, h=self.h, solver="cvxopt_glpk"
                    )
            else:
                if self.A is not None and self.b is not None:
                    sol = solvers.lp(
                        matrix(x), A=self.A, b=self.b, solver="cvxopt_glpk"
                    )
                else:
                    assert False, "The problem has no constraintsts"
            assert sol["status"] == "optimal", "Algorithm did not converge."
            return np.array(sol["x"]).flatten()
        else:
            if self.G is not None and self.h is not None:
                if self.A is not None and self.b is not None:
                    res = linprog(
                        x,
                        A_ub=self.G,
                        b_ub=self.h,
                        A_eq=self.A.T,
                        b_eq=self.b,
                        method=self.scipy_solver,
                        bounds=(-np.inf, np.inf),
                    )
                else:
                    res = linprog(
                        x,
                        A_ub=self.G,
                        b_ub=self.h,
                        method=self.scipy_solver,
                        bounds=(-np.inf, np.inf),
                    )
            else:
                if self.A is not None and self.b is not None:
                    res = linprog(
                        x,
                        A_eq=self.A.T,
                        b_eq=self.b,
                        method=self.scipy_solver,
                        bounds=(-np.inf, np.inf),
                    )
                else:
                    assert False, "The problem has no constraints"
            assert res.status == 0, "LP oracle did not return succesfully."
            return np.array(res.x).flatten()

    def away_oracle(self, grad, point_x):
        return max_vertex(grad, point_x.support)


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


class Spectrahedron(_AbstractFeasibleRegion):
    """TODO: Add description."""

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


# TODO: Some of the following functions should go into experiment_helper.py

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


class FlowPolytope(_AbstractFeasibleRegion):
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
