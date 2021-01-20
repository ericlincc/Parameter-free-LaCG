# codeing=utf-8
"""This module contains main functions used for executing experiements."""


import numpy as np
from gurobipy import GRB
import networkx as nx

from pflacg.algorithms._algorithms_utils import Point

# TODO: add proper docstrings.


def fake_callback(model, where, value):
    """Callback function used to interface with Gurobi."""
    ggEps = 1e-08
    if where == GRB.Callback.MIP:
        objBnd = model.cbGet(GRB.Callback.MIP_OBJBND)
        if objBnd >= value + ggEps:
            pass


def max_vertex(grad, point):
    """Iterate over current active set and return vertex with greatest inner product."""
    max_prod = grad.dot(point.support[0])
    max_ind = 0
    for i in range(1, len(point.support)):
        if grad.dot(point.support[i]) > max_prod:
            max_prod = grad.dot(point.support[i])
            max_ind = i
    barycentric = np.zeros(len(point.support))
    barycentric[max_ind] = 1.0
    return Point(point.support[max_ind], barycentric, point.support), max_ind


def max_vertex_old(grad, active_vertex):
    """Iterate over current active set and return vertex with greatest inner product."""
    max_prod = np.dot(active_vertex[0], grad)
    max_ind = 0
    for i in range(len(active_vertex)):
        if np.dot(active_vertex[i], grad) > max_prod:
            max_prod = np.dot(active_vertex[i], grad)
            max_ind = i
    return active_vertex[max_ind], max_ind


def generate_random_graph(n, p):
    """
    Randamly generate graph.

    Paremeters
    ----------
    p: float
        The redirection probability

    Returns
    -------
    DiGraph
    """

    DG = nx.gnr_graph(n, p)
    return DG


def generate_structured_graph(layers, nodes_per_layer):
    """
    Two vertices in the end.

    Parameters
    ----------
    layers: int
        Number of layers
    nodes_per_layer: int
        Number of nodes per layer

    Returns
    -------
    DiGraph
    """

    m = layers
    s = nodes_per_layer
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


def construct_dictionary_indices(graph):
    """
    Given a DAG, returns a mapping from the edges to indices from 0 to N where N represents the
    number of Edges.
    """

    # Construct a dictionary of the indices
    dictionary = {}
    itCount = 0
    for i in graph.edges:
        dictionary[i] = itCount
        itCount += 1
    return dictionary
