# codeing=utf-8
"""This module contains main functions used for executing experiements."""


import numpy as np
from matplotlib import rc
import matplotlib.pyplot as plt
import networkx as nx
from scipy.interpolate import griddata

from pflacg.algorithms._algorithms_utils import Point


def max_vertex(d, vertices):
    """
    Iterate over current active set and return vertex with greatest inner product.

    Parameters
    ----------
    d: np.ndarray
        Direction.
    vertices: tuple(np.ndarray) or list(np.ndarray)
        Tuple or list of vertices.

    Returns
    -------
    Point
    """

    max_prod = d.dot(vertices[0])
    max_ind = 0
    for i in range(1, len(vertices)):
        if d.dot(vertices[i]) > max_prod:
            max_prod = d.dot(vertices[i])
            max_ind = i
    barycentric = np.zeros(len(vertices))
    barycentric[max_ind] = 1.0
    return Point(vertices[max_ind], barycentric, vertices), max_ind


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
    iter_count = 0
    for i in graph.edges:
        dictionary[i] = iter_count
        iter_count += 1
    return dictionary


def plot_pretty(
    list_x,
    list_y,
    list_legend,
    colors,
    markers,
    x_label,
    y_label,
    save_path,
    title=None,
    log_x=False,
    log_y=True,
    legend_location=None,
    put_legend_outside=False,
    x_limits=None,
    y_limits=None,
    dpi=None,
    title_font_size=19,
    label_font_size=19,
    axis_font_size=12,
    marker_size=12,
    legend_font_size=20,
    linewidth_figures=4.0,
    **kwargs,
):

    plt.rcParams.update({"font.size": axis_font_size})

    for i in range(len(list_y)):
        if list_legend:
            if log_x and log_y:
                plt.loglog(
                    list_x[i],
                    list_y[i],
                    colors[i],
                    marker=markers[i],
                    markersize=marker_size,
                    markevery=np.logspace(0, np.log10(len(list_y[i]) - 2), 10)
                    .astype(int)
                    .tolist(),
                    linewidth=linewidth_figures,
                    label=list_legend[i],
                )
            if log_x and not log_y:
                plt.semilogx(
                    list_x[i],
                    list_y[i],
                    colors[i],
                    marker=markers[i],
                    markersize=marker_size,
                    linewidth=linewidth_figures,
                    label=list_legend[i],
                )
            if not log_x and log_y:
                plt.semilogy(
                    list_x[i],
                    list_y[i],
                    colors[i],
                    marker=markers[i],
                    markersize=marker_size,
                    markevery=np.linspace(
                        0, len(list_y[i]) - 2, 10, dtype=int
                    ).tolist(),
                    linewidth=linewidth_figures,
                    label=list_legend[i],
                )
            if not log_x and not log_y:
                plt.plot(
                    list_x[i],
                    list_y[i],
                    colors[i],
                    marker=markers[i],
                    markersize=marker_size,
                    markevery=np.linspace(
                        0, len(list_y[i]) - 2, 10, dtype=int
                    ).tolist(),
                    linewidth=2.0,
                    label=list_legend[i],
                )
        else:
            if log_x and log_y:
                plt.loglog(
                    list_x[i],
                    list_y[i],
                    colors[i],
                    marker=markers[i],
                    markersize=marker_size,
                    markevery=np.logspace(0, np.log10(len(list_y[i]) - 2), 10)
                    .astype(int)
                    .tolist(),
                    linewidth=linewidth_figures,
                )
            if log_x and not log_y:
                plt.semilogx(
                    list_x[i],
                    list_y[i],
                    colors[i],
                    marker=markers[i],
                    markersize=marker_size,
                    linewidth=linewidth_figures,
                )
            if not log_x and log_y:
                plt.semilogy(
                    list_x[i],
                    list_y[i],
                    colors[i],
                    marker=markers[i],
                    markersize=marker_size,
                    markevery=np.linspace(
                        0, len(list_y[i]) - 2, 10, dtype=int
                    ).tolist(),
                    linewidth=linewidth_figures,
                )
            if not log_x and not log_y:
                plt.plot(
                    list_x[i],
                    list_y[i],
                    colors[i],
                    marker=markers[i],
                    markersize=marker_size,
                    markevery=np.linspace(
                        0, len(list_y[i]) - 2, 10, dtype=int
                    ).tolist(),
                    linewidth=linewidth_figures,
                )
    if title:
        plt.title(title, fontsize=title_font_size)
    plt.ylabel(y_label, fontsize=label_font_size)
    plt.xlabel(x_label, fontsize=label_font_size)
    if x_limits is not None:
        plt.xlim(x_limits)
    if y_limits is not None:
        plt.ylim(y_limits)
    if list_legend:
        if legend_location is not None:
            plt.legend(fontsize=legend_font_size, loc=legend_location)
        elif put_legend_outside:
            plt.legend(
                fontsize=legend_font_size, loc="center left", bbox_to_anchor=(1, 0.5)
            )
        else:
            plt.legend(fontsize=legend_font_size)
    plt.tight_layout()
    plt.grid(True, which="both")
    plt.savefig(save_path, dpi=dpi, format="png", bbox_inches="tight")
    plt.close()
