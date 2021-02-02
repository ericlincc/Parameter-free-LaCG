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
    list_xs,
    list_ys,
    legends,
    colors,
    markers,
    list_x_label,
    y_label,
    save_path,
    title=None,
    list_legend_location=None,
    put_legend_outside=False,  # TODO: If true, may have unexpected behaviours.
    log_x=False,
    log_y=True,
    list_x_limits=None,
    y_limits=None,
    dpi=None,
    figsize=(16, 9),
    title_font_size=19,
    label_font_size=19,
    axis_font_size=12,
    marker_size=12,
    legend_font_size=20,
    linewidth_figures=4.0,
    **kwargs,
):
    # TODO: This function contains known bugs.

    if len(list_xs) > 9:
        raise Exception("plot_pretty() is not designed to handle more than 9 subplots")

    plt.rcParams.update({"font.size": axis_font_size})
    plt.rcParams["figure.figsize"] = figsize
    is_leftmost = True
    ax = None

    for j, (xs, ys) in enumerate(zip(list_xs, list_ys)):

        ax = plt.subplot(int(f"{1}{len(list_xs)+1}{j+1}"), sharey=ax)

        for i in range(len(ys)):
            if legends:
                if log_x and log_y:
                    plt.loglog(
                        xs[i],
                        ys[i],
                        colors[i],
                        marker=markers[i],
                        markersize=marker_size,
                        markevery=np.logspace(0, np.log10(len(ys[i]) - 2), 10)
                        .astype(int)
                        .tolist(),
                        linewidth=linewidth_figures,
                        label=legends[i],
                    )
                if log_x and not log_y:
                    plt.semilogx(
                        xs[i],
                        ys[i],
                        colors[i],
                        marker=markers[i],
                        markersize=marker_size,
                        linewidth=linewidth_figures,
                        label=legends[i],
                    )
                if not log_x and log_y:
                    plt.semilogy(
                        xs[i],
                        ys[i],
                        colors[i],
                        marker=markers[i],
                        markersize=marker_size,
                        markevery=np.linspace(
                            0, len(ys[i]) - 2, 10, dtype=int
                        ).tolist(),
                        linewidth=linewidth_figures,
                        label=legends[i],
                    )
                if not log_x and not log_y:
                    plt.plot(
                        xs[i],
                        ys[i],
                        colors[i],
                        marker=markers[i],
                        markersize=marker_size,
                        markevery=np.linspace(
                            0, len(ys[i]) - 2, 10, dtype=int
                        ).tolist(),
                        linewidth=2.0,
                        label=legends[i],
                    )
            else:
                if log_x and log_y:
                    plt.loglog(
                        xs[i],
                        ys[i],
                        colors[i],
                        marker=markers[i],
                        markersize=marker_size,
                        markevery=np.logspace(0, np.log10(len(ys[i]) - 2), 10)
                        .astype(int)
                        .tolist(),
                        linewidth=linewidth_figures,
                    )
                if log_x and not log_y:
                    plt.semilogx(
                        xs[i],
                        ys[i],
                        colors[i],
                        marker=markers[i],
                        markersize=marker_size,
                        linewidth=linewidth_figures,
                    )
                if not log_x and log_y:
                    plt.semilogy(
                        xs[i],
                        ys[i],
                        colors[i],
                        marker=markers[i],
                        markersize=marker_size,
                        markevery=np.linspace(
                            0, len(ys[i]) - 2, 10, dtype=int
                        ).tolist(),
                        linewidth=linewidth_figures,
                    )
                if not log_x and not log_y:
                    plt.plot(
                        xs[i],
                        ys[i],
                        colors[i],
                        marker=markers[i],
                        markersize=marker_size,
                        markevery=np.linspace(
                            0, len(ys[i]) - 2, 10, dtype=int
                        ).tolist(),
                        linewidth=linewidth_figures,
                    )

        plt.xlabel(list_x_label[j], fontsize=label_font_size)
        if is_leftmost:
            plt.ylabel(y_label, fontsize=label_font_size)
        else:
            plt.setp(ax.get_yticklabels(), visible=False)

        if legends:
            if list_legend_location[j] is not None:
                plt.legend(fontsize=legend_font_size, loc=list_legend_location[j])
            elif put_legend_outside:
                plt.legend(
                    fontsize=legend_font_size,
                    loc="center left",
                    bbox_to_anchor=(1, 0.5),
                )
            else:
                pass
        if list_x_limits[j] is not None:
            plt.xlim(list_x_limits[j])
        if y_limits is not None:
            plt.ylim(y_limits)

        plt.tight_layout()
        plt.grid(True, which="both")

        is_leftmost = False

    if title:
        plt.suptitle(title, fontsize=title_font_size)
    plt.savefig(save_path, dpi=dpi, format="png", bbox_inches="tight")
    plt.close()
