# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 17:20:13 2021

@author: pccom
"""

from pflacg.experiments.feasible_regions import *
from pflacg.experiments.objective_functions import *
from pflacg.algorithms.fw_variants_simplex import *
from pflacg.algorithms._algorithms_utils import ExitCriterion

dimension = 100

mat = np.random.rand(dimension, dimension)
matrix = mat.dot(mat.T) + np.identity(dimension)
b = np.random.rand(dimension)

fun = Quadratic(dimension, matrix, b)
feasible_region = ProbabilitySimplexPolytope(dimension)
exit_criterion = ExitCriterion("SWG", 0.001, max_time= 60)

CGS_alg = ConditionalGradientSliding()
cgs_results = CGS_alg.run(fun, feasible_region, exit_criterion)

AFW_alg = FrankWolfeSimplex("AFW", "line_search")
afw_results = AFW_alg.run(fun, feasible_region, exit_criterion)

PFW_alg = FrankWolfeSimplex("PFW", "line_search")
pfw_results = PFW_alg.run(fun, feasible_region, exit_criterion)

FW_alg = FrankWolfeSimplex("FW", "line_search")
fw_results = FW_alg.run(fun, feasible_region, exit_criterion)

DIPFW_alg = FrankWolfeSimplex("DIPFW", "line_search")
dipfw_results = DIPFW_alg.run(fun, feasible_region, exit_criterion)

colors = ["k", "c", "b", "m", "r", "g"]
markers = ["o", "s", "^", "P", "D", "p"]
size_marker = 12
fontsize = 19
fontsize_legend = 20
linewidth_figures = 1.50

import matplotlib.pyplot as plt

plt.subplot(1, 2, 1)
AFW_run_iter_w = [(run_status[1], run_status[4]) for run_status in afw_results]
PFW_run_iter_w = [(run_status[1], run_status[4]) for run_status in pfw_results]
FW_run_iter_w = [(run_status[1], run_status[4]) for run_status in fw_results]
DIFW_run_iter_w = [(run_status[1], run_status[4]) for run_status in dipfw_results]
CGS_run_iter_w = [(run_status[1], run_status[4]) for run_status in cgs_results]
plt.semilogy(
    [s[0] for s in AFW_run_iter_w],
    [s[1] for s in AFW_run_iter_w],
    colors[0],
    # marker=markers[1],
    # markersize=size_marker,
    # markevery = np.linspace(0, len([s[0] for s in AFW_run_iter_w]) - 2, 10, dtype = int).tolist(),
    linewidth=linewidth_figures,
    label = "AFW"
)
plt.semilogy(
    [s[0] for s in AFW_run_iter_w],
    [s[1] for s in AFW_run_iter_w],
    colors[1],
    # marker=markers[1],
    # markersize=size_marker,
    # markevery = np.linspace(0, len([s[0] for s in AFW_run_iter_w]) - 2, 10, dtype = int).tolist(),
    linewidth=linewidth_figures,
    label = "PFW"
)
plt.semilogy(
    [s[0] for s in AFW_run_iter_w],
    [s[1] for s in AFW_run_iter_w],
    colors[2],
    # marker=markers[1],
    # markersize=size_marker,
    # markevery = np.linspace(0, len([s[0] for s in AFW_run_iter_w]) - 2, 10, dtype = int).tolist(),
    linewidth=linewidth_figures,
    label = "FW"
)
plt.semilogy(
    [s[0] for s in AFW_run_iter_w],
    [s[1] for s in AFW_run_iter_w],
    colors[3],
    # marker=markers[1],
    # markersize=size_marker,
    # markevery = np.linspace(0, len([s[0] for s in AFW_run_iter_w]) - 2, 10, dtype = int).tolist(),
    linewidth=linewidth_figures,
    label = "DIPFW"
)
plt.semilogy(
    [s[0] for s in CGS_run_iter_w],
    [s[1] for s in CGS_run_iter_w],
    colors[4],
    # marker=markers[1],
    # markersize=size_marker,
    # markevery = np.linspace(0, len([s[0] for s in AFW_run_iter_w]) - 2, 10, dtype = int).tolist(),
    linewidth=linewidth_figures,
    label = "CGS"
)
plt.xlabel("Time")
plt.ylabel("Strong Wolfe Gap")
plt.legend()

plt.subplot(1, 2, 2)
AFW_run_iter_w = [(run_status[0], run_status[4]) for run_status in afw_results]
PFW_run_iter_w = [(run_status[0], run_status[4]) for run_status in pfw_results]
FW_run_iter_w = [(run_status[0], run_status[4]) for run_status in fw_results]
DIFW_run_iter_w = [(run_status[0], run_status[4]) for run_status in dipfw_results]
CGS_run_iter_w = [(run_status[0], run_status[4]) for run_status in cgs_results]
plt.semilogy(
    [s[0] for s in AFW_run_iter_w],
    [s[1] for s in AFW_run_iter_w],
    colors[0],
    # marker=markers[1],
    # markersize=size_marker,
    # markevery = np.linspace(0, len([s[0] for s in AFW_run_iter_w]) - 2, 10, dtype = int).tolist(),
    linewidth=linewidth_figures,
    label = "AFW"
)
plt.semilogy(
    [s[0] for s in AFW_run_iter_w],
    [s[1] for s in AFW_run_iter_w],
    colors[1],
    # marker=markers[1],
    # markersize=size_marker,
    # markevery = np.linspace(0, len([s[0] for s in AFW_run_iter_w]) - 2, 10, dtype = int).tolist(),
    linewidth=linewidth_figures,
    label = "PFW"
)
plt.semilogy(
    [s[0] for s in AFW_run_iter_w],
    [s[1] for s in AFW_run_iter_w],
    colors[2],
    # marker=markers[1],
    # markersize=size_marker,
    # markevery = np.linspace(0, len([s[0] for s in AFW_run_iter_w]) - 2, 10, dtype = int).tolist(),
    linewidth=linewidth_figures,
    label = "FW"
)
plt.semilogy(
    [s[0] for s in AFW_run_iter_w],
    [s[1] for s in AFW_run_iter_w],
    colors[3],
    # marker=markers[1],
    # markersize=size_marker,
    # markevery = np.linspace(0, len([s[0] for s in AFW_run_iter_w]) - 2, 10, dtype = int).tolist(),
    linewidth=linewidth_figures,
    label = "DIPFW"
)
plt.semilogy(
    [s[0] for s in CGS_run_iter_w],
    [s[1] for s in CGS_run_iter_w],
    colors[4],
    # marker=markers[1],
    # markersize=size_marker,
    # markevery = np.linspace(0, len([s[0] for s in AFW_run_iter_w]) - 2, 10, dtype = int).tolist(),
    linewidth=linewidth_figures,
    label = "CGS"
)
plt.xlabel("Iteration")
plt.ylabel("Strong Wolfe Gap")
plt.legend()
plt.savefig("L1_ball_comparison8.pdf")
# plt.show()
plt.close()