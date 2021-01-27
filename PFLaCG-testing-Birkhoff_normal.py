#!/usr/bin/env python
# coding: utf-8

# In[1]:


from multiprocessing import shared_memory, Value, Process, Lock

import matplotlib.pyplot as plt

import os
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
os.environ["OMP_NUM_THREADS"] = "1" 

import numpy as np

from pflacg.algorithms._algorithms_utils import *
from pflacg.algorithms.pflacg import ParameterFreeLaCG
from pflacg.algorithms.fw_variants import FrankWolfe, FrankWolfe_sparse
from pflacg.experiments.objective_functions import *
from pflacg.experiments.feasible_regions import *
from scipy.sparse import csc_matrix

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s :: %(asctime)s :: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# In[5]:


ITER_SYNC = False
RANDOM_SEED = 1


# In[3]:


np.random.seed(RANDOM_SEED)

test_exit_criterion = ExitCriterion(
    "SWG",
    1.0e-5,
    criterion_reference=0.0,
    max_time=600.0,
    max_iter=10000
)
optimal_exit_criterion = ExitCriterion(
    "DG",
    1.0e-5,
    criterion_reference=0.0,
    max_time=600.0,
    max_iter=10000
)
dimension = 100


# matrix = np.random.rand(dimension, dimension)
# M = matrix.T.dot(matrix)
# M = M + 500 * np.identity(dimension)
# b = np.random.rand(dimension)
# objective_function = Quadratic(dimension, M, b)

objective_function = Quadratic_Diagonal(dimension, np.random.rand(dimension), Mu = 1.0, L = 10000.0)

feasible_region = BirkhoffPolytope(dimension)
feasible_region_sparse = BirkhoffPolytope_sparse(dimension)

print(objective_function.Mu)
print(objective_function.L)

AFW_algorithm_sparse = FrankWolfe_sparse("AFW", "line_search")
AFW_run_sparse = AFW_algorithm_sparse.run(objective_function, feasible_region_sparse, test_exit_criterion)

AFW_algorithm = FrankWolfe("AFW", "line_search")
AFW_run = AFW_algorithm.run(objective_function, feasible_region, test_exit_criterion)

DIPFW_algorithm = FrankWolfe("DIPFW", "line_search")
# optimal_AFW_run = DIPFW_algorithm.run(objective_function, feasible_region, optimal_exit_criterion)

# approx_f_opt = optimal_AFW_run[-1][2]

# # if ITER_SYNC:
#     x_index, x_axis = 0, "Iter"
# else:
#     x_index, x_axis = 1, "Time"
# colors = ["k", "c", "b", "m", "r", "g"]
# markers = ["o", "s", "^", "P", "D", "p"]
# size_marker = 12
# fontsize = 19
# fontsize_legend = 20
# linewidth_figures = 4.0


# plt.subplot(1, 2, 1)
# AFW_run_iter_w = [(run_status[x_index], run_status[4]) for run_status in AFW_run]
# AFW_run_iter_w_sparse = [(run_status[x_index], run_status[4]) for run_status in AFW_run_sparse]

# plt.semilogy(
#     [s[0] for s in AFW_run_iter_w_sparse],
#     [s[1] for s in AFW_run_iter_w_sparse],
#     colors[0],
#     marker=markers[0],
#     markersize=size_marker,
#     markevery = np.linspace(0, len([s[0] for s in AFW_run_iter_w_sparse]) - 2, 10, dtype = int).tolist(),
#     linewidth=linewidth_figures,
#     label = "AFW_sparse"
# )
# plt.semilogy(
#     [s[0] for s in AFW_run_iter_w],
#     [s[1] for s in AFW_run_iter_w],
#     colors[1],
#     marker=markers[1],
#     markersize=size_marker,
#     markevery = np.linspace(0, len([s[0] for s in AFW_run_iter_w]) - 2, 10, dtype = int).tolist(),
#     linewidth=linewidth_figures,
#     label = "AFW"
# )
# plt.title(
#     f"{type(objective_function).__name__}\n"
#     f"{type(feasible_region).__name__}\n"
#     f"dim = {dimension}\n"
#     f"L/m = {int(objective_function.L / objective_function.Mu)}"
# )
# plt.xlabel(x_axis)
# plt.ylabel("Strong Wolfe Gap")
# plt.legend()

# plt.subplot(1, 2, 2)
# AFW_run_iter_w = [(run_status[x_index], run_status[2] - approx_f_opt) for run_status in AFW_run]
# AFW_run_iter_w_sparse = [(run_status[x_index], run_status[2] - approx_f_opt) for run_status in AFW_run_sparse]
# plt.semilogy(
#     [s[0] for s in AFW_run_iter_w_sparse],
#     [s[1] for s in AFW_run_iter_w_sparse],
#     colors[0],
#     marker=markers[0],
#     markersize=size_marker,
#     markevery = np.linspace(0, len([s[0] for s in AFW_run_iter_w_sparse]) - 2, 10, dtype = int).tolist(),
#     linewidth=linewidth_figures,
#     label = "AFW_sparse"
# )
# plt.semilogy(
#     [s[0] for s in AFW_run_iter_w],
#     [s[1] for s in AFW_run_iter_w],
#     colors[1],
#     marker=markers[1],
#     markersize=size_marker,
#     markevery = np.linspace(0, len([s[0] for s in AFW_run_iter_w]) - 2, 10, dtype = int).tolist(),
#     linewidth=linewidth_figures,
#     label = "AFW"
# )
# plt.title(
#     f"{type(objective_function).__name__}\n"
#     f"{type(feasible_region).__name__}\n"
#     f"dim = {dimension}\n"
#     f"L/m = {int(objective_function.L / objective_function.Mu)}"
# )
# plt.xlabel(x_axis)
# plt.ylabel("$f - f*$")
# plt.legend()
# plt.savefig("Sparsity_comparison_Time_Birkhoff_1024.pdf")
# # plt.show()
# plt.close()


# plt.subplot(1, 2, 1)
# AFW_run_iter_w = [(run_status[0], run_status[4]) for run_status in AFW_run]
# AFW_run_iter_w_sparse = [(run_status[0], run_status[4]) for run_status in AFW_run_sparse]
# plt.semilogy(
#     [s[0] for s in AFW_run_iter_w_sparse],
#     [s[1] for s in AFW_run_iter_w_sparse],
#     colors[0],
#     marker=markers[0],
#     markersize=size_marker,
#     markevery = np.linspace(0, len([s[0] for s in AFW_run_iter_w_sparse]) - 2, 10, dtype = int).tolist(),
#     linewidth=linewidth_figures,
#     label = "AFW_sparse"
# )
# plt.semilogy(
#     [s[0] for s in AFW_run_iter_w],
#     [s[1] for s in AFW_run_iter_w],
#     colors[1],
#     marker=markers[1],
#     markersize=size_marker,
#     markevery = np.linspace(0, len([s[0] for s in AFW_run_iter_w]) - 2, 10, dtype = int).tolist(),
#     linewidth=linewidth_figures,
#     label = "AFW"
# )
# plt.title(
#     f"{type(objective_function).__name__}\n"
#     f"{type(feasible_region).__name__}\n"
#     f"dim = {dimension}\n"
#     f"L/m = {int(objective_function.L / objective_function.Mu)}"
# )
# plt.xlabel("Iteration")
# plt.ylabel("Strong Wolfe Gap")
# plt.legend()

# plt.subplot(1, 2, 2)
# AFW_run_iter_w = [(run_status[0], run_status[2] - approx_f_opt) for run_status in AFW_run]
# AFW_run_iter_w_sparse = [(run_status[0], run_status[2] - approx_f_opt) for run_status in AFW_run_sparse]
# plt.semilogy(
#     [s[0] for s in AFW_run_iter_w_sparse],
#     [s[1] for s in AFW_run_iter_w_sparse],
#     colors[0],
#     marker=markers[0],
#     markersize=size_marker,
#     markevery = np.linspace(0, len([s[0] for s in AFW_run_iter_w_sparse]) - 2, 10, dtype = int).tolist(),
#     linewidth=linewidth_figures,
#     label = "AFW_sparse"
# )
# plt.semilogy(
#     [s[0] for s in AFW_run_iter_w],
#     [s[1] for s in AFW_run_iter_w],
#     colors[1],
#     marker=markers[1],
#     markersize=size_marker,
#     markevery = np.linspace(0, len([s[0] for s in AFW_run_iter_w]) - 2, 10, dtype = int).tolist(),
#     linewidth=linewidth_figures,
#     label = "AFW"
# )
# plt.title(
#     f"{type(objective_function).__name__}\n"
#     f"{type(feasible_region).__name__}\n"
#     f"dim = {dimension}\n"
#     f"L/m = {int(objective_function.L / objective_function.Mu)}"
# )
# plt.xlabel("Iteration")
# plt.ylabel("$f - f*$")
# plt.legend()
# plt.savefig("Sparsity_comparison_Iteration_Birkhoff_1024.pdf")
# # plt.show()
# plt.close()


PFW_algorithm = FrankWolfe("PFW", "line_search")
PFW_run = PFW_algorithm.run(objective_function, feasible_region, test_exit_criterion)

Lazy_algorithm = FrankWolfe("lazy", "line_search")
Lazy_run = Lazy_algorithm.run(objective_function, feasible_region, test_exit_criterion)

DIPFW_algorithm = FrankWolfe("DIPFW", "line_search")
DIPFW_run = DIPFW_algorithm.run(objective_function, feasible_region, test_exit_criterion)

PFLaCG = ParameterFreeLaCG(iter_sync=ITER_SYNC)
PFLaCG_run = PFLaCG.run(objective_function, feasible_region, test_exit_criterion)

optimal_AFW_run = DIPFW_algorithm.run(objective_function, feasible_region, optimal_exit_criterion)


approx_f_opt = optimal_AFW_run[-1][2]

if ITER_SYNC:
    x_index, x_axis = 0, "Iter"
else:
    x_index, x_axis = 1, "Time"


colors = ["k", "c", "b", "m", "r", "g"]
markers = ["o", "s", "^", "P", "D", "p"]
size_marker = 12
fontsize = 19
fontsize_legend = 20
linewidth_figures = 4.0


plt.subplot(1, 2, 1)
PFLaCG_run_iter_w = [(run_status[x_index], run_status[4]) for run_status in PFLaCG_run]
AFW_run_iter_w = [(run_status[x_index], run_status[4]) for run_status in AFW_run]
AFW_run_iter_w_sparse = [(run_status[x_index], run_status[4]) for run_status in AFW_run_sparse]
PFW_run_iter_w = [(run_status[x_index], run_status[4]) for run_status in PFW_run]
Lazy_run_iter_w = [(run_status[x_index], run_status[4]) for run_status in Lazy_run]
DICG_run_iter_w = [(run_status[x_index], run_status[4]) for run_status in DIPFW_run]

plt.semilogy(
    [s[0] for s in PFLaCG_run_iter_w],
    [s[1] for s in PFLaCG_run_iter_w],
    colors[0],
    marker=markers[0],
    markersize=size_marker,
    markevery = np.linspace(0, len([s[0] for s in PFLaCG_run_iter_w]) - 2, 10, dtype = int).tolist(),
    linewidth=linewidth_figures,
    label = "PFLaCG"
)
plt.semilogy(
    [s[0] for s in AFW_run_iter_w],
    [s[1] for s in AFW_run_iter_w],
    colors[1],
    marker=markers[1],
    markersize=size_marker,
    markevery = np.linspace(0, len([s[0] for s in AFW_run_iter_w]) - 2, 10, dtype = int).tolist(),
    linewidth=linewidth_figures,
    label = "AFW"
)
plt.semilogy(
    [s[0] for s in PFW_run_iter_w],
    [s[1] for s in PFW_run_iter_w],
    colors[2],
    marker=markers[2],
    markersize=size_marker,
    markevery = np.linspace(0, len([s[0] for s in PFW_run_iter_w]) - 2, 10, dtype = int).tolist(),
    linewidth=linewidth_figures,
    label = "PFW"
)
plt.semilogy(
    [s[0] for s in Lazy_run_iter_w],
    [s[1] for s in Lazy_run_iter_w],
    colors[3],
    marker=markers[3],
    markersize=size_marker,
    markevery = np.linspace(0, len([s[0] for s in Lazy_run_iter_w]) - 2, 10, dtype = int).tolist(),
    linewidth=linewidth_figures,
    label = "Lazy"
)
plt.semilogy(
    [s[0] for s in DICG_run_iter_w],
    [s[1] for s in DICG_run_iter_w],
    colors[4],
    marker=markers[4],
    markersize=size_marker,
    markevery = np.linspace(0, len([s[0] for s in DICG_run_iter_w]) - 2, 10, dtype = int).tolist(),
    linewidth=linewidth_figures,
    label = "DICG"
)
plt.semilogy(
    [s[0] for s in AFW_run_iter_w_sparse],
    [s[1] for s in AFW_run_iter_w_sparse],
    colors[5],
    marker=markers[5],
    markersize=size_marker,
    markevery = np.linspace(0, len([s[0] for s in AFW_run_iter_w_sparse]) - 2, 10, dtype = int).tolist(),
    linewidth=linewidth_figures,
    label = "AFW_sparse"
)
plt.title(
    f"{type(objective_function).__name__}\n"
    f"{type(feasible_region).__name__}\n"
    f"dim = {dimension}\n"
    f"L/m = {int(objective_function.L / objective_function.Mu)}"
)
plt.xlabel(x_axis)
plt.ylabel("Strong Wolfe Gap")
plt.legend()

plt.subplot(1, 2, 2)
PFLaCG_run_iter_w = [(run_status[x_index], run_status[2]- approx_f_opt) for run_status in PFLaCG_run]
AFW_run_iter_w = [(run_status[x_index], run_status[2] - approx_f_opt) for run_status in AFW_run]
AFW_run_iter_w_sparse = [(run_status[x_index], run_status[2] - approx_f_opt) for run_status in AFW_run_sparse]
PFW_run_iter_w = [(run_status[x_index], run_status[2] - approx_f_opt) for run_status in PFW_run]
Lazy_run_iter_w = [(run_status[x_index], run_status[2]- approx_f_opt) for run_status in Lazy_run]
DICG_run_iter_w = [(run_status[x_index], run_status[2]- approx_f_opt) for run_status in DIPFW_run]
plt.semilogy(
    [s[0] for s in PFLaCG_run_iter_w],
    [s[1] for s in PFLaCG_run_iter_w],
    colors[0],
    marker=markers[0],
    markersize=size_marker,
    markevery = np.linspace(0, len([s[0] for s in PFLaCG_run_iter_w]) - 2, 10, dtype = int).tolist(),
    linewidth=linewidth_figures,
    label = "PFLaCG"
)
plt.semilogy(
    [s[0] for s in AFW_run_iter_w],
    [s[1] for s in AFW_run_iter_w],
    colors[1],
    marker=markers[1],
    markersize=size_marker,
    markevery = np.linspace(0, len([s[0] for s in AFW_run_iter_w]) - 2, 10, dtype = int).tolist(),
    linewidth=linewidth_figures,
    label = "AFW"
)
plt.semilogy(
    [s[0] for s in PFW_run_iter_w],
    [s[1] for s in PFW_run_iter_w],
    colors[2],
    marker=markers[2],
    markersize=size_marker,
    markevery = np.linspace(0, len([s[0] for s in PFW_run_iter_w]) - 2, 10, dtype = int).tolist(),
    linewidth=linewidth_figures,
    label = "PFW"
)
plt.semilogy(
    [s[0] for s in Lazy_run_iter_w],
    [s[1] for s in Lazy_run_iter_w],
    colors[3],
    marker=markers[3],
    markersize=size_marker,
    markevery = np.linspace(0, len([s[0] for s in Lazy_run_iter_w]) - 2, 10, dtype = int).tolist(),
    linewidth=linewidth_figures,
    label = "Lazy"
)
plt.semilogy(
    [s[0] for s in DICG_run_iter_w],
    [s[1] for s in DICG_run_iter_w],
    colors[4],
    marker=markers[4],
    markersize=size_marker,
    markevery = np.linspace(0, len([s[0] for s in DICG_run_iter_w]) - 2, 10, dtype = int).tolist(),
    linewidth=linewidth_figures,
    label = "DICG"
)
plt.semilogy(
    [s[0] for s in AFW_run_iter_w_sparse],
    [s[1] for s in AFW_run_iter_w_sparse],
    colors[5],
    marker=markers[5],
    markersize=size_marker,
    markevery = np.linspace(0, len([s[0] for s in AFW_run_iter_w_sparse]) - 2, 10, dtype = int).tolist(),
    linewidth=linewidth_figures,
    label = "AFW_sparse"
)
plt.title(
    f"{type(objective_function).__name__}\n"
    f"{type(feasible_region).__name__}\n"
    f"dim = {dimension}\n"
    f"L/m = {int(objective_function.L / objective_function.Mu)}"
)
plt.xlabel(x_axis)
plt.ylabel("$f - f*$")
plt.legend()
plt.savefig("Birkhoff_Comparison_time_standard_test2.pdf")
# plt.show()
plt.close()


plt.subplot(1, 2, 1)
PFLaCG_run_iter_w = [(run_status[0], run_status[4]) for run_status in PFLaCG_run]
AFW_run_iter_w = [(run_status[0], run_status[4]) for run_status in AFW_run]
AFW_run_iter_w_sparse = [(run_status[0], run_status[4]) for run_status in AFW_run_sparse]
PFW_run_iter_w = [(run_status[0], run_status[4]) for run_status in PFW_run]
Lazy_run_iter_w = [(run_status[0], run_status[4]) for run_status in Lazy_run]
DICG_run_iter_w = [(run_status[0], run_status[4]) for run_status in DIPFW_run]
plt.semilogy(
    [s[0] for s in PFLaCG_run_iter_w],
    [s[1] for s in PFLaCG_run_iter_w],
    colors[0],
    marker=markers[0],
    markersize=size_marker,
    markevery = np.linspace(0, len([s[0] for s in PFLaCG_run_iter_w]) - 2, 10, dtype = int).tolist(),
    linewidth=linewidth_figures,
    label = "PFLaCG"
)
plt.semilogy(
    [s[0] for s in AFW_run_iter_w],
    [s[1] for s in AFW_run_iter_w],
    colors[1],
    marker=markers[1],
    markersize=size_marker,
    markevery = np.linspace(0, len([s[0] for s in AFW_run_iter_w]) - 2, 10, dtype = int).tolist(),
    linewidth=linewidth_figures,
    label = "AFW"
)
plt.semilogy(
    [s[0] for s in PFW_run_iter_w],
    [s[1] for s in PFW_run_iter_w],
    colors[2],
    marker=markers[2],
    markersize=size_marker,
    markevery = np.linspace(0, len([s[0] for s in PFW_run_iter_w]) - 2, 10, dtype = int).tolist(),
    linewidth=linewidth_figures,
    label = "PFW"
)
plt.semilogy(
    [s[0] for s in Lazy_run_iter_w],
    [s[1] for s in Lazy_run_iter_w],
    colors[3],
    marker=markers[3],
    markersize=size_marker,
    markevery = np.linspace(0, len([s[0] for s in Lazy_run_iter_w]) - 2, 10, dtype = int).tolist(),
    linewidth=linewidth_figures,
    label = "Lazy"
)
plt.semilogy(
    [s[0] for s in DICG_run_iter_w],
    [s[1] for s in DICG_run_iter_w],
    colors[4],
    marker=markers[4],
    markersize=size_marker,
    markevery = np.linspace(0, len([s[0] for s in DICG_run_iter_w]) - 2, 10, dtype = int).tolist(),
    linewidth=linewidth_figures,
    label = "DICG"
)
plt.semilogy(
    [s[0] for s in AFW_run_iter_w_sparse],
    [s[1] for s in AFW_run_iter_w_sparse],
    colors[5],
    marker=markers[5],
    markersize=size_marker,
    markevery = np.linspace(0, len([s[0] for s in AFW_run_iter_w_sparse]) - 2, 10, dtype = int).tolist(),
    linewidth=linewidth_figures,
    label = "AFW_sparse"
)
plt.title(
    f"{type(objective_function).__name__}\n"
    f"{type(feasible_region).__name__}\n"
    f"dim = {dimension}\n"
    f"L/m = {int(objective_function.L / objective_function.Mu)}"
)
plt.xlabel("Iteration")
plt.ylabel("Strong Wolfe Gap")
plt.legend()

plt.subplot(1, 2, 2)
PFLaCG_run_iter_w = [(run_status[0], run_status[2]- approx_f_opt) for run_status in PFLaCG_run]
AFW_run_iter_w = [(run_status[0], run_status[2] - approx_f_opt) for run_status in AFW_run]
AFW_run_iter_w_sparse = [(run_status[0], run_status[2] - approx_f_opt) for run_status in AFW_run_sparse]
PFW_run_iter_w = [(run_status[0], run_status[2] - approx_f_opt) for run_status in PFW_run]
Lazy_run_iter_w = [(run_status[0], run_status[2]- approx_f_opt) for run_status in Lazy_run]
DICG_run_iter_w = [(run_status[0], run_status[2]- approx_f_opt) for run_status in DIPFW_run]
plt.semilogy(
    [s[0] for s in PFLaCG_run_iter_w],
    [s[1] for s in PFLaCG_run_iter_w],
    colors[0],
    marker=markers[0],
    markersize=size_marker,
    markevery = np.linspace(0, len([s[0] for s in PFLaCG_run_iter_w]) - 2, 10, dtype = int).tolist(),
    linewidth=linewidth_figures,
    label = "PFLaCG"
)
plt.semilogy(
    [s[0] for s in AFW_run_iter_w],
    [s[1] for s in AFW_run_iter_w],
    colors[1],
    marker=markers[1],
    markersize=size_marker,
    markevery = np.linspace(0, len([s[0] for s in AFW_run_iter_w]) - 2, 10, dtype = int).tolist(),
    linewidth=linewidth_figures,
    label = "AFW"
)
plt.semilogy(
    [s[0] for s in PFW_run_iter_w],
    [s[1] for s in PFW_run_iter_w],
    colors[2],
    marker=markers[2],
    markersize=size_marker,
    markevery = np.linspace(0, len([s[0] for s in PFW_run_iter_w]) - 2, 10, dtype = int).tolist(),
    linewidth=linewidth_figures,
    label = "PFW"
)
plt.semilogy(
    [s[0] for s in Lazy_run_iter_w],
    [s[1] for s in Lazy_run_iter_w],
    colors[3],
    marker=markers[3],
    markersize=size_marker,
    markevery = np.linspace(0, len([s[0] for s in Lazy_run_iter_w]) - 2, 10, dtype = int).tolist(),
    linewidth=linewidth_figures,
    label = "Lazy"
)
plt.semilogy(
    [s[0] for s in DICG_run_iter_w],
    [s[1] for s in DICG_run_iter_w],
    colors[4],
    marker=markers[4],
    markersize=size_marker,
    markevery = np.linspace(0, len([s[0] for s in DICG_run_iter_w]) - 2, 10, dtype = int).tolist(),
    linewidth=linewidth_figures,
    label = "DICG"
)
plt.semilogy(
    [s[0] for s in AFW_run_iter_w_sparse],
    [s[1] for s in AFW_run_iter_w_sparse],
    colors[5],
    marker=markers[5],
    markersize=size_marker,
    markevery = np.linspace(0, len([s[0] for s in AFW_run_iter_w_sparse]) - 2, 10, dtype = int).tolist(),
    linewidth=linewidth_figures,
    label = "AFW_sparse"
)
plt.title(
    f"{type(objective_function).__name__}\n"
    f"{type(feasible_region).__name__}\n"
    f"dim = {dimension}\n"
    f"L/m = {int(objective_function.L / objective_function.Mu)}"
)
plt.xlabel("Iteration")
plt.ylabel("$f - f*$")
plt.legend()
plt.savefig("Birkhoff_Comparison_iteration_standard_test2.pdf")
# plt.show()
plt.close()
