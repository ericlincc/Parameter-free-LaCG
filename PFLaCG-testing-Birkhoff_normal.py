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
from pflacg.algorithms.fw_variants import FrankWolfe
from pflacg.experiments.objective_functions import *
from pflacg.experiments.feasible_regions import *


# In[5]:


ITER_SYNC = False
RANDOM_SEED = 1


# In[3]:


np.random.seed(RANDOM_SEED)

test_exit_criterion = ExitCriterion(
    "SWG",
    1.0e-6,
    criterion_reference=0.0,
    max_time=600.0,
    max_iter=10000
)
optimal_exit_criterion = ExitCriterion(
    "DG",
    1.0e-6,
    criterion_reference=0.0,
    max_time=600.0,
    max_iter=10000
)
dimension = 10000


matrix = np.random.rand(dimension, dimension)
M = matrix.T.dot(matrix)
M = M + 500 * np.identity(dimension)
b = np.random.rand(dimension)
objective_function = Quadratic(dimension, M, b)
feasible_region = BirkhoffPolytope(dimension)
point_x = Point(
    feasible_region.initial_point,
    (1,),
    (feasible_region.initial_point,)
)
print(objective_function.Mu)
print(objective_function.L)

DICG_algorithm = FrankWolfe("DIPFW", "line_search")
DICG_run = PFW_algorithm.run(objective_function, feasible_region, test_exit_criterion, point_x)

Lazy_algorithm = FrankWolfe("lazy", "line_search")
Lazy_run = PFW_algorithm.run(objective_function, feasible_region, test_exit_criterion, point_x)

PFLaCG = ParameterFreeLaCG(iter_sync=ITER_SYNC)
PFLaCG_run = PFLaCG.run(objective_function, feasible_region, test_exit_criterion, point_x)

AFW_algorithm = FrankWolfe("AFW", "line_search")
AFW_run = AFW_algorithm.run(objective_function, feasible_region, test_exit_criterion, point_x)

PFW_algorithm = FrankWolfe("PFW", "line_search")
PFW_run = PFW_algorithm.run(objective_function, feasible_region, test_exit_criterion, point_x)

optimal_AFW_run = DICG_algorithm.run(objective_function, feasible_region, optimal_exit_criterion, point_x)


approx_f_opt = optimal_AFW_run[-1][2]

if ITER_SYNC:
    x_index, x_axis = 0, "Iter"
else:
    x_index, x_axis = 1, "Time"

plt.subplot(1, 2, 1)
PFLaCG_run_iter_w = [(run_status[x_index], run_status[4]) for run_status in PFLaCG_run]
AFW_run_iter_w = [(run_status[x_index], run_status[4]) for run_status in AFW_run]
PFW_run_iter_w = [(run_status[x_index], run_status[4]) for run_status in PFW_run]
Lazy_run_iter_w = [(run_status[x_index], run_status[4]) for run_status in Lazy_run]
DICG_run_iter_w = [(run_status[x_index], run_status[4]) for run_status in DICG_run]
plt.semilogy(
    [s[0] for s in PFLaCG_run_iter_w],
    [s[1] for s in PFLaCG_run_iter_w],
    label = "PFLaCG"
)
plt.semilogy(
    [s[0] for s in AFW_run_iter_w],
    [s[1] for s in AFW_run_iter_w],
    label = "AFW"
)
plt.semilogy(
    [s[0] for s in PFW_run_iter_w],
    [s[1] for s in PFW_run_iter_w],
    label = "PFW"
)
plt.semilogy(
    [s[0] for s in Lazy_run_iter_w],
    [s[1] for s in Lazy_run_iter_w],
    label = "Lazy"
)
plt.semilogy(
    [s[0] for s in DICG_run_iter_w],
    [s[1] for s in DICG_run_iter_w],
    label = "DICG"
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
PFW_run_iter_w = [(run_status[x_index], run_status[2] - approx_f_opt) for run_status in PFW_run]
Lazy_run_iter_w = [(run_status[x_index], run_status[2]) for run_status in Lazy_run]
DICG_run_iter_w = [(run_status[x_index], run_status[2]) for run_status in DICG_run]
plt.semilogy(
    [s[0] for s in PFLaCG_run_iter_w],
    [s[1] for s in PFLaCG_run_iter_w],
    label = "PFLaCG"
)
plt.semilogy(
    [s[0] for s in AFW_run_iter_w],
    [s[1] for s in AFW_run_iter_w],
    label = "AFW"
)
plt.semilogy(
    [s[0] for s in PFW_run_iter_w],
    [s[1] for s in PFW_run_iter_w],
    label = "PFW"
)
plt.semilogy(
    [s[0] for s in Lazy_run_iter_w],
    [s[1] for s in Lazy_run_iter_w],
    label = "Lazy"
)
plt.semilogy(
    [s[0] for s in DICG_run_iter_w],
    [s[1] for s in DICG_run_iter_w],
    label = "DICG"
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
plt.savefig("Birkhoff_Comparison_time_standard_v12.pdf")
# plt.show()
plt.close()


plt.subplot(1, 2, 1)
PFLaCG_run_iter_w = [(run_status[0], run_status[4]) for run_status in PFLaCG_run]
AFW_run_iter_w = [(run_status[0], run_status[4]) for run_status in AFW_run]
PFW_run_iter_w = [(run_status[0], run_status[4]) for run_status in PFW_run]
plt.semilogy(
    [s[0] for s in PFLaCG_run_iter_w],
    [s[1] for s in PFLaCG_run_iter_w],
    label = "PFLaCG"
)
plt.semilogy(
    [s[0] for s in AFW_run_iter_w],
    [s[1] for s in AFW_run_iter_w],
    label = "AFW"
)
plt.semilogy(
    [s[0] for s in PFW_run_iter_w],
    [s[1] for s in PFW_run_iter_w],
    label = "PFW"
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
PFW_run_iter_w = [(run_status[0], run_status[2] - approx_f_opt) for run_status in PFW_run]
plt.semilogy(
    [s[0] for s in PFLaCG_run_iter_w],
    [s[1] for s in PFLaCG_run_iter_w],
    label = "PFLaCG"
)
plt.semilogy(
    [s[0] for s in AFW_run_iter_w],
    [s[1] for s in AFW_run_iter_w],
    label = "AFW"
)
plt.semilogy(
    [s[0] for s in PFW_run_iter_w],
    [s[1] for s in PFW_run_iter_w],
    label = "PFW"
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
plt.savefig("Birkhoff_Comparison_iteration_standard_v12.pdf")
# plt.show()
plt.close()




