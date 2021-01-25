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
from pflacg.algorithms.pflacg_simplex import ParameterFreeLaCG_simplex, ParameterFreeAGD_simplex
from pflacg.algorithms.fw_variants import FrankWolfe_simplex
from pflacg.experiments.objective_functions import *
from pflacg.experiments.feasible_regions import *


# In[2]:


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
dimension = 5000


matrix = np.random.rand(dimension, dimension)
M = matrix.T.dot(matrix)
M = M + 500.0 * np.identity(dimension)
b = np.random.rand(dimension)
objective_function = Quadratic(dimension, M, b)
feasible_region = ProbabilitySimplexPolytope(dimension)
x = feasible_region.initial_point

print(objective_function.Mu)
print(objective_function.L)


# # active_set_point = np.ones(dimension)/dimension

# active_set_point = np.zeros(dimension)
# for i in range(int(dimension/2.0)):
#     active_set_point[i] = 1.0/(dimension/2.0)

# init_point = np.zeros(dimension)
# init_point[0] = 1.0

# ParameterFreeAGD_alg = ParameterFreeAGD_simplex()
# ParameterFreeAGD_alg.run(objective_function, active_set_point,init_point, epsilon=1.0e-4, initial_eta = 0.0, initial_sigma=0.0)

# quit()

# Computing approx x* using AFW
AFW_algorithm = FrankWolfe_simplex("AFW", "line_search")
optimal_AFW_run, FW_gap, SW_gap = AFW_algorithm.run(objective_function, feasible_region, optimal_exit_criterion, x, save_and_output_results = False)

print("Number of vertices in the optimal solution: ", np.count_nonzero(optimal_AFW_run))
approx_f_opt = objective_function.evaluate(optimal_AFW_run)

# In[5]:


PFLaCG = ParameterFreeLaCG_simplex(iter_sync=ITER_SYNC)
PFLaCG_run = PFLaCG.run(objective_function, feasible_region, test_exit_criterion, x)

# In[5]:



AFW_run = AFW_algorithm.run(objective_function, feasible_region, test_exit_criterion, x)


# In[6]:


PFW_algorithm = FrankWolfe_simplex("PFW", "line_search")
PFW_run = PFW_algorithm.run(objective_function, feasible_region, test_exit_criterion, x)


# ----------

# In[7]:



# ----------

# In[8]:



if ITER_SYNC:
    x_index, x_axis = 0, "Iter"
else:
    x_index, x_axis = 1, "Time"

plt.subplot(1, 2, 1)
PFLaCG_run_iter_w = [(run_status[x_index], run_status[4]) for run_status in PFLaCG_run]
AFW_run_iter_w = [(run_status[x_index], run_status[4]) for run_status in AFW_run]
PFW_run_iter_w = [(run_status[x_index], run_status[4]) for run_status in PFW_run]
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
plt.xlabel(x_axis)
plt.ylabel("Strong Wolfe Gap")
plt.legend()

plt.subplot(1, 2, 2)
PFLaCG_run_iter_w = [(run_status[x_index], run_status[2]- approx_f_opt) for run_status in PFLaCG_run]
AFW_run_iter_w = [(run_status[x_index], run_status[2] - approx_f_opt) for run_status in AFW_run]
PFW_run_iter_w = [(run_status[x_index], run_status[2] - approx_f_opt) for run_status in PFW_run]
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
plt.xlabel(x_axis)
plt.ylabel("$f - f*$")
plt.legend()
plt.savefig("Comparison_time_v6.pdf")
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
plt.savefig("Comparison_iteration_v6.pdf")
# plt.show()
plt.close()


# In[ ]:





# In[ ]:




