import logging
import time
from multiprocessing import shared_memory, Value, Process, Lock

import numpy as np

from pflacg.algorithms._algorithms_utils import *
from pflacg.algorithms.pflacg import ParameterFreeAGD, ParameterFreeLaCG

from pflacg.algorithms.fw_variants import FrankWolfe

from pflacg.experiments.objective_functions import *
from pflacg.experiments.feasible_regions import *

import numpy as np

np.random.seed(60)

test_exit_criterion = ExitCriterion("SWG", 1.0e-5, criterion_reference=0.0, max_time=600.0, max_iter=10000)
layers = 30
nodes_per_layer = 5

feasible_region = flow_polytope(layers, nodes_per_layer)

dimension = feasible_region.dimension()
matrix = np.random.rand(dimension, dimension)
M = matrix.T.dot(matrix)
M = M + np.identity(dimension)
b = np.random.rand(dimension)
objective_function = Quadratic(dimension, M, b)

PFLaCG = ParameterFreeLaCG()
point_x = Point(
    feasible_region.initial_point,
    (1,),
    (feasible_region.initial_point,)
)


print(objective_function.Mu)
print(objective_function.L)


# results_PFLaCG = PFLaCG.run(objective_function, feasible_region, test_exit_criterion, point_x)

FW_algorithm_new = FrankWolfe("AFW", "line_search")
results_new = FW_algorithm_new.run(objective_function, feasible_region, test_exit_criterion)

dual_gap_new = [strong_wolfe_gap for iteration, duration, f_val, dual_gap, strong_wolfe_gap in results_new]
timing_new = [duration for iteration, duration, f_val, dual_gap, strong_wolfe_gap in results_new]

# dual_gap_PFLaCG = [strong_wolfe_gap for iteration, duration, f_val, dual_gap, strong_wolfe_gap in results_PFLaCG]
# iterations_PFLaCG = [iteration for iteration, duration, f_val, dual_gap, strong_wolfe_gap in results_PFLaCG]
# timing_PFLaCG = [duration for iteration, duration, f_val, dual_gap, strong_wolfe_gap in results_PFLaCG]

import matplotlib.pyplot as plt

plt.subplot(1, 2, 1)
plt.semilogy(timing_new, dual_gap_new, label = 'AFW')
# plt.semilogy(timing_PFLaCG, dual_gap_PFLaCG, label = 'PFLaCG')
plt.xlabel('Time')
plt.ylabel('Strong-Wolfe gap')
plt.legend()

plt.subplot(1, 2, 2)
plt.semilogy(np.arange(len(dual_gap_new)), dual_gap_new, label = 'AFW')
# plt.semilogy(iterations_PFLaCG, dual_gap_PFLaCG, label = 'PFLaCG')
plt.xlabel('Iteration')
plt.ylabel('Strong-Wolfe gap')
plt.legend()
plt.savefig('Example_run.pdf')
