# -*- coding: utf-8 -*-
"""
Created on Sun Jan  3 16:05:54 2021

@author: pccom
"""
from pflacg.algorithms.fw_variants import FrankWolfe, FrankWolfe_backup
from pflacg.algorithms.pflacg import FractionalAwayStepFW
from pflacg.algorithms._algorithms_utils import ExitCriterion
from pflacg.experiments.objective_functions import Quadratic
from pflacg.experiments.feasible_regions import ProbabilitySimplexPolytope, BirkhoffPolytope, L1UnitBallPolytope, L2UnitBallPolytope

import numpy as np

test_exit_criterion = ExitCriterion("DG", 1.0e-12, criterion_reference=0.0, max_time=60.0, max_iter=1000)
dimension = 1600
matrix = np.random.rand(dimension, dimension)
M = matrix.T.dot(matrix)
b = np.random.rand(dimension)
function = Quadratic(dimension, M, b)
feasible_region = ProbabilitySimplexPolytope(dimension)
# feasible_region = BirkhoffPolytope(dimension) #For the Birkhoff polytope the dimension needs to be such that sqrt(dimension) is an integer.
# feasible_region = L1UnitBallPolytope(dimension)
# feasible_region = L2UnitBallPolytope(dimension)

FW_algorithm_new = FrankWolfe("AFW", "line_search")
results_new = FW_algorithm_new.run(function, feasible_region, test_exit_criterion)

FW_algorithm = FrankWolfe_backup("AFW", "line_search")
results_old = FW_algorithm.run(function, feasible_region, test_exit_criterion)

dual_gap_new = [dual_gap for iteration, duration, f_val, dual_gap, strong_wolfe_gap in results_new]
timing_new = [duration for iteration, duration, f_val, dual_gap, strong_wolfe_gap in results_new]
# primal_gap_new = [f_val-fVal_opt for iteration, duration, f_val, dual_gap, strong_wolfe_gap in results_new]

dual_gap_old = [dual_gap for iteration, duration, f_val, dual_gap, strong_wolfe_gap in results_old]
timing_old = [duration for iteration, duration, f_val, dual_gap, strong_wolfe_gap in results_old]

import matplotlib.pyplot as plt

plt.subplot(1, 2, 1)
plt.semilogy(timing_new, dual_gap_new, label = 'New implementation')
plt.semilogy(timing_old, dual_gap_old, label = 'Old implementation')
plt.xlabel('Time')
plt.ylabel('FW gap')
plt.legend()

plt.subplot(1, 2, 2)
plt.semilogy(np.arange(len(dual_gap_new)), dual_gap_new, label = 'New implementation')
plt.semilogy(np.arange(len(dual_gap_old)), dual_gap_old, label = 'Old implementation')
plt.xlabel('Iteration')
plt.ylabel('FW gap')
plt.legend()
plt.show()