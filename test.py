# -*- coding: utf-8 -*-
"""
Created on Sun Jan  3 16:05:54 2021

@author: pccom
"""
from pflacg.algorithms.fw_variants import FrankWolfe, FrankWolfe_backup
from pflacg.algorithms.pflacg import FractionalAwayStepFW
from pflacg.algorithms._algorithms_utils import ExitCriterion
from pflacg.experiments.objective_functions import Quadratic
from pflacg.experiments.feasible_regions import ProbabilitySimplexPolytope, BirkhoffPolytope

import numpy as np






test_exit_criterion = ExitCriterion("DG", 1.0e-6, criterion_reference=0.0, max_time=20.0, max_iter=100)
dimension = 25
matrix = np.random.rand(dimension, dimension)
M = matrix.T.dot(matrix)
b = np.random.rand(dimension)
function = Quadratic(dimension, M, b)
feasible_region = ProbabilitySimplexPolytope(dimension)
# feasible_region = BirkhoffPolytope(dimension)

# initial_point = np.zeros(dimension)
# initial_point[0] = 1.0
# active_set = [initial_point]
# lambdas = [1.0]
# FAFW = FractionalAwayStepFW(fw_variant = "AFW")
# output = FAFW.run(function, feasible_region, initial_point, active_set, lambdas)

# quit()

FW_algorithm = FrankWolfe_backup("AFW", "adaptive_short_step")
results = FW_algorithm.run(function, feasible_region, test_exit_criterion)

FW_algorithm = FrankWolfe("AFW", "adaptive_short_step")
results = FW_algorithm.run(function, feasible_region, test_exit_criterion)





