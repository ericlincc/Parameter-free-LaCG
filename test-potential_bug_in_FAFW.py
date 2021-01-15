import logging
import time
from multiprocessing import shared_memory, Value, Process, Lock

import numpy as np

from pflacg.algorithms._algorithms_utils import *
from pflacg.algorithms.pflacg import ParameterFreeAGD, ParameterFreeLaCG, FractionalAwayStepFW

from pflacg.experiments.objective_functions import *
from pflacg.experiments.feasible_regions import *

import numpy as np

np.random.seed(1)

test_exit_criterion = ExitCriterion("SWG", 1.0e-10, criterion_reference=0.0, max_time=60.0, max_iter=1000)
dimension = 500

matrix = np.random.rand(dimension, dimension)
M = matrix.T.dot(matrix)
M = M + np.identity(dimension)
b = np.random.rand(dimension)
objective_function = Quadratic(dimension, M, b)
feasible_region = ProbabilitySimplexPolytope(dimension)


point_x = Point(
    feasible_region.initial_point,
    (1,),
    (feasible_region.initial_point,)
)

# FAFW_algorithm = FractionalAwayStepFW()
# for i in range(10):
#     point_x = FAFW_algorithm.run(objective_function,feasible_region, point_x)
#     print()

# quit()

PFLaCG = ParameterFreeLaCG()


print(objective_function.Mu)
print(objective_function.L)


PFLaCG.run(objective_function, feasible_region, test_exit_criterion, point_x)