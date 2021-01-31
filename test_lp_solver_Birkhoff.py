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
from scipy.sparse import csc_matrix

from scipy.sparse import identity
from scipy.sparse.linalg import eigsh


dimension = 400
lp_solver_scipy = ConstrainedBirkhoffPolytope(
    dimension,
    scipy_solver="revised simplex",
)

Birkhoff_polytope = BirkhoffPolytope(dimension)

random_vector = np.random.rand(dimension)
import time

ref_time = time.time()
sol_lp_solver_scipy = lp_solver_scipy.lp_oracle(random_vector)
print("scipy time ", time.time() - ref_time)
ref_time = time.time()
sol_closed_form = Birkhoff_polytope.lp_oracle(random_vector)
print("Closed-form time ", time.time() - ref_time)

print("scipy solution ", sol_lp_solver_scipy)
print("Birkhoff solution ", sol_closed_form)
print(
    "Difference between l1 and scipy solution ",
    np.linalg.norm(sol_closed_form - sol_lp_solver_scipy),
)


print("Testing perturbed RHS.")

right_hand_side_vector = np.ones(2*int(np.sqrt(dimension)) - 1) +  0.1*(np.random.rand(2*int(np.sqrt(dimension))- 1) - 0.5)

lp_solver_scipy = ConstrainedBirkhoffPolytope(
    dimension,
    linear_equality_vector = right_hand_side_vector,
    scipy_solver="revised simplex",
)

ref_time = time.time()
sol_lp_solver_scipy = lp_solver_scipy.lp_oracle(random_vector)
print("scipy time ", time.time() - ref_time)
print("scipy solution ", sol_lp_solver_scipy)



import random
num_extra_constraints = 10
equality_constraint_matrix = np.zeros((num_extra_constraints, dimension))
entries = random.sample(range(dimension), num_extra_constraints)
for i in range(num_extra_constraints):
    equality_constraint_matrix[i,entries[i]] = 1.0
equality_constraint_vector = np.zeros(num_extra_constraints)
    
lp_solver_scipy = ConstrainedBirkhoffPolytope(
    dimension,
    const_matrix_eq=equality_constraint_matrix,
    const_vector_eq=equality_constraint_vector,
    scipy_solver="revised simplex",
)

ref_time = time.time()
sol_lp_solver_scipy = lp_solver_scipy.lp_oracle(random_vector)
print("scipy time ", time.time() - ref_time)
print("scipy solution ", sol_lp_solver_scipy)