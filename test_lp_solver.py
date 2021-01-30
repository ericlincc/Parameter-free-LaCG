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


dimension = 5
l1_regularization = 1.0
solver_type = "cvxopt"
sparse_solver = False
num_extra_constraints = 100

inequality_matrix = np.zeros((num_extra_constraints, dimension))
inequality_vector = np.zeros((num_extra_constraints))

for i in range(num_extra_constraints):
    vect = 2.0*(np.random.rand(dimension) - 0.5)
    vect = vect/np.linalg.norm(vect)
    inequality_matrix[i,:] = vect
    inequality_vector[i] = 0.5

lp_solver_scipy = ConstrainedL1BallPolytope(l1_regularization, dimension,const_matrix_ineq = inequality_matrix, const_vector_ineq = inequality_vector,  solver_type = "scipy", sparse_solver = False)
lp_solver_cvxopt = ConstrainedL1BallPolytope(l1_regularization, dimension,const_matrix_ineq = inequality_matrix, const_vector_ineq = inequality_vector,  solver_type = "cvxopt", sparse_solver = False)
l1_ball = L1UnitBallPolytope(dimension)

random_vector = np.random.rand(dimension)
sol_lp_solver_scipy = lp_solver_scipy.lp_oracle(random_vector)
sol_lp_solver_cvxopt = lp_solver_cvxopt.lp_oracle(random_vector)
sol_l1_ball  = l1_ball.lp_oracle(random_vector)

print("Are all the constraints satisfied? ", np.all(inequality_matrix.dot(sol_lp_solver_scipy) <= inequality_vector), np.all(inequality_matrix.dot(sol_lp_solver_cvxopt) <= inequality_vector))
print("scipy solution ", sol_lp_solver_scipy)
print("cvxopt solution ", sol_lp_solver_cvxopt)
print("l1 solution ", sol_l1_ball)
print("Difference between cvxopt and scipy solution ", np.linalg.norm(sol_lp_solver_scipy - sol_lp_solver_cvxopt))
print("Difference between l1 and scipy solution ", np.linalg.norm(sol_l1_ball - sol_lp_solver_scipy))
print("Are the two solutions close enough? ", np.allclose(sol_lp_solver_scipy, sol_lp_solver_cvxopt))


print("\n General setup")
lp_solver_general_scipy = GeneralPolytope(dimension,const_matrix_ineq = inequality_matrix, const_vector_ineq = inequality_vector,  solver_type = "scipy", sparse_solver = False)
lp_solver_general_cvxopt = GeneralPolytope(dimension, const_matrix_ineq = inequality_matrix, const_vector_ineq = inequality_vector,  solver_type = "cvxopt", sparse_solver = False)

sol_lp_solver_general_scipy = lp_solver_general_scipy.lp_oracle(random_vector)
sol_lp_solver_general_cvxopt = lp_solver_general_cvxopt.lp_oracle(random_vector)

print(sol_lp_solver_general_scipy)
print(sol_lp_solver_general_cvxopt)

print(np.linalg.norm(sol_lp_solver_general_scipy - sol_lp_solver_general_cvxopt))
