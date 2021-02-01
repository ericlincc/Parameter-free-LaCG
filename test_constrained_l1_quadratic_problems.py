from pflacg.experiments.feasible_regions import *
from pflacg.experiments.objective_functions import *
from pflacg.algorithms.fw_variants import *
from pflacg.algorithms.pflacg import *
from pflacg.algorithms._algorithms_utils import ExitCriterion
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s :: %(asctime)s :: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

PATH_TO_PICKLE_BASE = "/scratch/share/pflacg_experiments/pickled_objects"
dim = 400
l1_regularization = 1.0

# constraints = 50
# inequality_matrix = np.zeros((constraints, dim))
# inequality_vector = np.zeros((constraints))
# for i in range(constraints):
#     vect = 2.0 * (np.random.rand(dim) - 0.5)
#     vect = vect / np.linalg.norm(vect)
#     inequality_matrix[i, :] = vect
#     inequality_vector[i] = np.sqrt(1 / dim)

feasible_region = ConstrainedL1BallPolytope(
    l1_regularization,
    dim,
    # const_matrix_ineq=inequality_matrix,
    # const_vector_ineq=inequality_vector,
    solver_type="scipy",
    sparse_solver=False,
)


mat = np.random.rand(dim, dim)
mat = mat.dot(mat.T) + np.identity(dim)
num_nonzero_elem = 20

import random
entries = random.sample(range(dim), num_nonzero_elem)
optimum = np.zeros(dim)
optimum[entries] = 0.1*(np.random.rand(num_nonzero_elem) - 0.5)

assert np.sum(np.abs(optimum)) <= 1.0, "Global optimum is not contained in the interior."

print(np.sum(np.abs(optimum)))

fun = Quadratic(dim, mat, optimum)


exit_criterion = ExitCriterion("SWG", 0.001)
AFW_algorithm = FrankWolfe("AFW", "line_search") 
AFW_results = AFW_algorithm.run(fun, feasible_region, exit_criterion)

PFLaCG_algorithm = ParameterFreeLaCG("AFW") 
PFLaCG_results = PFLaCG_algorithm.run(fun, feasible_region, exit_criterion)

colors = ["k", "c", "b", "m", "r", "g"]
markers = ["o", "s", "^", "P", "D", "p"]
size_marker = 12
fontsize = 19
fontsize_legend = 20
linewidth_figures = 1.50

import matplotlib.pyplot as plt

plt.subplot(1, 2, 1)
PFLaCG_run_iter_w = [(run_status[1], run_status[4]) for run_status in PFLaCG_results]
AFW_run_iter_w = [(run_status[1], run_status[4]) for run_status in AFW_results]
plt.semilogy(
    [s[0] for s in PFLaCG_run_iter_w],
    [s[1] for s in PFLaCG_run_iter_w],
    colors[0],
    # marker=markers[0],
    # markersize=size_marker,
    # markevery = np.linspace(0, len([s[0] for s in PFLaCG_run_iter_w]) - 2, 10, dtype = int).tolist(),
    linewidth=linewidth_figures,
    label = "PFLaCG"
)
plt.semilogy(
    [s[0] for s in AFW_run_iter_w],
    [s[1] for s in AFW_run_iter_w],
    colors[1],
    # marker=markers[1],
    # markersize=size_marker,
    # markevery = np.linspace(0, len([s[0] for s in AFW_run_iter_w]) - 2, 10, dtype = int).tolist(),
    linewidth=linewidth_figures,
    label = "AFW"
)
plt.xlabel("Time")
plt.ylabel("Strong Wolfe Gap")
plt.legend()

plt.subplot(1, 2, 2)
PFLaCG_run_iter_w = [(run_status[0], run_status[4]) for run_status in PFLaCG_results]
AFW_run_iter_w = [(run_status[0], run_status[4]) for run_status in AFW_results]
plt.semilogy(
    [s[0] for s in PFLaCG_run_iter_w],
    [s[1] for s in PFLaCG_run_iter_w],
    colors[0],
    # marker=markers[0],
    # markersize=size_marker,
    # markevery = np.linspace(0, len([s[0] for s in PFLaCG_run_iter_w]) - 2, 10, dtype = int).tolist(),
    linewidth=linewidth_figures,
    label = "PFLaCG"
)
plt.semilogy(
    [s[0] for s in AFW_run_iter_w],
    [s[1] for s in AFW_run_iter_w],
    colors[1],
    # marker=markers[1],
    # markersize=size_marker,
    # markevery = np.linspace(0, len([s[0] for s in AFW_run_iter_w]) - 2, 10, dtype = int).tolist(),
    linewidth=linewidth_figures,
    label = "AFW"
)
plt.xlabel("Iteration")
plt.ylabel("Strong Wolfe Gap")
plt.legend()
plt.savefig("L1_ball_comparison.pdf")
# plt.show()
plt.close()