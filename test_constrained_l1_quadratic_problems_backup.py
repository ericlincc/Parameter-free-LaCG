from os import environ
environ["MKL_NUM_THREADS"] = "1"
environ["NUMEXPR_NUM_THREADS"] = "1"
environ["OMP_NUM_THREADS"] = "1"

from pflacg.experiments.feasible_regions import *
from pflacg.experiments.objective_functions import *
from pflacg.algorithms.fw_variants import *
from pflacg.algorithms.pflacg import *
from pflacg.algorithms._algorithms_utils import ExitCriterion

import random
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s :: %(asctime)s :: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

PATH_TO_PICKLE_BASE = "/scratch/share/pflacg_experiments/pickled_objects"
dim = 100
l1_regularization = 1.0

constraints = 25
equality_matrix = np.zeros((constraints, dim))
equality_vector = np.zeros((constraints))
entries = random.sample(range(dim), 2*constraints)
entries_left = entries[:int(len(entries)/2)]
entries_right = entries[int(len(entries)/2):]
print(equality_matrix.shape, equality_vector.shape)
for i in range(constraints):
    equality_matrix[i, entries_left[i]] = 1.0
    equality_matrix[i, entries_right[i]] = -1.0
    equality_vector[i] = 0.0

feasible_region = ConstrainedL1BallPolytope(
    l1_regularization,
    dim,
    const_matrix_eq=equality_matrix,
    const_vector_eq=equality_vector,
    solver_type="scipy",
    sparse_solver=False,
)

mat = np.random.rand(dim, dim)
mat = mat.dot(mat.T) + np.identity(dim)

optimum = 10.0*np.random.rand(dim)
fun = Quadratic(dim, mat, optimum)

accuracy = 0.00000001
exit_criterion = ExitCriterion("SWG", accuracy, max_time=1000)
AFW_algorithm = FrankWolfe("AFW", "line_search") 
AFW_results, output_point = AFW_algorithm.run(fun, feasible_region, exit_criterion)

#AGD_alg = ParameterFreeAGD(iter_sync = True)
#feasible_region_convex_hull = ConvexHull(list(output_point.support))
#barycentric_coordinates = np.zeros(len(output_point.support))
#barycentric_coordinates[0] = 1.0
#print()
#point_x = Point(output_point.support[0], tuple(barycentric_coordinates), output_point.support)
#gradient_output = fun.evaluate_grad(point_x.cartesian_coordinates)
#v = feasible_region.lp_oracle(gradient_output)
#a, index_max = feasible_region.away_oracle(gradient_output, point_x)
#print("SWG gap before: ", gradient_output.dot(a.cartesian_coordinates - v))
#point_x, eta, sigma, iteration = AGD_alg.run(fun, feasible_region_convex_hull, point_x, epsilon = accuracy)
#gradient_output = fun.evaluate_grad(point_x.cartesian_coordinates)
#v = feasible_region.lp_oracle(gradient_output)
#a, index_max = feasible_region.away_oracle(gradient_output, point_x)
#print("SWG gap: ", gradient_output.dot(a.cartesian_coordinates - v))
#quit()

PFLaCG_algorithm = ParameterFreeLaCG("AFW", iter_sync=False) 
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
plt.savefig("L1_ball_comparison_2.pdf")
# plt.show()
plt.close()