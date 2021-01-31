import pickle
from os import path

import numpy as np
from scipy.sparse import csr_matrix

from pflacg.experiments.feasible_regions import *


PATH_TO_PICKLE_BASE = "/scratch/share/pflacg_experiments/pickled_objects"

DIMENSIONS = [100, 500, 1000]
NUM_EXTRA_CONSTRAINTS = [100, 500, 1000]
SEEDS = [1, 2, 3]
l1_regularization = 1.0

# Dense quadratic generation - uniform evalues
for dim in DIMENSIONS:
    for constraints in NUM_EXTRA_CONSTRAINTS:
        for seed in SEEDS:
            np.random.rand(seed)
            inequality_matrix = np.zeros((constraints, dim))
            inequality_vector = np.zeros((constraints))

            for i in range(constraints):
                vect = 2.0 * (np.random.rand(dim) - 0.5)
                vect = vect / np.linalg.norm(vect)
                inequality_matrix[i, :] = vect
                inequality_vector[i] = np.sqrt(1 / dimension)

            lp_solver_scipy = ConstrainedL1BallPolytope(
                l1_regularization,
                dimension,
                const_matrix_ineq=inequality_matrix,
                const_vector_ineq=inequality_vector,
                solver_type="scipy",
                sparse_solver=False,
            )

            with open(
                path.join(
                    PATH_TO_PICKLE_BASE,
                    "feasible_regions",
                    f"{type(feasible_region).__name__}-dim_{dim}_constraints_{constraints}_seed_{seed}.pickle",
                ),
                "wb",
            ) as f:
                pickle.dump(lp_solver_scipy, f)
                print("Pickle object dumped.")
