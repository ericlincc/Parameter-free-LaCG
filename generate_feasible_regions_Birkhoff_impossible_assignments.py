import pickle
from os import path

import numpy as np
from scipy.sparse import csr_matrix

from pflacg.experiments.feasible_regions import *


PATH_TO_PICKLE_BASE = "/scratch/share/pflacg_experiments/pickled_objects"

DIMENSIONS = [20**2]
SEEDS = [1, 2, 3, 4, 5]
NUM_EXTRA_CONSTRAINTS = [10, 20, 50, 100]
# Dense quadratic generation - uniform evalues
for dim in DIMENSIONS:
    for seed in SEEDS:
        for num_constraints in NUM_EXTRA_CONSTRAINTS:
            np.random.seed(seed)
            import random
            equality_constraint_matrix = np.zeros((num_constraints, dim))
            entries = random.sample(range(dim), num_constraints)
            for i in range(num_constraints):
                equality_constraint_matrix[i,entries[i]] = 1.0
            equality_constraint_vector = np.zeros(num_constraints)
            feasible_region = ConstrainedBirkhoffPolytope(
                dim,
                const_matrix_eq=equality_constraint_matrix,
                const_vector_eq=equality_constraint_vector,
                scipy_solver="revised simplex",
            )
            
            with open(
                path.join(
                    PATH_TO_PICKLE_BASE,
                    "feasible_regions",
                    f"{type(feasible_region).__name__}-impossibleAssignments--dim_{dim}_extraassignments_{num_constraints}_seed_{seed}.pickle",
                ),
                "wb",
            ) as f:
                pickle.dump(feasible_region, f)
                print("Pickle object dumped.")
