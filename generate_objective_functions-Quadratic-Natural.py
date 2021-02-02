import pickle
from os import path

import numpy as np
from scipy.sparse import csr_matrix

from pflacg.experiments.objective_functions import *


PATH_TO_PICKLE_BASE = "/scratch/share/pflacg_experiments/pickled_objects"



DIMENSIONS = [10000, 15000, 20000]
CONDITION_NUMS = [1e6]
RANDOM_SEEDS = [i for i in range(2)]


# Dense quadratic generation - natural way
for dim in DIMENSIONS:
    for condition_num in CONDITION_NUMS:
        for rand_seed in RANDOM_SEEDS:
            np.random.seed(rand_seed)

            matrix = np.random.rand(dim, dim)
            matrix = matrix.T.dot(matrix)
            L, mu = calculate_max_and_min_eigenvalues(matrix)
            matrix = matrix * (condition_num / L) + np.identity(dim)
            b = np.random.rand(dim)
            objective_function = Quadratic(dim, matrix, b)

            with open(
                path.join(
                    PATH_TO_PICKLE_BASE,
                    "objective_functions",
                    f"{type(objective_function).__name__}"
                    f"-Natural-dim_{dim}-cond_{condition_num}-seed_{rand_seed}.pickle",
                ),
                "wb"
            ) as f:
                pickle.dump(objective_function, f)
                print("Pickle object dumped.")


