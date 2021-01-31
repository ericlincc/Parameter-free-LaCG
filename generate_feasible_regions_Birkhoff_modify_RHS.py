import pickle
from os import path

import numpy as np
from scipy.sparse import csr_matrix

from pflacg.experiments.feasible_regions import *


PATH_TO_PICKLE_BASE = "/scratch/share/pflacg_experiments/pickled_objects"

DIMENSIONS = [20**2, 30**2, 40**2, 50**2]
SEEDS = [1, 2, 3, 4, 5]
# Dense quadratic generation - uniform evalues
for dim in DIMENSIONS:
    for seed in SEEDS:
        np.random.seed(seed)
        right_hand_side_vector = np.ones(2*int(np.sqrt(dim)) - 1) +  0.1*(np.random.rand(2*int(np.sqrt(dim))- 1) - 0.5)
        feasible_region = ConstrainedBirkhoffPolytope(
            dim,
            linear_equality_vector = right_hand_side_vector,
        )
        
        with open(
            path.join(
                PATH_TO_PICKLE_BASE,
                "feasible_regions",
                f"{type(feasible_region).__name__}-modifiedRHS-dim_{dim}_seed_{seed}.pickle",
            ),
            "wb",
        ) as f:
            pickle.dump(feasible_region, f)
            print("Pickle object dumped.")
