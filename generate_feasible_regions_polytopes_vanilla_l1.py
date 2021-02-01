import pickle
from os import path

import numpy as np
from scipy.sparse import csr_matrix

from pflacg.experiments.feasible_regions import *


PATH_TO_PICKLE_BASE = "/scratch/share/pflacg_experiments/pickled_objects"

DIMENSIONS = [100, 250, 500, 750, 1000]
l1_regularization = 1.0

# Dense quadratic generation - uniform evalues
for dim in DIMENSIONS:
    feasible_region = ConstrainedL1BallPolytope(
        l1_regularization,
        dim,
        solver_type="scipy",
        sparse_solver=False,
    )

    with open(
        path.join(
            PATH_TO_PICKLE_BASE,
            "feasible_regions",
            f"{type(feasible_region).__name__}-dim_{dim}.pickle",
            ),
        "wb",
        ) as f:
        pickle.dump(feasible_region, f)
        print("Pickle object dumped.")
