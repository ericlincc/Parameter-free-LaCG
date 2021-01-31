import pickle
from os import path

import numpy as np
from scipy.sparse import csr_matrix

from pflacg.experiments.feasible_regions import *


PATH_TO_PICKLE_BASE = "/scratch/share/pflacg_experiments/pickled_objects"

DIMENSIONS = [20**2, 30**2, 40**2, 50**2]
# Dense quadratic generation - uniform evalues
for dim in DIMENSIONS:
    feasible_region = ConstrainedBirkhoffPolytope(
        dim,
    )

    with open(
        path.join(
            PATH_TO_PICKLE_BASE,
            "feasible_regions",
            f"{type(feasible_region).__name__}-dim_{dim}_constraints_{constraints}.pickle",
        ),
        "wb",
    ) as f:
        pickle.dump(feasible_region, f)
        print("Pickle object dumped.")
