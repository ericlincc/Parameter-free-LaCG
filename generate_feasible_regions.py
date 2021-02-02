import pickle
from os import path

import numpy as np
from scipy.sparse import csr_matrix

from pflacg.experiments.feasible_regions import *


PATH_TO_PICKLE_BASE = "/scratch/bzfcarde/pflacg_experiments/pickled_objects"



DIMENSIONS = [10000, 15000, 20000]


# Dense quadratic generation - uniform evalues
for dim in DIMENSIONS:
    feasible_region_list = []

    feasible_region_list.append(ProbabilitySimplexPolytope(dim))


    for feasible_region in feasible_region_list:
        with open(
            path.join(
                PATH_TO_PICKLE_BASE,
                "feasible_regions",
                f"{type(feasible_region).__name__}-_{dim}.pickle",
            ),
            "wb"
        ) as f:
            pickle.dump(feasible_region, f)
            print("Pickle object dumped.")


