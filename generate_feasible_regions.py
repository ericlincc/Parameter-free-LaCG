import pickle
from os import path

import numpy as np
from scipy.sparse import csr_matrix

from pflacg.experiments.feasible_regions import *


PATH_TO_PICKLE_BASE = "/scratch/bzfcarde/pflacg_experiments/pickled_objects"



DIMENSIONS = [100*2, 200**2, 400**2, 600**2]


# Dense quadratic generation - uniform evalues
for dim in DIMENSIONS:
    feasible_region_list = []

    feasible_region_list.append(ProbabilitySimplexPolytope(dim))
    feasible_region_list.append(BirkhoffPolytope(dim))
    feasible_region_list.append(L1UnitBallPolytope(dim))


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


