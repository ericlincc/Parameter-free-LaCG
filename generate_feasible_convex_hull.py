import pickle
from os import path

import numpy as np
from scipy.sparse import csr_matrix

from pflacg.experiments.feasible_regions import *


PATH_TO_PICKLE_BASE = "/scratch/share/pflacg_experiments/pickled_objects"

DIMENSIONS = [100, 500, 1000]
NUM_VERTICES = [100, 500, 1000]
SEEDS = [1, 2, 3]
l1_regularization = 1.0

# Dense quadratic generation - uniform evalues
for dim in DIMENSIONS:
    for number_of_vertex in NUM_VERTICES:
        for seed in SEEDS:
            np.random.rand(seed)
            set_of_vertices = []
            for i in range(number_of_vertex):
                vect = 2.0 * (np.random.rand(dim) - 0.5)
                vect = vect / np.linalg.norm(vect)
                set_of_vertices.append(vect)

            convex_hull = ConvexHull(
                set_of_vertices,
            )

            with open(
                path.join(
                    PATH_TO_PICKLE_BASE,
                    "feasible_regions",
                    f"{type(feasible_region).__name__}-dim_{dim}_numvertex_{number_of_vertex}_seed_{seed}.pickle",
                ),
                "wb",
            ) as f:
                pickle.dump(convex_hull, f)
                print("Pickle object dumped.")
