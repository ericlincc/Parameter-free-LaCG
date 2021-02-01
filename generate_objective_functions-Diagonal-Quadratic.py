# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 08:35:40 2021

@author: pccom
"""
import pickle
from os import path

import numpy as np
from scipy.sparse import identity
from pflacg.experiments.objective_functions import *

PATH_TO_PICKLE_BASE = "/scratch/bzfcarde/pflacg_experiments/pickled_objects"

DIMENSIONS = [100*2, 200**2, 400**2, 600**2]
CONDITION_NUMS = [100,1000,10000]
SEEDS = [1,2,3]

# Sparse quadratic generation
for dim in DIMENSIONS:
    for condition_num in CONDITION_NUMS:
        for seed in SEEDS:
            np.random.seed(seed)
            b = np.random.rand(dim)
            objective_function = QuadraticDiagonal(dim, b, Mu=1.0, L=condition_num)
            with open(
                path.join(
                    PATH_TO_PICKLE_BASE,
                    "objective_functions",
                    f"{type(objective_function).__name__}"
                    f"-Uniform_evalues-dim_{dim}-cond_{condition_num}-seed{seed}.pickle",
                ),
                "wb"
            ) as f:
                pickle.dump(objective_function, f)
                print("Pickle object dumped.")


