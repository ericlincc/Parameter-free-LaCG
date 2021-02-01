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
from scipy.sparse.linalg import eigsh

import scipy.stats as stats
import scipy.sparse as sparse
dimension = 10000
def sprandsym(n, density):
    rvs = stats.norm().rvs
    X = sparse.random(n, n, density=density, data_rvs=rvs)
    upper_X = sparse.triu(X)
    result = upper_X + upper_X.T - sparse.diags(X.diagonal())
    return result

def sparse_compute_largest_and_smallest_eigenvalue(matrix):
    L = eigsh(matrix, k=1, which = 'LA', return_eigenvectors = False)[0]
    mu = eigsh(matrix, k=1, which = 'SA', return_eigenvectors = False)[0]
    return mu, L

PATH_TO_PICKLE_BASE = "/scratch/bzfcarde/pflacg_experiments/pickled_objects"

DIMENSIONS = [2500, 6400]
DENSITY = [0.1, 0.01, 0.001]
CONDITION_NUMS = [1e3, 1e4, 1e5]

# Sparse quadratic generation
for dim in DIMENSIONS:
    for condition_num in CONDITION_NUMS:
        for density in DENSITY:
            matrix = sprandsym(dim, density)
            mu, L = sparse_compute_largest_and_smallest_eigenvalue(matrix)
            new_mat = matrix*(condition_num - 1.0)/(L-mu)
            new_mu, new_L = sparse_compute_largest_and_smallest_eigenvalue(new_mat)
            final_mat = new_mat + (1 - new_mu)*identity(dim)
            b = np.random.rand(dim)
            objective_function = QuadraticSparse(dim, matrix, b)
            final_mu, final_L = sparse_compute_largest_and_smallest_eigenvalue(final_mat)
            final_cond_number = int(final_L/final_mu) 

            with open(
                path.join(
                    PATH_TO_PICKLE_BASE,
                    "objective_functions",
                    f"{type(objective_function).__name__}"
                    f"-Uniform_evalues-dim_{dim}-cond_{final_cond_number}-density_{density}.pickle",
                ),
                "wb"
            ) as f:
                pickle.dump(objective_function, f)
                print("Pickle object dumped.")


