# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 07:29:51 2021

@author: pccom
"""
import numpy as np
import timeit
from scipy.sparse import csr_matrix

dimension = 10000
mat_dim = int(np.sqrt(dimension))
num_elem_active_set = 1000

active_set = []
active_set_sparse = []
for i in range(num_elem_active_set):
    vect = np.zeros(dimension)
    vect[i] = 1.0
    active_set.append(vect)
    active_set_sparse.append(csr_matrix(vect))
    
    
new_vertex_sparse = csr_matrix(vect)
new_vertex = vect
    
def find_sparse_vertex_simplex(active_set,new_vertex):
    for i in range(len(active_set)):
        if active_set[i].dot(new_vertex.T)[0,0] > 0.5:
            barycentric = np.zeros(len(active_set))
            barycentric[i] = 1.0
            return 
    barycentric = np.zeros(len(active_set) + 1)
    barycentric[-1] = 1.0
    new_list = list(active_set)
    new_list.append(new_vertex)
    return

def find_vertex(active_set,new_vertex):
    for i in range(len(active_set)):
        if np.allclose(active_set[i], new_vertex ):
            barycentric = np.zeros(len(active_set))
            barycentric[i] = 1.0
            return 
    barycentric = np.zeros(len(active_set) + 1)
    barycentric[-1] = 1.0
    new_list = list(active_set)
    new_list.append(new_vertex)
    return


print("\nUsing timeit")
t = timeit.Timer(lambda: find_sparse_vertex_simplex(active_set_sparse, new_vertex_sparse))
print ("Sparse --- %s seconds ---" %t.timeit(number=10))
t = timeit.Timer(lambda: find_vertex(active_set, new_vertex))
print ("Normal --- %s seconds ---" %t.timeit(number=10))


dimension = int(500*500)
mat_dim = int(np.sqrt(dimension))
num_elem_active_set = 1000

def lp_oracle(d):
    from scipy.optimize import linear_sum_assignment
    objective = d.reshape((mat_dim, mat_dim))
    matching = linear_sum_assignment(objective)
    solution = np.zeros((mat_dim, mat_dim))
    solution[matching] = 1
    return solution.reshape(dimension)

active_set = []
active_set_sparse = []
for i in range(num_elem_active_set):
    vect = lp_oracle(np.random.rand(dimension))
    active_set.append(vect)
    active_set_sparse.append(csr_matrix(vect))
    
new_vertex_sparse = csr_matrix(vect)
new_vertex = vect
    
def find_sparse_vertex_Birkhoff(active_set,new_vertex):
    for i in range(len(active_set)):
        if active_set[i].dot(new_vertex.T)[0,0] > mat_dim - 0.1:
            barycentric = np.zeros(len(active_set))
            barycentric[i] = 1.0
            return 
    barycentric = np.zeros(len(active_set) + 1)
    barycentric[-1] = 1.0
    new_list = list(active_set)
    new_list.append(new_vertex)
    return

print("\nUsing timeit")
t = timeit.Timer(lambda: find_sparse_vertex_Birkhoff(active_set_sparse, new_vertex_sparse))
print ("Sparse --- %s seconds ---" %t.timeit(number=10))
t = timeit.Timer(lambda: find_vertex(active_set, new_vertex))
print ("Normal --- %s seconds ---" %t.timeit(number=10))