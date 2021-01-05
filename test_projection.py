# -*- coding: utf-8 -*-
"""
Created on Sun Jan  3 16:05:54 2021

@author: pccom
"""
from pflacg.algorithms.numba_projection import numba_project_onto_active_set
from pflacg.algorithms._algorithms_utils import project_onto_active_set, stopping_criterion
import numpy as np
from numba import jit



@jit
def NbFunc(Num_vert, dimension):
    dimension = dimension
    num_vert = Num_vert
    active_set = []
    barycentric_coordinates = []
    for i in range(num_vert):
        vect = np.random.rand(dimension)
        active_set.append(vect / np.linalg.norm(vect))
        barycentric_coordinates.append(1.0/num_vert)
        
    linear_vector = np.random.rand(dimension) 
    reference_vector = np.random.rand(dimension)    
    
    active_set_matrix = np.zeros((num_vert, dimension), vect.dtype)
    for i in range(num_vert):
        active_set_matrix[i,:] = active_set[i]
    
    barycentric_coord = np.asarray(barycentric_coordinates)
    
    numba_project_onto_active_set(1.0, 
                                            linear_vector, 
                                            1.0, 
                                            active_set_matrix, 
                                            barycentric_coord, 
                                            True,
                                            1.0e-4,
                                            1.0,
                                            reference_vector,
                                            )
    return
    

def Func(Num_vert, dimension):
    dimension = dimension
    num_vert = Num_vert
    active_set = []
    barycentric_coordinates = []
    for i in range(num_vert):
        vect = np.random.rand(dimension)
        active_set.append(vect / np.linalg.norm(vect))
        barycentric_coordinates.append(1.0/num_vert)
        
        
    linear_vector = np.random.rand(dimension) 
    stop_crit = stopping_criterion(tolerance = 1.0e-4)
    
    
    project_onto_active_set(1.0, 
                            linear_vector, 
                            1.0, 
                            active_set, 
                            barycentric_coordinates, 
                            stop_crit,
                            )
    return
    



dimension = 100
num_vert = 200
active_set = []
barycentric_coordinates = []
for i in range(num_vert):
    vect = np.random.rand(dimension)
    active_set.append(vect / np.linalg.norm(vect))
    barycentric_coordinates.append(1.0/num_vert)
    
linear_vector = np.random.rand(dimension) 
reference_vector = np.random.rand(dimension)    

active_set_matrix = np.vstack(active_set)
barycentric_coord = np.asarray(barycentric_coordinates)

results = numba_project_onto_active_set(1.0, 
                                        linear_vector, 
                                        1.0, 
                                        active_set_matrix, 
                                        barycentric_coord, 
                                        True,
                                        1.0e-4,
                                        1.0,
                                        reference_vector,
                                        )

dimension = 100
num_vert = 200
import timeit

print("\nUsing timeit")
t = timeit.Timer(lambda: Func(num_vert, dimension))
print ("Normal --- %s seconds ---" %t.timeit(number=1000))
t = timeit.Timer(lambda: NbFunc(num_vert, dimension))
print ("Numba --- %s seconds ---" %t.timeit(number=1000))