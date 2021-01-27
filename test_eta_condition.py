# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 07:51:08 2021

@author: pccom
"""

import numpy as np
dimension = 500
matrix = np.random.rand(dimension, dimension)
M = matrix.T.dot(matrix)
M = M + 500 * np.identity(dimension)
b = np.random.rand(dimension)

def f(x):
    return 0.5* x.dot(M.dot(x)) + b.dot(x)

def grad(x):
    return M.dot(x)+ b

def check_eta_condition(x, y):
    return (f(x) - f(y) - grad(y).dot(x - y))#/np.dot(x - y, x - y)

def check_eta_condition_Hessian(x, y):
    diff_norm = (x - y) #/np.linalg.norm(x - y)
    return 0.5*diff_norm.dot(M.dot(diff_norm))

vec1 = np.random.rand(dimension)
vec2 = vec1 + 1.0e-9*np.random.rand(dimension)


cond1 = check_eta_condition(vec1, vec2)
cond2 = check_eta_condition_Hessian(vec1, vec2)
print(cond1)
print(cond2)
print(cond1 / cond2)