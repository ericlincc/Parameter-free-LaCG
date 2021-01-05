# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 09:55:07 2021

@author: pccom
"""

import numpy as np
from numba import jit
import time

@jit(nopython = True)
def numba_project_onto_active_set(
    quadratic_coefficient,
    linear_vector,
    constant,
    active_set,
    barycentric,
    FW_criterion,
    tolerance,
    coefficient,
    reference_point,
):
    active_set_matrix = np.zeros((len(active_set), len(active_set[0])), active_set[0].dtype)
    for i in range(len(active_set)):
        active_set_matrix[i,:] = active_set[i]
    barycentric_coord = np.asarray(barycentric)
    
    quadratic = quadratic_coefficient*active_set_matrix.dot(active_set_matrix.T)
    linear = active_set_matrix.dot(linear_vector)
    x, polished_barycentric_coordinates, gap_values1 = accelerated_projected_gradient_descent_over_simplex(
        quadratic,
        linear,
        active_set_matrix,
        barycentric_coord,
        FW_criterion,
        tolerance,
        coefficient,
        reference_point,
        )
    return x, polished_barycentric_coordinates, gap_values1

@jit(nopython = True)
def projection_onto_simplex(x):
    (n,) = x.shape  # will raise ValueError if v is not 1-D
    if x.sum() == 1.0 and np.all(x >= 0):
        return x
    v = x - np.max(x)
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.count_nonzero(u * np.arange(1, n + 1) > (cssv - 1.0)) - 1
    theta = float(cssv[rho] - 1.0) / (rho + 1)
    w = np.minimum(1.0, np.maximum(v - theta, 0.0))
    # w = (v - theta).clip(min=0)
    return w

@jit(nopython = True)
def lp_oracle_simplex(x):
    v = np.zeros(x.shape, x.dtype)
    v[np.argmin(x)] = 1.0
    return v

@jit(nopython = True)
def gradient(Q, b, x):
    return b + Q.dot(x)
    
@jit(nopython = True)
def accelerated_projected_gradient_descent_over_simplex(
    quadratic,
    linear,
    active_set,
    alpha0,
    FW_criterion,
    tolerance,
    coefficient,
    reference_point,
):
    # Quantities we want to output.
    w = np.linalg.eigvalsh(quadratic)
    L = np.max(w)
    mu = np.min(w)
    initial_point = np.asarray(alpha0)
    x = initial_point
    x_old = initial_point
    y = initial_point
    q = mu / L
    
    if(mu < 1.0e-3):
        alpha = 0.0
        alpha_old = 0.0
    else:
        alpha = np.sqrt(q)
        alpha_old = 0.0
    
    grad = gradient(quadratic, linear, x)
    FWGap = grad.dot(x - lp_oracle_simplex(grad))
    it_count = 0
    gap_values = [FWGap]
    while (FW_criterion and FWGap > tolerance)  or (not FW_criterion and coefficient*np.linalg.norm(x - reference_point)**2 < FWGap):
        # print(FWGap)
        x_old = x.copy()
        x = projection_onto_simplex(y - 1 / L * gradient(quadratic, linear, y))
  
        if(mu < 1.0e-3):
            alpha_old = alpha
            alpha = 0.5 * (1 + np.sqrt(1 + 4 * alpha_old * alpha_old))
            beta = (alpha_old - 1.0) / alpha
        else:
            root = np.roots(np.asarray([1, alpha ** 2 - q, -alpha ** 2]))
            root = root[(root >= 0.0) & (root < 1.0)]
            assert len(root) != 0, "Root does not meet desired criteria.\n"
            alpha_old = alpha
            alpha = root[0]
            beta = alpha_old * (1 - alpha_old) / (alpha_old ** 2 - alpha)
            
        y = x + beta * (x - x_old)
        grad = gradient(quadratic, linear, x)
        FWGap = grad.dot(x - lp_oracle_simplex(grad))
        it_count += 1
        gap_values.append(FWGap)
    num_vert, dimension = active_set.shape
    w = np.zeros(active_set[0,:].shape, active_set[0,:].dtype)
    for i in range(num_vert):
        w += x[i] * active_set[i,:]
    return w, x, gap_values