# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 09:55:07 2021

@author: pccom
"""


# from numba import jit


from numba.pycc import CC
cc = CC('argmin_module')
import numpy as np

#If true will run FW gap, otherwise will run with gradient mapping criterion.
@cc.export('solve_argmin_aot', 'f8[:](f8[:,:], f8[:], f8, f8[:,:], f8[:], f8[:], b1, f8)')
def solve_argmin_aot(
    quadratic,
    linear,
    constant,
    active_set,
    initial_x,
    reference_x,
    tolerance_type,
    tolerance,
):
    def f_evaluate_grad(x, quadratic, linear, constant):
        return quadratic.dot(x) + linear
    
    def has_met_stopping_criterion(
        x, wolfe_gap, active_set, reference_x, tolerance_type, tolerance
    ):
        if tolerance_type:
            return wolfe_gap <= tolerance
        else:
            w = np.zeros(active_set.shape[1])
            for i in range(active_set.shape[0]):
                w += x[i] * active_set[i]
            return wolfe_gap <= tolerance * np.linalg.norm(w - reference_x) ** 2

    def simplex_projection(x):
        (n,) = x.shape  # will raise ValueError if v is not 1-D
        if x.sum() == 1.0 and np.all(x >= 0.0):
            return x
        v = x - np.max(x)
        u = np.sort(v)[::-1]
        cssv = np.cumsum(u)
        rho = np.count_nonzero(u * np.arange(1, n + 1) > (cssv - 1.0)) - 1
        theta = float(cssv[rho] - 1.0) / (rho + 1)
        w = np.minimum(1.0, np.maximum(v - theta, 0.0))
        return w

    def simplex_lp_oracle(d):
        v = np.zeros(d.shape)
        v[np.argmin(d)] = 1.0
        return v
    
    w = np.linalg.eigvalsh(quadratic)
    L = np.max(w)
    mu = np.min(w)
    q = mu / L
    if (mu < 1.0e-3) or q == 1:
        alpha = 0
    else:
        alpha = np.sqrt(q)

    x = initial_x
    y = initial_x

    grad = f_evaluate_grad(x, quadratic, linear, constant)
    wolfe_gap = grad.dot(x - simplex_lp_oracle(grad))


    while not has_met_stopping_criterion(
            x, wolfe_gap, active_set, reference_x, tolerance_type, tolerance
        ):
        x_ = x
        
        x = simplex_projection(
            y - 1 / L * f_evaluate_grad(y, quadratic, linear, constant)
        )
        if (mu < 1.0e-3) or q == 1:
            alpha_ = alpha
            alpha = 0.5 * (1 + np.sqrt(1 + 4 * alpha_ * alpha_))
            beta = (alpha_ - 1.0) / alpha
        else:
            root = np.roots(np.array([1, alpha ** 2 - q, -(alpha ** 2)]))
            root = root[(root >= 0.0) & (root < 1.0)]
            assert len(root) != 0, "Root does not meet desired criteria.\n"
            _alpha = alpha
            alpha = root[0]
            beta = _alpha * (1 - _alpha) / (_alpha ** 2 - alpha)
        y = x + beta * (x - x_)
        grad = f_evaluate_grad(x, quadratic, linear, constant)
        _wolfe_gap = wolfe_gap
        wolfe_gap = grad.dot(x - simplex_lp_oracle(grad))
        # # Detecting if subproblem is stuck
        # if abs(wolfe_gap - _wolfe_gap) <= 1e-6 and wolfe_gap < 1e-12:
        #     return np.zeros(len(initial_x))
    return x


@cc.export('solve_argmin_aot_dirty', 'f8[:](f8[:,:], f8[:], f8, f8[:,:], f8[:], f8[:], b1, f8)')
def solve_argmin_aot_dirty(
    quadratic,
    linear,
    constant,
    active_set,
    initial_x,
    reference_x,
    tolerance_type,
    tolerance,
):
    w = np.linalg.eigvalsh(quadratic)
    L = np.max(w)
    mu = np.min(w)
    q = mu / L
    if (mu < 1.0e-3) or q == 1:
        alpha = 0
    else:
        alpha = np.sqrt(q)

    x = initial_x
    y = initial_x

    grad = np.dot(quadratic, x) + linear
    v_FW = np.zeros(grad.shape)
    v_FW[np.argmin(grad)] = 1.0
    wolfe_gap = grad.dot(x - v_FW)


    if tolerance_type:
        if(wolfe_gap <= tolerance):
            stopping_flag = True
        else:
            stopping_flag = False
    else:
        w = np.zeros(active_set.shape[1])
        for i in range(active_set.shape[0]):
            w += x[i] * active_set[i]
        if(wolfe_gap <= tolerance * np.linalg.norm(w - reference_x) ** 2):
            stopping_flag =  True
        else:
            stopping_flag =  False 

    while not stopping_flag:
        x_ = x
        
        vect = y - 1 / L *(np.dot(quadratic,y) + linear)
        
        (n,) = vect.shape  # w
        v = vect - np.max(vect)
        u = np.sort(v)[::-1]
        cssv = np.cumsum(u)
        rho = np.count_nonzero(u * np.arange(1, n + 1) > (cssv - 1.0)) - 1
        theta = float(cssv[rho] - 1.0) / (rho + 1)
        x = np.minimum(1.0, np.maximum(v - theta, 0.0))
        
        # x = simplex_projection(
        #     y - 1 / L *(quadratic.dot(y) + linear)
        # )
        if (mu < 1.0e-3) or q == 1:
            alpha_ = alpha
            alpha = 0.5 * (1 + np.sqrt(1 + 4 * alpha_ * alpha_))
            beta = (alpha_ - 1.0) / alpha
        else:
            root = np.roots(np.array([1, alpha ** 2 - q, -(alpha ** 2)]))
            root = root[(root >= 0.0) & (root < 1.0)]
            assert len(root) != 0, "Root does not meet desired criteria.\n"
            _alpha = alpha
            alpha = root[0]
            beta = _alpha * (1 - _alpha) / (_alpha ** 2 - alpha)
        y = x + beta * (x - x_)
        grad = quadratic.dot(x) + linear
        
        
        _wolfe_gap = wolfe_gap
        v_FW = np.zeros(grad.shape)
        v_FW[np.argmin(grad)] = 1.0
        wolfe_gap = grad.dot(x - v_FW)
        # # Detecting if subproblem is stuck
        # if abs(wolfe_gap - _wolfe_gap) <= 1e-6 and wolfe_gap < 1e-12:
        #     return np.zeros(len(initial_x))
        
        if tolerance_type:
            if(wolfe_gap <= tolerance):
                stopping_flag = True
            else:
                stopping_flag = False
        else:
            w = np.zeros(active_set.shape[1])
            for i in range(active_set.shape[0]):
                w += x[i] * active_set[i]
            if(wolfe_gap <= tolerance * np.linalg.norm(w - reference_x) ** 2):
                stopping_flag =  True
            else:
                stopping_flag =  False 
    return x

if __name__ == "__main__":
    cc.compile()