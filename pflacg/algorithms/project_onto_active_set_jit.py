# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 09:55:07 2021

@author: pccom
"""


from numba import jit
import numpy as np


@jit(nopython=True, cache=True)
def accelerated_projected_gradient_descent_over_simplex_jit(
    quadratic,
    linear,
    constant,
    active_set,
    initial_x,
    reference_x,
    tolerance_type,
    tolerance,
):
    def f_evaluate(x, quadratic, linear, constant):
        """f (x) = 0.5 x^T Q x + b^T x + c"""
        return 0.5 * x.T.dot(quadratic.dot(x)) + linear.dot(x) + constant

    def f_evaluate_grad(x, quadratic, linear, constant):
        return quadratic.dot(x) + linear

    def has_met_stopping_criterion(
        x, wolfe_gap, active_set, reference_x, tolerance_type, tolerance
    ):
        if tolerance_type == "dual gap":
            return wolfe_gap <= tolerance
        elif tolerance_type == "gradient mapping":
            w = np.zeros(active_set.shape[1])
            for i in range(active_set.shape[0]):
                w += x[i] * active_set[i]
            return wolfe_gap <= tolerance * np.linalg.norm(w - reference_x) ** 2
        else:
            return True

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

        # Detecting if subproblem is stuck
        if abs(wolfe_gap - _wolfe_gap) <= 1e-6 and wolfe_gap < 1e-12:
            return np.zeros(len(initial_x))
    return x
