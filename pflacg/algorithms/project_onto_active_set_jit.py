# coding=utf-8
"""Separate module for speedup using numba just-in-time compilation."""


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
    """
    TODO: Add a description of the algorithm and its reference.

    Parameters
    ----------
    quadratic: np.ndarray
        The matrix of the quadratic form.
    linear: np.ndarray
        The linear vector of the quadratic form.
    constant: float
        The constant term of the quadratic form.
    active_set: np.ndarray
        The stacked array of the active vertices.
    initial_x: np.ndarray
        The initial barycentric coordinates w.r.t. to the active set.
    reference_x: np.ndarray
        Reference point for computing the gradient mapping if
        tolerance_type="gradient mapping".
    tolerance_type: string
        Either "dual gap" or "gradient mapping".
    tolerance: float
        The accuracy to which we need to solve this subproblem to.

    Returns
    -------
    np.ndarray
        The barycentric coordinates w.r.t. to the active set. Zeros if it is detected
        that no more progress can be made.
    """

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
        n = x.shape[0]
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

    num_elem_moving_average = 20
    iteration = 0
    previous_wolfe_gaps = np.zeros(num_elem_moving_average)
    previous_wolfe_gaps[0] = wolfe_gap
    moving_average = wolfe_gap

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

        iteration += 1
        moving_average_ = moving_average
        if iteration < num_elem_moving_average:
            previous_wolfe_gaps[iteration] = wolfe_gap
            moving_average = np.sum(previous_wolfe_gaps) / iteration
        else:
            previous_wolfe_gaps[iteration % num_elem_moving_average] = wolfe_gap
            moving_average = np.mean(previous_wolfe_gaps)

        # Detecting if subproblem progress is stuck due to numpy computation inaccuracies.
        if abs(moving_average - moving_average_) <= 1e-6 and moving_average < 1e-7:
            return x  # np.zeros(len(initial_x))
    return x
