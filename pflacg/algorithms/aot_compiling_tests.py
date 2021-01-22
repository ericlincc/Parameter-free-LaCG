# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 17:38:53 2021

@author: pccom
"""
import argmin_module
import time

from project_onto_active_set_jit import accelerated_projected_gradient_descent_over_simplex_jit

def solve_argmin_aot_backup(
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

class ProbabilitySimplexPolytope:
    def __init__(self, dim):
        self.dim = dim

    def initial_point(self):
        v = np.zeros(self.dim)
        v[0] = 1.0
        return v

    def initial_active_set(self):
        return [self.initial_point()]

    def lp_oracle(self, x):
        v = np.zeros(len(x), dtype=float)
        v[np.argmin(x)] = 1.0
        return v

    def projection(self, x):
        (n,) = x.shape  # will raise ValueError if v is not 1-D
        if x.sum() == 1.0 and np.alltrue(x >= 0):
            return x
        v = x - np.max(x)
        u = np.sort(v)[::-1]
        cssv = np.cumsum(u)
        rho = np.count_nonzero(u * np.arange(1, n + 1) > (cssv - 1.0)) - 1
        theta = float(cssv[rho] - 1.0) / (rho + 1)
        w = (v - theta).clip(min=0)
        return w


class Quadratic:
    """Given matrix M and vector b, f(x) = 0.5 x^T M * x + b^T x."""

    def __init__(self, dim, M, b):
        self._dim = dim
        self.M = M
        self.b = b
        u,v = np.linalg.eig(M)
        self.L= np.max(u)
        self.Mu  = np.min(u)

    @property
    def dim(self):
        return self._dim

    def smallest_eigenvalue(self):
        return self.Mu

    def largest_eigenvalue(self):
        return self.L

    def line_search(self, grad, d, x):
        return -np.dot(grad, d) / np.dot(d, self.M.dot(d))

    def evaluate(self, x):
        return 0.5 * np.dot(x, self.M.dot(x)) + np.dot(self.b, x)

    def evaluate_grad(self, x):
        return self.M.dot(x) + self.b
    
def accelerated_projected_gradient_descent(
    f,
    feasible_region,
    active_set,
    tolerance,
    initial_x,
    max_iteration=100,
):
    # TODO: The below description is not to date.
    """
    Run Nesterov's accelerated projected gradient descent.

    References
    ----------
    Nesterov, Y. (2018). Lectures on convex optimization (Vol. 137).
    Berlin, Germany: Springer. (Constant scheme II, Page 93)

    Parameters
    ----------
    x0 : numpy array.
        Initial point.
    function: function being minimized
        Function that we will minimize. Gradients are computed through a
        function.grad(x) function that returns the gradient at x as a
        numpy array.
    feasible_region : feasible region function.
        Returns projection oracle of a point x onto the feasible region,
        which are computed through the function feasible_region.project(x).
        Additionally, a LMO is used to compute the Frank-Wolfe gap (used as a
        stopping criterion) through the function
        feasible_region.linear_optimization_oracle(grad) function, which
        minimizes <x, grad> over the feasible region.
    tolerance : float
        Frank-Wolfe accuracy to which the solution is outputted.

    Returns
    -------
    x : numpy array
        Outputted solution with primal gap below the target tolerance
    """

    # Quantities we want to output.
    L = f.largest_eigenvalue()
    mu = f.smallest_eigenvalue()
    q = mu / L
    x = initial_x
    y = initial_x
    if (mu < 1.0e-3) or q == 1:  # TODO: Alex check later
        alpha = 0
    else:
        alpha = np.sqrt(q)
    grad = f.evaluate_grad(x)
    wolfe_gap = grad.dot(x - feasible_region.lp_oracle(grad))
    it_count = 0

    while wolfe_gap > tolerance:
        x_ = x
        x = feasible_region.projection(y - 1 / L * f.evaluate_grad(y))
        if (mu < 1.0e-3) or q == 1:  # TODO: Alex check later
            alpha_ = alpha
            alpha = 0.5 * (1 + np.sqrt(1 + 4 * alpha_ * alpha_))
            beta = (alpha_ - 1.0) / alpha
        else:
            root = np.roots([1, alpha ** 2 - q, -(alpha ** 2)])
            root = root[(root >= 0.0) & (root < 1.0)]
            assert len(root) != 0, "Root does not meet desired criteria.\n"
            _alpha = alpha
            alpha = root[0]
            beta = _alpha * (1 - _alpha) / (_alpha ** 2 - alpha)
        y = x + beta * (x - x_)
        grad = f.evaluate_grad(x)
        _wolfe_gap = wolfe_gap
        wolfe_gap = grad.dot(x - feasible_region.lp_oracle(grad))
        it_count += 1
        # LOGGER.info(f"wolfe_gap = {wolfe_gap}")
        if np.isclose(wolfe_gap, _wolfe_gap) and wolfe_gap < 1e-13:
            raise Exception("projection step is getting stuck")
    w = np.zeros(len(active_set[0]))
    for i in range(len(active_set)):
        w += x[i] * active_set[i]
    return w, x.tolist()

import numpy as np
dimension = 10
quadratic_mat = np.random.rand(dimension, dimension)
matrix = quadratic_mat.T.dot(quadratic_mat)
linear =  np.random.rand(dimension)

active_set = []
for i in range(dimension):
    vect = np.zeros(dimension)
    vect[i] = 1.0
    active_set.append(vect)
active_set = np.vstack(active_set)

initial_x = np.ones(dimension)/dimension
reference_x = vect
tolerance_type = True
tolerance = 1.0e-3

#Running clean AOT
solution_aot = argmin_module.solve_argmin_aot(matrix, linear, 0.0, active_set, initial_x, reference_x, True, tolerance)
#Running dirty AOT
solution_aot_dirty = argmin_module.solve_argmin_aot_dirty(matrix, linear, 0.0, active_set, initial_x, reference_x, tolerance_type, tolerance)
#Running backup file
solution = solve_argmin_aot_backup(matrix, linear, 0.0, active_set, initial_x, reference_x, tolerance_type, tolerance)
#Running backup JIT
solution_jit = accelerated_projected_gradient_descent_over_simplex_jit(matrix, linear, 0.0, active_set, initial_x, reference_x, "dual gap", tolerance)
#Running directly accelerated gradient descent
fun = Quadratic(dimension, matrix, linear)
feasible_region = ProbabilitySimplexPolytope(dimension)
reference_solution, barycentric_coordinates =  accelerated_projected_gradient_descent( fun, feasible_region, active_set,tolerance, initial_x, max_iteration=100)

print("Check that all the algorithms converge approximately to the same solution")
print("Difference between direct accelerated gradient descent and aot: ", np.linalg.norm(solution_aot - barycentric_coordinates))
print("Difference between backup and aot: ", np.linalg.norm(solution_aot - solution))
print("Difference between direct accelerated gradient descent and aot dirty: ", np.linalg.norm(solution_aot_dirty - barycentric_coordinates))
print("Difference between backup and aot dirty: ", np.linalg.norm(solution_aot_dirty - solution))
print("Difference between direct accelerated gradient descent and jit: ", np.linalg.norm(solution_jit - barycentric_coordinates))
print("Difference between backup and jit: ", np.linalg.norm(solution_jit - solution))

def Func1(dimension):
    quadratic_mat = np.random.rand(dimension, dimension)
    matrix = quadratic_mat.T.dot(quadratic_mat)
    linear =  np.random.rand(dimension)
    
    num_elem_active_set = dimension
    active_set = []
    for i in range(num_elem_active_set):
        vect = np.zeros(dimension)
        vect[i] = 1.0
        active_set.append(vect)
    active_set = np.vstack(active_set)
    
    initial_x = np.ones(dimension)/dimension
    reference_x = vect
    tolerance_type = True
    tolerance = 1.0e-4
    argmin_module.solve_argmin_aot(matrix, linear, 0.0, active_set, initial_x, reference_x, tolerance_type, tolerance)
    return

def Func2(dimension):
    quadratic_mat = np.random.rand(dimension, dimension)
    matrix = quadratic_mat.T.dot(quadratic_mat)
    linear =  np.random.rand(dimension)
    
    num_elem_active_set = dimension
    active_set = []
    for i in range(num_elem_active_set):
        vect = np.zeros(dimension)
        vect[i] = 1.0
        active_set.append(vect)
    active_set = np.vstack(active_set)
    
    initial_x = np.ones(dimension)/dimension
    reference_x = vect
    tolerance_type = True
    tolerance = 1.0e-3
    argmin_module.solve_argmin_aot_dirty(matrix, linear, 0.0, active_set, initial_x, reference_x, tolerance_type, tolerance)
    return

def Func3(dimension):
    quadratic_mat = np.random.rand(dimension, dimension)
    matrix = quadratic_mat.T.dot(quadratic_mat)
    linear =  np.random.rand(dimension)
    
    num_elem_active_set = dimension
    active_set = []
    for i in range(num_elem_active_set):
        vect = np.zeros(dimension)
        vect[i] = 1.0
        active_set.append(vect)
    active_set = np.vstack(active_set)
    
    initial_x = np.ones(dimension)/dimension
    reference_x = vect
    tolerance_type = True
    tolerance = 1.0e-3
    solve_argmin_aot_backup(matrix, linear, 0.0, active_set, initial_x, reference_x, tolerance_type, tolerance)
    return

def Func4(dimension):
    quadratic_mat = np.random.rand(dimension, dimension)
    matrix = quadratic_mat.T.dot(quadratic_mat)
    linear =  np.random.rand(dimension)
    
    num_elem_active_set = dimension
    active_set = []
    for i in range(num_elem_active_set):
        vect = np.zeros(dimension)
        vect[i] = 1.0
        active_set.append(vect)
    active_set = np.vstack(active_set)
    
    initial_x = np.ones(dimension)/dimension
    reference_x = vect
    tolerance = 1.0e-3
    accelerated_projected_gradient_descent_over_simplex_jit(matrix, linear, 0.0, active_set, initial_x, reference_x, "dual gap", tolerance)
    return

dimension = 400
import timeit

print("\nTimeit tests")
print("Using timeit")
t = timeit.Timer(lambda: Func1(dimension))
print ("Numba normal AOT --- %s seconds ---" %t.timeit(number=20))
t = timeit.Timer(lambda: Func2(dimension))
print ("Numba dirty AOT --- %s seconds ---" %t.timeit(number=20))
t = timeit.Timer(lambda: Func3(dimension))
print ("Python --- %s seconds ---" %t.timeit(number=20))
t = timeit.Timer(lambda: Func4(dimension))
print ("Numba JIT --- %s seconds ---" %t.timeit(number=20))

