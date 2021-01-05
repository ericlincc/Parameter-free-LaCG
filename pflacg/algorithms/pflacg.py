# codeing=utf-8
"""Contains code for parameter-free locally accelerated conditional gradient."""

from copy import deepcopy
import logging
import time
from multiprocessing import shared_memory, Value, Process, Lock
import numpy as np

from pflacg.experiments.objective_function import RegularizedObjectiveFunction
from pflacg.algorithms._abstract_algorithm import _AbstractAlgorithm
from pflacg.algorithms._algorithms_utils import *


logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s :: %(asctime)s :: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOGGER = logging.getLogger()


GM_COEFF = 1.0 / 32.0


# TODO: maybe switch to using Point instead of x.
def compute_strong_FW_gap(x, active_set, objective_function, feasible_region):
    grad = objective_function.evaluate_grad(x)
    v = feasible_region.lp_oracle(grad)
    a, indexMax = feasible_region.away_oracle(grad, active_set)
    strong_FW_gap = np.dot(grad, a - v)
    return strong_FW_gap


def active_set_is_subset_of(active_set_1, active_set_2):
    id_set_2 = set([id(vertex) for vertex in active_set_2])
    for vertex in active_set_1:
        if id(vertex) not in id_set_2:
            return False
    return True


class ParameterFreeLaCG(_AbstractAlgorithm):
    def __init__(self,
        fw_variant="AFW",
        ratio=0.5,
        async=False,
    ):
        self.fw_variant = fw_variant
        self.ratio = 0.5
        self.async = async

    def run(
        self,
        objective_function,
        feasible_region,
        exit_criterion,
        initial_point,
        initial_active_set,
        initial_eta = 1.0,
    ):

        FAFW = FractionalAwayStepFW(fw_variant=self.fw_variant, ratio=self.ratio)
        wACC = ParameterFreeAGD(ratio=self.ratio)

        # Initialization
        strong_FW_gap_out = compute_strong_FW_gap(initial_point, initial_active_set, objective_function, feasible_region)
        iteration = 0
        duration = 0.0
        f_val = objective_function.evaluate(initial_point)
        run_status = (
            iteration,
            duration,
            f_val,
            strong_FW_gap_out,
        )
        run_history = [run_status]

        x_FAFW = initial_point
        active_set_FAFW = initial_active_set

        # Create a new shared memory block for wACC (and its lock)
        # shared_ref looks like [returned parameters, yh]
        dummy = np.zeros(shape=(objective_function.dim), dtype=np.float64)
        shared_ret_x_buffer = shared_memory.SharedMemory(create=Trye, size=dummy.nbytes)
        shared_ret_x = np.ndarray(a.shape, dtype=np.float64, buffer=shared_ret_x_buffer.buf)
        global_eta = Value("d", initial_eta)
        gloabl_sigma = Value("d", initial_eta)
        wACC_output_lock = Lock()

        if not self.aysnc:
            global_iter_count = Value("i", 0)
            global_iter_count_lock = Lock()


        num_halvings = 0
        while not exit_criterion.has_met_exit_criterion(run_status):
            # Set halving strong Wolfe gap
            target_accuracy = strong_FW_gap_FAFW * self.ratio

            if wACC_restart_flag:

                dummy_lambdas = np.zero
                active_set_wACC = active_set_FAFW

                dummy = np.zeros(shape=(len(active_set_FAFW)), dtype=np.float64)
                shread_ret_lambdas_buffer = shared_memory.SharedMemory(create=Trye, size=dummy.nbytes)
                shared_ret_lambdas = np.ndarray(a.shape, dtype=np.float64, buffer=shread_ret_lambdas_buffer.buf)

                shared_memory_names = [shared_ret_y_buffer.name, shared_ret_lambdas.name, shared_ref]
                wACC_process = Process(
                    wACC.run,
                    args=(
                        None
                    ),
                )
            else:
                pass
                # resume previous wACC process

            # Run FAFW and wait for the output
            x_FAFW, active_set_FAFW, lambdas_FAFW = FAFW.run(
                objective_function,
                feasible_region,
                x_FAFW,  # TODO: Change it to Point?
                active_set_FAFW,
                lambdas_FAFW,
                global_iter_count=global_iter_count,  # TODO: Need to include a global count
            )

            # TODO: Add a wait until iter_ACC > iter_global

            # retrieve the most recent output
            wACC_output_lock.acquire()
            x_wACC = np.copy(shared_ret_x)
            lambdas_wACC = np.copy(shared_ret_lambdas)
            wACC_output_lock.release()


            # Compute Strong Wolfe gap (or dual gap)
            strong_FW_gap_wACC = compute_strong_FW_gap(x_wACC, active_set_wACC, objective_function, feasible_region)
            strong_FW_gap_FAFW = compute_strong_FW_gap(x_FAFW, active_set_FAFW, objective_function, feasible_region)

            if strong_FW_gap_wACC > strong_FW_gap_FAFW or not active_set_is_subset_of(active_set_wACC, active_set_FAFW): 
                # Terminate wACC process and

                wACC_process.terminate()
                wACC_process.join()
                shread_ret_lambdas_buffer.close()
                shread_ret_lambdas_buffer.unlink()
                wACC_restart_flag = True

                # Set output points
                x_out = x_FAFW
                strong_FW_gap_out = strong_FW_gap_FAFW
            else:
                # Allow wACC to continue its execution
                wACC_restart_flag = False

                # Set output points
                x_out = x_wACC
                strong_FW_gap_out = strong_FW_gap_wACC

            # Append output points
            iteration = global_iter_count.value
            duration = time.time() - start_time
            f_val = objective_function.evaluate(x_out)
            run_status = (
                iteration,
                duration,
                f_val,
                strong_FW_gap_out,
            )
            LOGGER.info(
                "Running " + str(self.fw_variant) + ": "
                "iteration = {1}, duration = {2:.{0}f}, f_val = {3:.{0}f}, dual_gap = {4:.{0}f}".format(
                    DISPLAY_DECIMALS, *run_status
                )
            )
            run_history.append(run_status)

        shared_ret_x_buffer.close()
        shared_ret_x_buffer.unlink()

        return run_history





class ParameterFreeAGD(_AbstractAlgorithm):
    def __init__(self, ratio=0.5, **kwargs):
        self.ratio = ratio

    def compute_PGD_step(x, eta, objective_function, feasible_region):
        # TODO
        pass

    def run(
        self,
        objective_function,
        feasible_region,
        initial_point,
        initial_eta,
        epsilon,
        shared_ret_buffers_dict=None,
        global_iter_count=global_iter_count,
        **kwargs
    ):
        """Run PF-ACC given an initial point and an active set/feasible region.

        Returns
        -------
        
        """
        

        # Initial shared buffers based on shared_ref_buffers_dict

        # Initializtion
        x = initial_point
        eta = global_eta.value
        sigma = global_sigma.value
        x_plus = compute_PGD_step(x, eta, objective_function, feasible_region)
        norm_grad_mapping = np.linalg.norm(eta * (x_plus - x))

        while np.isclose(norm_grad_mapping, epsilon) or norm_grad_mapping < epsilon:
            epsilon_r = 0.5 * norm_grad_mapping
            sigma_flag = False
            while not sigma_flag:
                _x, eta = self.ACC(objective_function, feasible_region, sigma, x, eta, 0.3 * epsilon_r)

                if sigma * np.linalg.norm(_x - x) <= 3 / 8 * epsilon_r:
                    sigma_flag = True
                    x = _x
                else:
                    sigma = sigma * 2.0  # TODO: need to update global_eta

            x_plus = compute_PGD_step(x, eta, objective_function, feasible_region)
            norm_grad_mapping = np.linalg.norm(eta * (x_plus - x))

            # TODO: Update shared buffers

        return None  # TODO








    def ACC(
        self,
        objective_function,
        feasible_region,
        sigma,
        initial_point,
        initial_eta,
        epsilon,
        max_iteration=1e5,
    ):
        """Inner ACC with exact projection."""

        # Initialization and compute eta_0
        reg_objective_function = RegularizedObjectiveFunction(
            objective_function=objective_function,
            sigma=sigma,
            reference_point=initial_point.cartesian_coordinates,
        )
        point_x = initial_point
        a = 1
        A = 1
        eta_0 = initial_eta
        epsilon_M = a * epsilon / 8.0

        eta_flag = False
        while not eta_flag:
            eta_sigma = eta_0 + sigma
            point_v = project_onto_active_set(
                quadratic_coefficient=(sigma * A + eta) / 2.0,
                linear_vector=z,
                active_set=feasible_region.vertices,
                barycentric_coordinates=point_x.barycentric_coordinates,
                tolerance=epsilon_M,
                tolerance_type="dual gap",
            )
            point_yh = point_v
            if self._check_eta_condition_1(
                reg_objective_function,
                point_x.cartesian_coordinates,
                point_v.cartesian_coordinates,
                eta_sigma
            ):
                eta_flag = True
            else:
                eta_flag = False
                eta_0 = eta_0 * 2.0


        j = 1
        eta = eta_0
        while (j < max_iteration):
            eta_flag = False
            eta_sigma = eta + sigma

            _a = self.compute_a(A_=A, theta_max=np.sqrt(sigma / (2 * eta_sigma)))
            _A = A + _a

            _z, _point_v, _point_yh, _point_y = self.AGD_step(
                reg_objective_function,
                feasible_region,
                point_y,
                point_v,
                z,
                _a,
                _A,
                eta_sigma,
                sigma,
                epsilon_l,
                epsilon_M,
                eta_0,
            )

            if self._check_eta_condition_1(
                reg_objective_function,
                point_x.cartesian_coordinates,
                point_yh.cartesian_coordinates,
                eta_sigma,
            ) and self._check_eta_condition_1(
                reg_objective_function,
                point_yh.cartesian_coordinates,
                point_y.cartesian_coordinates,
                eta_sigma,
            ):
                eta_flag = True
            else:
                eta_flag = False

            if eta_flag:
                A = _A
                point_v = _point_v
                point_yh = _point_yh
                point_y = _point_y
                z = _z
                G_sigma = (eta + sigma) * (point_yh.cartesian_coordinates - point_y.cartesian_coordinates)

                if np.linalg.norm(G_sigma) <= epsilon:
                    return point_yh, eta
            else:
                eta = eta * 2.0
                continue

        return None


    def AGD_step(
            self,
            objective_function,
            feasible_region,
            point_y_,
            point_v_,
            z_,  # A numpy array
            a,
            A,
            eta,
            sigma,
            epsilon_l,
            epsilon_M,
            eta_0,
        ):
        """Compute x_k, y_k and y_k^+ according to a smoothness estimate eta.
        Parameters
        ----------
        point_y_: Point
        point_v_: Point
        z_: np.ndarray
        a: float
        theta: float
            Define to be a_k/A_k in PF-LaCG.
        eta: float
            Current estimate of smoothness.
        sigma: float
            Regularization parameter for objective function f.
        epsilon_l: float
            Accuracy to evaluate l(u)
        epsilon_M: float
            Accuracy to evaluate m(u)
        eta_0: float
            Initial estimate of smoothness

        Returns
        -------
        z: np.ndarray
        point_x: Point
        point_v: Point
        point_yh: Point
        point_y: Point
        """

        theta = a / A
        point_x = (1 / (1 + theta)) * point_y_ + (theta / (1 + theta)) * point_v_

        z = z_ - a * objective_function.evaluate_grad(point_x.cartesian_coordinates) + sigma * a * point_x.cartesian_coordinates
        point_v = project_onto_active_set(
            quadratic_coefficient=(sigma * A + eta_0) / 2.0,
            linear_vector=z,
            active_set=feasible_region.vertices,
            barycentric_coordinates=point_x.barycentric_coordinates,
            tolerance=epsilon_M,
            tolerance_type="dual gap",
        )
        point_yh = (1 - theta) * point_y_ + theta * point_v
        point_y = project_onto_active_set(
            quadratic_coefficient=(eta + sigma) / 2.0,
            linear_vector=objective_function.evaluate_grad(point_yh.cartesian_coordinates),
            active_set=feasible_region.vertices,
            barycentric_coordinates=point_yh.barycentric_coordinates,
            tolerance=epsilon_l,
            tolerance_type="dual gap",
        )
        return z, point_x, point_v, point_yh, point_y

    def compute_eta_for_initial_point():
        pass

    @staticmethod
    def _check_eta_condition_1(objective_function, x, y, eta):
        """Check if f(y) <= f(x) + <nabla f (x), y - x> + eta / 2 * ||y - x||^2."""
        f_diff = objective_function.evaluate(y) - objective_function.evaluate(x)
        grad_x = objective_function.evaluate_grad(x)
        y_x = y - x
        return (f_diff <= np.dot(grad_x, y_x) + eta / 2 * np.dot(y_x, y_x))

    @staticmethod
    def _check_eta_condition_2(objective_function, x, y, eta):
        """Check if ||nabla f (y) - nabla f (x)|| <= eta * ||y - x||."""
        grad_diff = objective_function.evaluate_grad(y) - objective_function.evaluate_grad(x)
        y_x = y - x
        return (np.dot(grad_diff, grad_diff) <= eta * np.linalg.norm(y_x))

    @staticmethod
    def _compute_a(A_, theta_max):
        """Compute largest a satisfying a / A <= theta_max."""
        if not (theta_max < 1.0 and theta_max > 0):
            raise ValueError("theta_max must be within 0 to 1.")
        return A_ * theta_max / (1 - theta_max)



















class FractionalAwayStepFW(_AbstractAlgorithm):
    def __init__(self, fw_variant = "AFW", ratio=0.5, **kwargs):
        self.ratio = 0.5
        self.fw_variant = fw_variant
        pass

    #Use AFW algorithm to halve the strong Wolfe gap until it is below a given tolerance.
    def run(
        self,
        objective_function,
        feasible_region,
        initial_point,
        active_set,
        lambdas,
    ):
        grad = objective_function.evaluate_grad(initial_point)
        v = feasible_region.lp_oracle(grad)
        a, indexMax = feasible_region.away_oracle(grad, active_set)
        strong_FW_gap = np.dot(grad, a - v)
        target_accuracy = strong_FW_gap*self.ratio
        from pflacg.algorithms.fw_variants import FrankWolfe
        fw_algorithm = FrankWolfe(self.fw_variant, "line_search")
        from pflacg.algorithms._algorithms_utils import ExitCriterion
        exit_criterion = ExitCriterion("SWG", target_accuracy)
        results = fw_algorithm.run(objective_function, feasible_region, exit_criterion, initial_point, active_set,lambdas, save_and_output_results = False)
        return results
