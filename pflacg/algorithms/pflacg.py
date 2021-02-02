# codeing=utf-8
"""Contains code for parameter-free locally accelerated conditional gradient."""


import logging
import time
from multiprocessing import shared_memory, Value, Process, Lock

import numpy as np

from pflacg.experiments.objective_functions import RegularizedObjectiveFunction
from pflacg.algorithms._abstract_algorithm import _AbstractAlgorithm
from pflacg.algorithms._algorithms_utils import *  # TODO: Import only the methods and classes we need
from pflacg.experiments.feasible_regions import ConvexHull
from pflacg.algorithms.fw_variants import FrankWolfe
from pflacg.algorithms.project_onto_active_set_jit import (
    accelerated_projected_gradient_descent_over_simplex_jit,
)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s :: %(asctime)s :: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
LOGGER = logging.getLogger()


WAIT_TIME_FOR_LOCK = 0.2
MAX_NUM_WAIT_INTERVALS = int(120 / WAIT_TIME_FOR_LOCK)


# Algorithms


class ParameterFreeLaCG(_AbstractAlgorithm):
    """TODO: Add descriptions and reference."""

    def __init__(
        self,
        fw_variant="AFW",
        ratio=0.5,
        iter_sync=True,
    ):
        self.fw_variant = fw_variant
        self.ratio = 0.5
        self.iter_sync = iter_sync
        self.FAFW = FractionalAwayStepFW(fw_variant=self.fw_variant, ratio=self.ratio)
        self.ACC = ParameterFreeAGD(iter_sync=self.iter_sync)

    def run(
        self,
        objective_function,
        feasible_region,
        exit_criterion,
        point_initial=None,
    ):

        if point_initial is None:
            vertex = feasible_region.initial_point.copy()
            point_initial = Point(vertex, (1.0,), (vertex,))
        else:
            point_initial = point_initial

        # Initialization
        strong_wolfe_gap_out = compute_strong_wolfe_gap(
            point_initial, objective_function, feasible_region
        )
        iteration = 0
        start_time = time.time()
        duration = 0.0
        num_halvings = 0
        f_val = objective_function.evaluate(point_initial.cartesian_coordinates)
        run_status = (
            iteration,
            duration,
            f_val,
            0.0,  # TODO: Fix or remove dummy dual gap
            strong_wolfe_gap_out,
        )
        run_history = [run_status]
        LOGGER.info(
            "Running PFLaCG: "
            "iteration = {1}, duration = {2:.{0}f}, "
            "f_val = {3:.{0}f}, dual_gap = {4:.{0}f}, SWG = {5:.{0}f}".format(
                DISPLAY_DECIMALS, *run_status
            )
        )

        point_x_FAFW = point_initial
        point_x_ACC = point_initial
        active_set_ACC = point_initial.support
        strong_wolfe_gap_FAFW = strong_wolfe_gap_out
        strong_wolfe_gap_ACC = strong_wolfe_gap_out
        eta = None
        sigma = None

        # Create new shared memory buffers and buffer lock
        ret_x_cartesian_coordinates_shm = shared_memory.SharedMemory(
            create=True,
            size=np.zeros(shape=objective_function.dim, dtype=np.float64).nbytes,
        )
        ret_x_cartesian_coordinates = np.ndarray(
            shape=objective_function.dim,
            dtype=np.float64,
            buffer=ret_x_cartesian_coordinates_shm.buf,
        )
        ret_x_cartesian_coordinates[:] = point_x_ACC.cartesian_coordinates[:]
        global_eta = Value("d", 0)
        global_sigma = Value("d", 0)
        ACC_paused_flag = Value("i", 0)
        buffer_lock = Lock()
        global_iter = Value("i", 0)

        ACC_restart_flag = True
        ACC_process_started = False
        while not exit_criterion.has_met_exit_criterion(run_status):
            # Set halving strong Wolfe gap
            target_accuracy = strong_wolfe_gap_FAFW * self.ratio
            LOGGER.info(f"SWG target accuracy = {target_accuracy}")
            num_halvings += 1

            if ACC_restart_flag:
                LOGGER.info("Restarting ACC")
                ret_x_barycentric_coordinates_shm = shared_memory.SharedMemory(
                    create=True,
                    size=np.zeros(shape=len(active_set_ACC), dtype=np.float64).nbytes,
                )
                ret_x_barycentric_coordinates = np.ndarray(
                    shape=len(active_set_ACC),
                    dtype=np.float64,
                    buffer=ret_x_barycentric_coordinates_shm.buf,
                )
                ret_x_barycentric_coordinates[:] = point_x_ACC.barycentric_coordinates[
                    :
                ]
                ret_x_cartesian_coordinates[:] = point_x_ACC.cartesian_coordinates[:]

                shared_buffers_dict = {
                    "buffer_lock": buffer_lock,
                    "global_iter": global_iter,
                    "ACC_paused_flag": ACC_paused_flag,
                    "global_eta": global_eta,
                    "global_sigma": global_sigma,
                    "ret_x_cartesian_coordinates": ret_x_cartesian_coordinates_shm.name,
                    "ret_x_barycentric_coordinates": ret_x_barycentric_coordinates_shm.name,
                }

                LOGGER.info(f"Creating ACC process with set size {len(active_set_ACC)}")
                convex_hull_ACC = ConvexHull(active_set_ACC)
                ACC_process = Process(
                    target=self.ACC.run,
                    args=(
                        objective_function,
                        convex_hull_ACC,
                        point_x_ACC,
                        0.0,
                        eta,
                        sigma,
                        shared_buffers_dict,
                        iteration,
                        exit_criterion.criterion_value,
                    ),
                )
                if len(active_set_ACC) > 1:
                    LOGGER.info("Starting ACC process")
                    ACC_process.start()
                    ACC_process_started = True

            LOGGER.info("Running FAFW")
            # Run FAFW and wait for the output
            point_x_FAFW, _, strong_wolfe_gap_FAFW = self.FAFW.run(
                objective_function,
                feasible_region,
                point_x_FAFW,
                target_accuracy=target_accuracy,
                global_iter=global_iter,
            )
            with global_iter.get_lock():
                _global_iter = global_iter.value
            LOGGER.info(f"FAFW returned at global_iter = {_global_iter}")

            # if iteration sync, need to wait for ACC to complete same #iterations
            num_wait_interval = 0
            while self.iter_sync and ACC_process_started and ACC_process.is_alive():
                with ACC_paused_flag.get_lock():
                    _ACC_paused_flag = ACC_paused_flag.value
                if _ACC_paused_flag == 1:
                    # ACC has paused (the buffer is been updated since last ACC restart)
                    break
                else:
                    LOGGER.info("Waiting for ACC")
                    time.sleep(WAIT_TIME_FOR_LOCK)
                    num_wait_interval += 1
                    if num_wait_interval > MAX_NUM_WAIT_INTERVALS:
                        LOGGER.info("ACC timed out. Continuing with PFLaCG.")
                        break

            LOGGER.info("Acquiring buffer")
            # retrieve the most recent output
            buffer_lock.acquire()
            point_x_ACC = Point(
                np.copy(ret_x_cartesian_coordinates),
                np.copy(ret_x_barycentric_coordinates),
                active_set_ACC,
            )
            if ACC_process_started:
                sigma = global_sigma.value
                eta = global_eta.value
            buffer_lock.release()

            # Compute Strong Wolfe gap (or dual gap)
            strong_wolfe_gap_ACC_prev = strong_wolfe_gap_ACC
            strong_wolfe_gap_ACC = compute_strong_wolfe_gap(
                point_x_ACC, objective_function, feasible_region
            )

            if strong_wolfe_gap_FAFW <= min(
                strong_wolfe_gap_ACC, strong_wolfe_gap_ACC_prev / 2
            ):
                # Terminate ACC process and set restart flag
                LOGGER.info("FAFW did better")
                if ACC_process_started and ACC_process.is_alive():
                    LOGGER.info("Terminating ACC")
                    ACC_process.terminate()
                    ACC_process.join()
                ACC_process_started = False
                with ACC_paused_flag.get_lock():
                    ACC_paused_flag.value = 0
                ret_x_barycentric_coordinates_shm.close()
                ret_x_barycentric_coordinates_shm.unlink()

                ACC_restart_flag = True
                point_x_ACC = point_x_FAFW
                active_set_ACC = point_x_FAFW.support

                # Set output points
                point_x_out = point_x_FAFW
                strong_wolfe_gap_out = strong_wolfe_gap_FAFW
            else:
                LOGGER.info("ACC did better")
                LOGGER.info("Not terminating ACC")
                # Allow ACC to continue its execution
                ACC_restart_flag = False

                # Couple FAFW by using the better point from ACC if condition satisfies
                if len(point_x_ACC.support) <= len(point_x_FAFW.support):
                    LOGGER.info("FAFW <- ACC")
                    point_x_FAFW = point_x_ACC
                    strong_wolfe_gap_FAFW = strong_wolfe_gap_ACC

                # Set output points
                point_x_out = point_x_ACC
                strong_wolfe_gap_out = strong_wolfe_gap_ACC

            # Append output points
            LOGGER.info("Outputting")
            with global_iter.get_lock():
                iteration = global_iter.value
            duration = time.time() - start_time
            f_val = objective_function.evaluate(point_x_out.cartesian_coordinates)
            run_status = (
                iteration,
                duration,
                f_val,
                0.0,  # TODO: Fix or remove dummy dual gap
                strong_wolfe_gap_out,
            )
            LOGGER.info(
                "Running PFLaCG: "
                "iteration = {1}, duration = {2:.{0}f},"
                " f_val = {3:.{0}f}, dual_gap = {4:.{0}f}, SWG = {5:.{0}f}".format(
                    DISPLAY_DECIMALS, *run_status
                )
            )
            run_history.append(run_status)

        # Cleaning up buffers and process
        ret_x_cartesian_coordinates_shm.close()
        ret_x_cartesian_coordinates_shm.unlink()
        if ACC_process_started and ACC_process.is_alive():
            ACC_process.terminate()
            ACC_process.join()

        return run_history


class ParameterFreeAGD:
    def __init__(self, iter_sync=True, estimate_ratio=2):
        self.iter_sync = iter_sync
        self.estimate_ratio = estimate_ratio

    def run(
        self,
        objective_function,
        feasible_region,
        point_initial,
        epsilon=0.0,
        initial_eta=None,
        initial_sigma=None,
        shared_buffers_dict=None,
        last_restart_iter=0,
        epsilon_f=1e-12,
    ):
        """Run PF-ACC given an initial point and an active set/feasible region.

        Returns
        -------
        list(run_status)
        """

        LOGGER.info(f"ACC process started at last_restart_iter = {last_restart_iter}")
        if len(feasible_region.vertices) <= 1:
            return point_initial, initial_eta, initial_sigma, 0

        # Precomputations
        matrix = np.vstack(feasible_region.vertices)
        base_quadratic = matrix.dot(matrix.T)

        # Initial shared buffers based on shared_buffers_dict
        if shared_buffers_dict:
            global_eta = shared_buffers_dict["global_eta"]
            global_sigma = shared_buffers_dict["global_sigma"]

            ret_x_cartesian_coordinates_shm = shared_memory.SharedMemory(
                name=shared_buffers_dict["ret_x_cartesian_coordinates"]
            )
            ret_x_cartesian_coordinates = np.ndarray(
                shape=objective_function.dim,
                dtype=np.float64,
                buffer=ret_x_cartesian_coordinates_shm.buf,
            )

            ret_x_barycentric_coordinates_shm = shared_memory.SharedMemory(
                name=shared_buffers_dict["ret_x_barycentric_coordinates"]
            )
            ret_x_barycentric_coordinates = np.ndarray(
                shape=len(feasible_region.vertices),
                dtype=np.float64,
                buffer=ret_x_barycentric_coordinates_shm.buf,
            )

            global_iter = shared_buffers_dict["global_iter"]
            ACC_paused_flag = shared_buffers_dict["ACC_paused_flag"]
            buffer_lock = shared_buffers_dict["buffer_lock"]
        else:
            global_eta, global_sigma = None, None

        # Initializtion
        point_x = point_initial
        if np.allclose(point_x.cartesian_coordinates, point_x.support[0]):
            point_y = Point(
                point_x.support[1],
                [1.0 if i == 1 else 0.0 for i in range(len(point_x.support))],
                point_x.support,
            )
        else:
            point_y = Point(
                point_x.support[0],
                [1.0 if i == 0 else 0.0 for i in range(len(point_x.support))],
                point_x.support,
            )

        # Guess a eta if initial_sigma is None
        if initial_sigma is None or initial_sigma == 0.0:
            x = point_x.cartesian_coordinates
            y = point_y.cartesian_coordinates
            initial_sigma = (
                2.0
                * (
                    objective_function.evaluate(y)
                    - objective_function.evaluate(x)
                    - np.dot(objective_function.evaluate_grad(x), y - x)
                )
                / (np.linalg.norm(y - x) ** 2)
            )

        # Set initial_eta to initial_sigma if initial_eta is None
        if initial_eta is None or initial_eta == 0.0:
            initial_eta = initial_sigma
        eta = initial_eta
        sigma = initial_sigma
        LOGGER.info(f"initial_sigma = {initial_sigma}")
        LOGGER.info(f"initial_eta = {initial_eta}")
        iteration = 0

        # Early return if primal gap is small
        strong_wolfe_gap = compute_strong_wolfe_gap(
            point_x, objective_function, feasible_region
        )
        if strong_wolfe_gap <= epsilon_f:
            LOGGER.info("Early halting ACC with wolfe_gap <= epsilon_f")
            if shared_buffers_dict:
                buffer_lock.acquire()
                global_eta.value = eta
                global_sigma.value = sigma
                buffer_lock.release()
            return point_x, eta, sigma, iteration

        LOGGER.info("1st time argmin_quadratic_over_active_set")
        point_x_plus = argmin_quadratic_over_active_set(
            quadratic_coefficient=eta / 2.0,
            linear_vector=(
                objective_function.evaluate_grad(point_x.cartesian_coordinates)
                - eta * point_x.cartesian_coordinates
            ),
            active_set=feasible_region.vertices,
            reference_point=point_x,
            tolerance_type="gradient mapping",
            tolerance=eta / 32,
            base_quadratic=base_quadratic,
        )
        LOGGER.info("1st time argmin_quadratic_over_active_set ended")
        grad_mapping = (
            point_x.cartesian_coordinates - point_x_plus.cartesian_coordinates
        )

        while np.linalg.norm(grad_mapping) > epsilon and strong_wolfe_gap > epsilon_f:
            (
                point_x,
                grad_mapping,
                strong_wolfe_gap,
                eta,
                sigma,
                _iteration,
            ) = self.ACC_iter(
                objective_function,
                feasible_region,
                point_initial=point_x,
                eta=eta,
                sigma=sigma,
                global_eta=global_eta,
                global_sigma=global_sigma,
                base_quadratic=base_quadratic,
                epsilon_f=epsilon_f,
            )
            iteration += _iteration

            LOGGER.info("ACC about to update buffer.")
            if shared_buffers_dict:
                buffer_lock.acquire()
                global_eta.value = eta
                global_sigma.value = sigma
                buffer_lock.release()
            while shared_buffers_dict:
                # if global_iter is None, then assume no iteration sync required.
                if global_iter:
                    with global_iter.get_lock():
                        _global_iter = global_iter.value
                else:
                    _global_iter = np.infty

                if _global_iter >= last_restart_iter + iteration or not self.iter_sync:
                    # Update shared buffers
                    buffer_lock.acquire()
                    ret_x_cartesian_coordinates[:] = point_x.cartesian_coordinates[:]
                    ret_x_barycentric_coordinates[:] = point_x.barycentric_coordinates[
                        :
                    ]
                    buffer_lock.release()

                    # Continue with the next ACC
                    with ACC_paused_flag.get_lock():
                        ACC_paused_flag.value = 0
                    break
                else:
                    # Pausing ACC's execution and sleep for some time.
                    with ACC_paused_flag.get_lock():
                        ACC_paused_flag.value = 1
                    time.sleep(WAIT_TIME_FOR_LOCK)

        if shared_buffers_dict:
            buffer_lock.acquire()
            global_eta.value = eta
            global_sigma.value = sigma
            buffer_lock.release()
        return point_x, eta, sigma, iteration

    def ACC_iter(
        self,
        objective_function,
        feasible_region,
        point_initial,
        eta,
        sigma,
        global_eta=None,
        global_sigma=None,
        base_quadratic=None,
        epsilon_f=1e-8,
    ):
        """Executes one call of ACC from Algorithm 4 in the paper.

        Parameters
        ----------
        objective_function: Implemented _AbstractObjectiveFunction
            Objective function of the problem instance.
        feasible_region: Implemented _AbstractFeasibleRegion
            Feasible region of the problem instance.
        sigma: float
            Most recent estimate of the regularization parameter.
        point_initial: Point
            Initial point to start executing this algorithm.
        initial_eta: float
            Initial estimation of smoothness.

        Returns
        -------
        point_yh: Point
        eta: float
        sigma: float

        """

        # Initialization
        point_x = point_initial
        a = 1
        A = 1
        eta_0 = eta

        iteration = 0
        sigma_flag = False

        while not sigma_flag:

            # Initialization
            grad_x = objective_function.evaluate_grad(point_x.cartesian_coordinates)
            point_y = argmin_quadratic_over_active_set(
                quadratic_coefficient=(eta_0 + sigma) / 2.0,
                linear_vector=(
                    grad_x - (eta_0 + sigma) * point_x.cartesian_coordinates
                ),
                active_set=feasible_region.vertices,
                reference_point=point_x,
                tolerance_type="gradient mapping",
                tolerance=(eta_0 + sigma) / 32,
                base_quadratic=base_quadratic,
            )
            epsilon_0 = ((eta_0 + sigma) / 32) * (
                np.linalg.norm(
                    point_y.cartesian_coordinates - point_x.cartesian_coordinates
                )
                ** 2
            )
            point_v = point_y
            point_yh = point_y
            z = (eta_0 + sigma) * point_x.cartesian_coordinates - grad_x
            a = 1.0
            A = 1.0

            reg_objective_function = RegularizedObjectiveFunction(
                objective_function=objective_function,
                sigma=sigma,
                reference_point=point_initial.cartesian_coordinates,
            )

            inner_complete_flag = False
            while not inner_complete_flag:
                theta = a / A
                epsilon_l = theta * epsilon_0 / 4
                epsilon_M = a * epsilon_0 / 4

                (
                    eta,
                    A,
                    z,
                    point_v,
                    point_yh,
                    point_y,
                    grad_mapping,
                    _iteration,
                ) = self.AGD_iter(
                    objective_function,
                    reg_objective_function,
                    feasible_region,
                    point_yh,
                    point_v,
                    z,
                    A,
                    eta,
                    sigma,
                    epsilon_0,
                    eta_0,
                    global_eta=global_eta,
                    base_quadratic=base_quadratic,
                    epsilon_f=epsilon_f,
                )
                iteration += _iteration

                strong_wolfe_gap = compute_strong_wolfe_gap(
                    point_yh, objective_function, feasible_region
                )
                if strong_wolfe_gap <= epsilon_f:
                    LOGGER.info(
                        "Early halt inside ACC with strong_wolfe_gap <= epsilon_f"
                    )
                    return (
                        point_yh,
                        grad_mapping,
                        strong_wolfe_gap,
                        eta,
                        sigma,
                        iteration,
                    )

                if (
                    np.linalg.norm(grad_mapping) ** 2 / (eta + sigma)
                    <= 9 * epsilon_0 / 4
                ):
                    inner_complete_flag = True

            if sigma * np.linalg.norm(
                point_yh.cartesian_coordinates - point_initial.cartesian_coordinates
            ) / np.sqrt(eta + sigma) <= np.sqrt(epsilon_0):
                sigma_flag = True
            else:
                sigma = sigma / self.estimate_ratio
                if global_sigma:
                    with global_sigma.get_lock():
                        global_sigma.value = sigma
                LOGGER.info(f"Sigma halved: sigma = {sigma}")

        return point_yh, grad_mapping, strong_wolfe_gap, eta, sigma, iteration

    def AGD_iter(
        self,
        objective_function,
        reg_objective_function,
        feasible_region,
        point_y_,
        point_v_,
        z_,
        A_,
        eta,
        sigma,
        epsilon_0,
        eta_0,
        global_eta=None,
        base_quadratic=None,
        epsilon_f=1e-8,
    ):
        """Executing one AGD-Iter as described in Algo 1 of the paper."""
        iteration = 0
        eta_flag = False

        while not eta_flag:
            iteration += 1

            theta_max = np.sqrt(sigma / (2 * (eta + sigma)))
            a = self._compute_a(A_, theta_max)
            A = A_ + a
            theta = a / A
            epsilon_l = theta * epsilon_0 / 4
            epsilon_M = a * epsilon_0 / 4

            point_x, z, point_v, point_yh, point_y = self.AGD_step(
                reg_objective_function,
                feasible_region,
                point_y_,
                point_v_,
                z_,
                a,
                A,
                eta,
                sigma,
                epsilon_l,
                epsilon_M,
                eta_0,
                base_quadratic=base_quadratic,
            )

            eta_x_yh = self._check_eta_condition(
                objective_function,
                point_x,
                point_yh,
                eta,
            )
            eta_yh_y = self._check_eta_condition(
                objective_function,
                point_yh,
                point_y,
                eta,
            )

            if eta_x_yh and eta_yh_y:
                eta_flag = True
            else:
                eta = self.estimate_ratio * eta  # Maybe udpate a global eta too
                if global_eta:
                    with global_eta.get_lock():
                        global_eta.value = eta
                LOGGER.info(f"Eta doubled: eta = {eta}")

        grad_mapping = (eta + sigma) * (
            point_yh.cartesian_coordinates - point_y.cartesian_coordinates
        )
        return eta, A, z, point_v, point_yh, point_y, grad_mapping, iteration

    def AGD_step(
        self,
        reg_objective_function,
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
        base_quadratic=None,
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

        z = (
            z_
            - a * reg_objective_function.evaluate_grad(point_x.cartesian_coordinates)
            + sigma * a * point_x.cartesian_coordinates
        )
        point_v = argmin_quadratic_over_active_set(
            quadratic_coefficient=(sigma * A + eta_0) / 2,
            linear_vector=(-z),
            active_set=feasible_region.vertices,
            reference_point=point_x,
            tolerance_type="dual gap",
            tolerance=epsilon_M,
            base_quadratic=base_quadratic,
        )
        point_yh = (1 - theta) * point_y_ + theta * point_v
        point_y = argmin_quadratic_over_active_set(
            quadratic_coefficient=(eta + sigma) / 2,
            linear_vector=(
                reg_objective_function.evaluate_grad(point_yh.cartesian_coordinates)
                - (eta + sigma) * point_yh.cartesian_coordinates
            ),
            active_set=feasible_region.vertices,
            reference_point=point_yh,
            tolerance_type="dual gap",
            tolerance=1e-10,
            base_quadratic=base_quadratic,
        )
        return (
            point_x,
            z,
            point_v,
            point_yh,
            point_y,
        )

    @staticmethod
    def _check_eta_condition(objective_function, point_x, point_y, eta):
        return (
            objective_function.evaluate_smoothness_inequality(
                point_x.cartesian_coordinates, point_y.cartesian_coordinates
            )
            <= 0.5 * eta
        )

    @staticmethod
    def _compute_a(A_, theta_max):
        """Compute largest a satisfying a / A <= theta_max."""
        if not (theta_max < 1.0 and theta_max > 0):
            raise ValueError("theta_max must be within 0 to 1.")
        return A_ * theta_max / (1 - theta_max)


class FractionalAwayStepFW:
    """TODO: Add descriptions and reference."""

    def __init__(self, fw_variant="AFW", ratio=0.5, **kwargs):
        assert (
            fw_variant == "AFW" or fw_variant == "PFW"
        ), "Wrong variant supplied to the adaptive algorithm"
        self.ratio = 0.5
        self.fw_variant = fw_variant

    # Use AFW algorithm to halve the strong Wolfe gap until it is below a given tolerance.
    def run(
        self,
        objective_function,
        feasible_region,
        point_initial,
        target_accuracy=None,
        global_iter=None,
    ):
        if target_accuracy is None:
            grad = objective_function.evaluate_grad(point_initial.cartesian_coordinates)
            v = feasible_region.lp_oracle(grad)
            point_a, index_max = feasible_region.away_oracle(grad, point_initial)
            strong_wolfe_gap = np.dot(grad, point_a.cartesian_coordinates - v)
            target_accuracy = strong_wolfe_gap * self.ratio

        fw_algorithm = FrankWolfe(self.fw_variant, "line_search")
        exit_criterion = ExitCriterion("SWG", target_accuracy)
        point_out, dual_gap_out, strong_wolfe_gap_out = fw_algorithm.run(
            objective_function,
            feasible_region,
            exit_criterion,
            point_initial,
            save_and_output_results=False,
            global_iter=global_iter,
        )
        return point_out, dual_gap_out, strong_wolfe_gap_out
