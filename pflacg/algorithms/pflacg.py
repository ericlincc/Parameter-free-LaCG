# codeing=utf-8
"""Contains code for parameter-free locally accelerated conditional gradient."""

from copy import deepcopy
import logging
import time
import numpy as np

from pflacg.algorithms._abstract_algorithm import _AbstractAlgorithm

from pflacg.algorithms._algorithms_utils import step_size, DISPLAY_DECIMALS, new_vertex_fail_fast, delete_vertex_index, calculate_stepsize


logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s :: %(asctime)s :: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOGGER = logging.getLogger()


class ParameterFreeLaCG(_AbstractAlgorithm):
    def __init__(self, **kwargs):
        pass

    def run(
        self,
        objective_function,
        feasible_region,
        exit_criterion,
        initial_point=None,
        initial_active_set=None,
        # async=False,
    ):
        pass


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

class wACC(_AbstractAlgorithm):
    def __init__(self, **kwargs):
        pass

    def run(self, **kwargs):
        pass