# codeing=utf-8
"""Contains code for parameter-free locally accelerated conditional gradient."""

from copy import deepcopy
import logging
import time
import numpy as np

from pflacg.algorithms._abstract_algorithm import _AbstractAlgorithm


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
        iniial_active_set=None
        async=False,
    ):
        pass


class FractionalAwayStepFW(_AbstractAlgorithm):
    def __init__(self, ratio=0.5, **kwargs):
        pass

    def run(self, **kwargs):
        pass


class wACC(_AbstractAlgorithm):
    def __init__(self, **kwargs):
        pass

    def run(self, **kwargs):
        pass
