# codeing=utf-8
"""This module contains the abastract algorithm for the experiements."""


from abc import ABC, abstractmethod


class _AbstractAlgorithm(ABC):
    def __init__(self):
        """
        Abstract __init__ method. This method should only contain algorithm-level variables.
        """

        pass

    @abstractmethod
    def run(
        self,
        objective_function,
        feasible_region,
        exit_criterion,
        point_initial=None,
        **kwargs,
    ):
        """
        Abstract method for running the algorithm on an objective function over a feasible region
        given an initial point x_0.

        Parameters
        ----------
        objective_function: Implemented _AbstractObjectiveFunction
            Objective function an algorithm will be optimizing for.
        feasible_region: Implemented _AbstractFeasibleRegion
            Feasible region an algorithm will be optimizing over.
        exit_criterion: ExitCriterion
            Stopping condidion of an algorihtm run.
        point_initial: Point
            Optional initial point of this algorithm run.

        Returns
        -------
        list(tuple)
        """

        pass
