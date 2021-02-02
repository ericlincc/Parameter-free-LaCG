# codeing=utf-8
"""This module is the driver for the experiments."""


import argparse
import datetime
import logging
import json
from os import environ, mkdir, path, system
import pickle
import time
import sys

import matplotlib.pyplot as plt

import pflacg.algorithms as algorithms
import pflacg.experiments.feasible_regions as feasible_regions
import pflacg.experiments.objective_functions as objective_functions
import pflacg.experiments.experiments_helper as helper


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s :: %(asctime)s :: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
LOGGER = logging.getLogger()


# Helper functions


def get_current_timestamp(time_format="%Y-%m-%d %H:%M:%S"):
    ts = time.time()
    return (
        datetime.datetime.fromtimestamp(ts)
        .strftime(time_format)
        .replace(" ", "-")
        .replace(":", "-")
    )


def load_pickled_object(filepath, parent_type=None):
    with open(filepath, "rb") as f:
        loaded_object = pickle.load(f)

    # Safety checks
    if parent_type is None and not issubclass(loaded_object.__class__, parent_type):
        raise TypeError(
            "Object pickled at {0} is not a subclass of {1}".format(
                filepath, str(parent_type)
            )
        )
    return loaded_object


def dump_pickled_object(filepath, target_object):
    with open(filepath, "wb") as f:
        pickle.dump(target_object, f)


# Core


def run_algorithms(args):
    """Run algorithms given objective function and feasible region, then save data."""

    # Save timestamp for later identification
    timestamp = get_current_timestamp()

    # Create directory if not already exists
    if not path.isdir(args.save_location):
        mkdir(args.save_location)

    # Reconfigure logging to save to file
    if args.save_logging:
        logging_filename = path.join(args.save_location, f"run_log-{timestamp}")
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        logging.basicConfig(
            filename=logging_filename,
            level=logging.INFO,
            format="%(levelname)s :: %(asctime)s :: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    # Clean all caches in __pycache__ directories
    if args.clear_pycache:
        system("py3clean .")

    # Limit numpy to use one CPU core by setting MKL single-thread only. This must be
    # done before numpy is imported for the first time.
    # This setting is machine-specific and can be brittle.
    if args.single_cpu_mode:
        environ["MKL_NUM_THREADS"] = "1"
        environ["NUMEXPR_NUM_THREADS"] = "1"
        environ["OMP_NUM_THREADS"] = "1"
    import numpy as np

    # Set random seed if given
    if args.random_seed:
        np.random.seed(args.random_seed)

    # Objective function
    if (args.objective_function_config is None) == (
        args.objective_function_use_pickled is None
    ):
        raise ValueError(
            "Exactly one of objective_function_config or objective_function_use_pickled "
            "should be passed as argument"
        )
    elif args.objective_function_config is not None:
        # Parse config
        with open(args.objective_function_config, "r") as f:
            objective_function_config = json.load(f)
        # Instantiate objective function
        if objective_function_config["class_name"] == "Quadratic":
            dim = objective_function_config["args"]["dim"]
            sparsity = 0.1
            M = objective_functions.random_psd_generator_sparse(dim, sparsity)
            objective_function = objective_functions.Quadratic(
                dim=dim, M=M, b=np.zeros(dim)
            )
        elif objective_function_config["class_name"] == "HuberLoss":
            dim = objective_function_config["args"]["dim"]
            radius = objective_function_config["args"]["radius"]
            objective_function = objective_functions.HuberLoss(
                dim=dim, reference_point=1, radius=radius
            )
        else:
            raise ValueError(
                "Objective function class {0} not implemented.".format(
                    objective_function_config["class_name"]
                )
            )
    else:
        # Load pickled objective function
        filepath = args.objective_function_use_pickled
        objective_function = load_pickled_object(
            filepath, objective_functions._AbstractObjectiveFunction
        )

    # Feasible region
    if (args.feasible_region_config is None) == (
        args.feasible_region_use_pickled is None
    ):
        raise ValueError(
            "Exactly one of feasible_region_config or feasible_region_use_pickled "
            "should be passed as argument"
        )
    elif args.feasible_region_config is not None:
        # Parse config
        with open(args.feasible_region_config, "r") as f:
            feasible_region_config = json.load(f)
        # Instantiate feasible region
        feasible_region_class = getattr(
            feasible_regions, feasible_region_config["class_name"]
        )
        feasible_region = feasible_region_class(**feasible_region_config["args"])
    else:
        # Load pickled feasible region
        filepath = args.feasible_region_use_pickled
        feasible_region = load_pickled_object(
            filepath, feasible_regions._AbstractFeasibleRegion
        )

    # Configure exit criterion
    exit_criterion = algorithms.ExitCriterion(
        args.exit_criterion_type,
        args.exit_criterion_value,
        criterion_reference=args.exit_criterion_reference,
        max_time=args.max_time,
    )

    # Algorithms
    # Parse config
    with open(args.algorithms_config, "r") as f:
        algorithms_config = json.load(f)
    LOGGER.info("Algorithm configs in this run:")
    for algo_config in algorithms_config:
        LOGGER.info(f"- {json.dumps(algo_config)}")

    # Instantiate algorithms
    list_algorithms = []
    for algorithm_config in algorithms_config:
        _algorithm_class = getattr(algorithms, algorithm_config["class_name"])
        _algorithm = _algorithm_class(**algorithm_config["args"])
        list_algorithms.append((algorithm_config["name"], _algorithm))
    # TODO: implement load algorithms from pickled if needed. Otherwise remove pickling algos.

    # Save all objects if save_objects is true
    if args.save_objects:
        pickled_path = path.join(args.save_location, "pickled_objects")
        if not path.isdir(pickled_path):
            mkdir(pickled_path)

        dump_pickled_object(
            path.join(
                pickled_path,
                "{0}-{1}.pickle".format(type(objective_function).__name__, timestamp),
            ),
            objective_function,
        )
        dump_pickled_object(
            path.join(
                pickled_path,
                "{0}-{1}.pickle".format(type(feasible_region).__name__, timestamp),
            ),
            feasible_region,
        )
        for algorithm_name, algorithm in list_algorithms:
            dump_pickled_object(
                path.join(
                    pickled_path, "{0}-{1}.pickle".format(algorithm_name, timestamp)
                ),
                algorithm,
            )

    # run the algorithms
    optional_run_args = {}  # No optional run args
    for algorithm_name, algorithm in list_algorithms:
        # run algorithm
        LOGGER.info(f"Running {algorithm_name} now.")
        run_history = algorithm.run(
            objective_function,
            feasible_region,
            exit_criterion,
            **optional_run_args,
        )

        # save data to file in json format
        LOGGER.info(f"Saving {algorithm_name} run history to {args.save_location}.")
        with open(
            path.join(
                args.save_location, algorithm_name + "_" + timestamp + ".run_history"
            ),
            "w",
        ) as f:
            for run_status in run_history:
                f.write(json.dumps(run_status) + "\n")


def plot_results(args):
    """Plot graphs from run histories."""

    run_status_index = {
        "iteration": 0,
        "time": 1,
        "primal_gap": 2,
        "wolfe_gap": 3,
        "strong_wolfe_gap": 4,
    }

    if args.plot_config:
        # If plot_config, use configs from plot_configs only.
        with open(args.plot_config, "r") as f:
            plot_config = json.load(f)

        if plot_config["x_axis"] == "time":
            x_label = r"t[s]"
        elif plot_config["x_axis"] == "iteration":
            x_label = r"$k$"
        else:
            raise ValueError("invalid value for y_axis")

        if plot_config["y_axis"] == "primal_gap":
            y_label = r"$f(x_k) - f(x^*)$"
            ref_opt = plot_config["known_optimal_f_val"]
        elif plot_config["y_axis"] == "strong_wolfe_gap":
            y_label = r"$w(x, S)$"
            ref_opt = 0
        else:
            raise ValueError("invalid value for y_axis")

        list_x = []
        list_y = []
        list_legend = []
        colors = []
        markers = []
        for run_result in plot_config["run_results"]:
            markers.append(run_result["marker"])
            colors.append(run_result["color"])
            list_legend.append(run_result["name"])

            x = []
            y = []
            with open(run_result["path_to_result"], "r") as f:
                for line in f:
                    run_status = json.loads(line.strip())
                    x.append(run_status[run_status_index[plot_config["x_axis"]]])
                    y.append(
                        run_status[run_status_index[plot_config["y_axis"]]] - ref_opt
                    )
            list_x.append(x)
            list_y.append(y)

        save_path = path.join(
            args.save_location,
            f"plot-{plot_config['y_axis']}-{plot_config['x_axis']}-{get_current_timestamp()}.png",
        )
        helper.plot_pretty(
            list_x=list_x,
            list_y=list_y,
            list_legend=list_legend,
            colors=colors,
            markers=markers,
            x_label=x_label,
            y_label=y_label,
            save_path=save_path,
            **plot_config,
        )

    else:
        # loading results into memory
        names_results = []
        for path_to_result in args.path_to_results:
            algorithm_name = path_to_result.split("/")[-1].split("_")[0]
            name_result = (algorithm_name, [])
            with open(path_to_result, "r") as f:
                for line in f:
                    run_status = json.loads(line.strip())
                    name_result[1].append(run_status)
            names_results.append(name_result)

        if args.y_axis == "primal_gap":
            ref_opt = args.known_optimal_f_val
            plt.ylabel(r"$f(x_k) - f(x^*)$")
        else:
            ref_opt = 0.0
            plt.ylabel(r"$g(x_k)$")
        if args.x_axis == "time":
            plt.xlabel(r"t[s]")
        else:
            plt.xlabel(r"$k$")

        for name_result in names_results:
            name, result = name_result
            gaps = [
                run_status[run_status_index[args.y_axis]] - ref_opt
                for run_status in result
            ]
            x_axis = [
                run_status[run_status_index[args.x_axis]] for run_status in result
            ]
            plt.semilogy(x_axis, gaps, label=name)

        plt.grid()
        plt.tight_layout()
        plt.legend()

        # Save timestamp for later identification
        plt.savefig(
            path.join(
                args.save_location,
                f"plot-{args.y_axis}-{args.x_axis}-{get_current_timestamp()}.png",
            )
        )


def run_algorithms_parser(argv_remaining):
    parser = argparse.ArgumentParser("Parse running algorithms settings")
    parser.add_argument(
        "--save_location",
        type=str,
        required=True,
        help="Save location of algorithm run results",
    )
    parser.add_argument(
        "--algorithms_config",
        type=str,
        required=True,
        help="path to algorithms configuration json file",
    )
    parser.add_argument(
        "--objective_function_config",
        type=str,
        required=False,
        help="path to objectve function configuration json file",
    )
    parser.add_argument(
        "--objective_function_use_pickled",
        type=str,
        required=False,
        help="use existing pickled objective function",
    )
    parser.add_argument(
        "--feasible_region_config",
        type=str,
        required=False,
        help="path to feasible region configuration json file",
    )
    parser.add_argument(
        "--feasible_region_use_pickled",
        type=str,
        required=False,
        help="use existing pickled feasible region",
    )
    parser.add_argument(
        "--exit_criterion_type",
        type=str,
        required=True,
        choices=["PG", "DG", "SWG", "IT"],
        help="type of stopping condition",
    )
    parser.add_argument(
        "--exit_criterion_value",
        type=float,
        required=True,
        help="threshold of stopping condition",
    )
    parser.add_argument(
        "--exit_criterion_reference",
        type=float,
        required=False,
        help="criterion reference for when criterion type is PG",
    )
    parser.add_argument(
        "--max_time",
        type=int,
        default=1800,
        help="max time in seconds allowed to run each algorithm. default=1800",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        required=False,
        help="Seed for random",
    )
    parser.add_argument(
        "--save_logging",
        action="store_true",
        help="Save execution logs to file.",
    )
    parser.add_argument(
        "--save_objects",
        action="store_true",
        help="Saving all experiment objects of later use.",
    )
    parser.add_argument(
        "--clear_pycache",
        action="store_true",
        help="Cleaning all python cache in __pycache__",
    )
    parser.add_argument(
        "--single_cpu_mode",
        action="store_true",
        help="Limiting each process to use at most one CPU core.",
    )
    return parser.parse_known_args(argv_remaining)


def plot_results_parser(argv_remaining):

    # Arg parsers
    parser = argparse.ArgumentParser("Parse running algorithms settings")
    parser.add_argument(
        "--save_location",
        type=str,
        required=True,
        help="Save location of plotted results",
    )
    parser.add_argument(
        "--plot_config",
        type=str,
        required=False,
        help=(
            "Config json file for plotting a nice graph (see example). "
            "If provided, ignoring other arguments except save_location"
        ),
    )
    parser.add_argument(
        "--path_to_results",
        type=str,
        nargs="+",
        required=False,
        help="Save locations of algorithm run results to be plotted",
    )
    parser.add_argument(
        "--known_optimal_f_val",
        type=float,
        required=False,
        default=0.0,
        help="Optimal value of f to the best of knowledge",
    )
    parser.add_argument(
        "--y_axis",
        type=str,
        required=False,
        choices=["primal_gap", "wolfe_gap", "strong_wolfe_gap"],
        help="Y-axis: primal gap or wolfe gap or strong_wolfe_gap",
    )
    parser.add_argument(
        "--x_axis",
        type=str,
        required=False,
        choices=["time", "iteration"],
        help="Quantity of x-axis to be plotted",
    )
    return parser.parse_known_args(argv_remaining)


TASKS = {
    "run_algorithms": (run_algorithms, run_algorithms_parser),
    "plot_results": (plot_results, plot_results_parser),
}


def get_task(argv):
    parser = argparse.ArgumentParser("Task parser")
    parser.add_argument(
        "--task",
        choices=TASKS.keys(),
        required=True,
        help="One of {0}".format(", ".join(TASKS.keys())),
    )
    return parser.parse_known_args(argv)


# Main


def main(argv):
    """Parse input and call task accordingly."""
    task_args, argv_remaining = get_task(argv)
    task, task_parser = TASKS[task_args.task]
    args, argv_remaining = task_parser(argv_remaining)
    if not argv_remaining:
        LOGGER.warning("There are unparsed arguments. Continuing")
    LOGGER.info("Executing task {0}".format(task_args.task))
    task(args)


if __name__ == "__main__":
    LOGGER.info(sys.argv)
    main(sys.argv)
