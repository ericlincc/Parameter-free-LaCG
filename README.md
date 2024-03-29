# Code for Parameter-free Locally Accelerated Conditional Gradient and its experiments.


## Setting up the conda environment to run experiments


We use [Anaconda](https://www.anaconda.com/) for Python package management. When you have it installed, you can create a new conda environment using `environment.yaml`:
```
$ conda env create --name pflacg --file=environment.yaml
```

Activate the `pflacg` conda environment for executing the code in the `pflacg` Python module:
```
$ conda activate pflacg
```

## A brief overview of our codebase


The bulk of the code for running the experiments is located within `pflacg` which can be imported as a Python module from the root of this repository (or alternative if copied to the binary directory of your Python installation). All implemented algorithms are located under `pflacg/algorithms` and the implementation for **Parameter-free Locally Accelerated Conditional Gradient** is located inside `pflacg/algorithms/pflacg.py`. The code for construsting relevant objective functions and feasible regions is located under `pflacg/experiemnts`. The primary tool we use for running our experiments is `pflacg/experiments/experiment_driver.py` and instructions for using it are included in the next section.


## Running the experiments


### Using the experiments driver

Here we provide detailed instructions on how to run experiments using `experiments_driver.py`. Please find example configurations for the algorithms, feasible regions and objectives functions under `examples`, or alternative you can construct the necessary objects using your custom Python script.

```
python -m pflacg.experiments.experiments_driver --task run_algorithms \
    --save_location $SAVE_LOCATION \
    --objective_function_config $PATH_TO_OBJECTIVE_FUNCTION_CONFIG \
    --feasible_region_config $PATH_TO_FEASIBLE_REGION_CONFIG \
    --algorithms_config $PATH_TO_ALGORITHMS_CONFIG \
    --exit_criterion_type $EXIT_CRITERION_TYPE \
    --exit_criterion_value $EXIT_CRITERION_VALUE \
    --max_time $MAX_TIME
```

Optional arguments:

* ```--save_objects```: Save all experiments related objects (objective function, feasible region and algorithms).
* ```--objective_function_use_pickled $PATH_TO_PICKLED_OBJECTIVE_FUNCTION```: Use an existing pickled objective function. Do not use this with ```--objective_function_config```.
* ```--feasible_region_use_pickled $PATH_TO_PICKLED_FEASIBLE_REGION```: Use an existing pickled feasible region. Do not use this with ```--feasible_region_config```.
* ```--num_cpu_per_process $NUM_CPU```: Allowed number of CPUs per process. If 0, then no limit. Default is 1.
* ```--random_seed $SEED```: Seed for random in NumPy.
* ```--save_logging```: Save logs to a file in save_location.
* ```--clear_pycache```: Clear python cache. Please include this option after any changes is made to the code.

Upon completion, the results from an experiment run will be save as text files within your specified `$SAVE_LOCATION`. You can then use `experiment_driver.py` to generate 


#### Example run commands


We provide two pseudo-commands for using the experiment driver. The first comand runs algorithms over the supplied objective function and feasible, while the second command plots the run results.

```
$ python -m pflacg.experiments.experiments_driver --task run_algorithms \
    --save_location test_runs \
    --objective_function_use_pickled ./Quadratic-dim_5000.pickle \
    --feasible_region_use_pickled ./ProbabilitySimplexPolytope-dim_5000.pickle \
    --algorithms_config ./examples/algorithms/example_PFLaCG.json \
    --exit_criterion_type SWG \
    --exit_criterion_value 0.001 \
    --num_cpu_per_process 1 \
    --max_time 100 \
    --clear_pycache \
    --save_logging
```

```
$ python -m pflacg.experiments.experiments_driver --task plot_results \
    --save_location "your/save/location" \
    --plot_config "examples/plots/plot_config.json"
```


### Using Jupyter notebook


Jupyter notebook comes installed with our `pflacg` conda environment. Start a Jupyter notebook server while in the root directory of this repository:

```
jupyter notebook
```

When inside a Jupyter notebook, you can import the `pflacg` module by simply running `import pflacg`. To set up your custom experiements, please first instantiate your desire feasible region and objective function objects, then instantiate algorithm objects such as `ParameterFreeLaCG`. To run your experiements, simply call the `run` method inside any algorihtm objects.


#### Example notebook

We provide a toy example notebook for using our `pflacg` module and it is available under `examples/Example-Quadratic-ProbSimplex.ipynb`.
