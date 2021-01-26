# Code for Parameter-free Locally Accelerated Conditional Gradient and its experiments.


## Setting up environment to run experiments

Creating a new conda environment using `Parameter-free-LaCG/environments.yml`:
```
$ conda env create --name pflacg --file=environments.yaml
```

Activate the conda environment:
```
$ conda activate pflacg
```

## Setting up environment to run experiments

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
* ```--random_seed```: Seed for random in NumPy.

### Examples

```
python -m pflacg.experiments.experiments_driver --task run_algorithms \
    --save_location test_run \
    --objective_function_config examples/objective_functions/example_quadratic.json \
    --feasible_region_config examples/feasible_regions/example_probability_simplex.json \
    --algorithms_config examples/algorithms/example_PFLaCG.json \
    --exit_criterion_type SWG \
    --exit_criterion_value 0.001 \
    --max_time 100