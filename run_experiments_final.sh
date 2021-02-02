# First argument must be the prefix (WITHOUT .pickle) of the feasible region pickled files to run
# Second arguemtn must be the prefix (WITHOUT .pickle) of the objective function pickled files to run
FEASIBLE_REGION_PREFIX=$1
OBJECTIVE_FUNCTION_PREFIX=$2
if [ -z "$FEASIBLE_REGION_PREFIX" ] || [ -z "OBJECTIVE_FUNCTION_PREFIX" ]
then
    echo "Mising argument(s). Read comments in script!"
    exit 1
fi


# Helper functions


timestamp() {
  date +"%Y-%m-%d_%H-%M-%S" # current datetime
}


# Main


set -e


# Directory
OBJECTIVE_FUNCTION_DIR="/scratch/share/pflacg_experiments/pickled_objects/objective_functions"
FESIBLE_REGION_DIR="/scratch/share/pflacg_experiments/pickled_objects/feasible_regions"
BASE_RESULTS_DIR="/scratch/share/pflacg_experiments/final_run_results"


# Run configs (Can be edited)
ALGO_CONFIG="/home/bzfcarde/Computations/Parameter-free-LaCG/examples/algorithms/run_all_algorithms_iter_sync_false.json"
EXIT_CRIT_TYPE="SWG"
EXIT_CRIT_VALUE="0.000001"
MAX_TIME="$((15 * 60))"


# !!!!! >>>>> TRY NOT TO CHANGE THE CODE BELOW THIS LINE <<<<< !!!!!

for feasible_region_path in $FESIBLE_REGION_DIR/$FEASIBLE_REGION_PREFIX*.pickle
do
    for objective_function_path in $OBJECTIVE_FUNCTION_DIR/$OBJECTIVE_FUNCTION_PREFIX*.pickle
    do
        feasible_region=$(basename ${feasible_region_path%.*})
        objective_function=$(basename ${objective_function_path%.*})
        echo "Running $feasible_region on $objective_function."

        save_dir="$BASE_RESULTS_DIR/$(timestamp)---$feasible_region---$objective_function"
        mkdir $save_dir

        python -m pflacg.experiments.experiments_driver --task run_algorithms \
            --save_location "$save_dir" \
            --feasible_region_use_pickled "$feasible_region_path" \
            --objective_function_use_pickled "$objective_function_path" \
            --algorithms_config "$ALGO_CONFIG" \
            --exit_criterion_type "$EXIT_CRIT_TYPE" \
            --exit_criterion_value "$EXIT_CRIT_VALUE" \
            --max_time "$MAX_TIME" \
            --single_cpu_mode \
			--save_objects \
            --save_logging
    done
done
