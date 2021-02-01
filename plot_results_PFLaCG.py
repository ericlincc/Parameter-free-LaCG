import os, sys

base_directory = '/scratch/share/pflacg_experiments/run_results/'

for directory in os.listdir(base_directory):
  #Directory does not contain an image, so generate it with the results inside the folder.
  subdirectory = os.path.join(base_directory, directory)
  list_subdirectories_inside = os.listdir(subdirectory)
  if(not any('.png' in file_string for file_string in list_subdirectories_inside)):
    run_histories = [subdirectory + '/' + results for results in list_subdirectories_inside if '.run_history' in results]      
    collect_run_histories = (' ').join(run_histories)
    #Plot with respect to iteration
    os_run_command = 'python -m pflacg.experiments.experiments_driver --task plot_results --x_axis iteration --y_axis strong_wolfe_gap --path_to_results ' + collect_run_histories + ' --save_location ' + subdirectory
    os.system(os_run_command)
    #Plot with respect to time
    os_run_command = 'python -m pflacg.experiments.experiments_driver --task plot_results --x_axis time --y_axis strong_wolfe_gap --path_to_results ' + collect_run_histories + ' --save_location ' + subdirectory
    os.system(os_run_command)