# -*- encoding: utf-8 -*-

import autosklearn.automl as autosk
import argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument('dataset')
parser.add_argument('output_directory', metavar='output-directory')
parser.add_argument('seed', type=int)
parser.add_argument('reproducible_ensemble', metavar='reproducible-ensemble')
args = parser.parse_args()

dataset = args.dataset
output_directory = args.output_directory
seed = args.seed
reproducible_ensemble = args.reproducible_ensemble

ensemble_size = 2
total_runtime = 600
time_per_run = 60

# Create model
modl = autosk.AutoML(time_left_for_this_task=total_runtime,
                     per_run_time_limit=time_per_run,
                     delete_tmp_folder_after_terminate=False,
                     tmp_dir=output_directory,
                     output_dir=output_directory,
                     include_estimators=['DeepNetIterative'],
                     include_preprocessors=['NoPreprocessing'],
                     ensemble_size=0,
                     ensemble_nbest=ensemble_size,
                     initial_configurations_via_metalearning=0,
                     seed=seed,
                     shared_mode=True,
                     ml_memory_limit=1000000000,  # max memory?
                     metadata_directory=None,
                     keep_models=True,
                     debug_mode=False,
                     resampling_strategy='holdout-iterative-fit',
                     resampling_strategy_arguments=None)

modl.fit_automl_dataset(dataset)
modl.run_ensemble_builder(0, 1, ensemble_size).wait()
time.sleep(5)

with open(reproducible_ensemble, "w") as fh:
    fh.write(modl.show_models())
