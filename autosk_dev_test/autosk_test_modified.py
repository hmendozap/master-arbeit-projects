# -*- encoding: utf-8 -*-

import numpy as np
from autosklearn.pipeline.components.classification import add_classifier
import time
import autosklearn.automl as autosk
from autosklearn.constants import BINARY_CLASSIFICATION
from component import DeepFeedNet

dataset_dir = 

# Load our training data
X_train = np.load(dataset_dir + 'train.npy')
y_train = np.load(dataset_dir + 'train_labels.npy')

add_classifier(DeepFeedNet.DeepFeedNet)

ensemble_size = 1
# Create model
modl = autosk.AutoML(time_left_for_this_task=600, per_run_time_limit=90,
                     delete_tmp_folder_after_terminate=False,
                     tmp_dir='tmp/autosk_tmp', output_dir='tmp/autosk_out',
                     log_dir='tmp/autosk_log',
                     include_estimators=['DeepFeedNet'],
                     include_preprocessors=['NoPreprocessing'],
                     ensemble_size=0,
                     ensemble_nbest=0,
                     initial_configurations_via_metalearning=0,
                     seed=50,
                     ml_memory_limit=2048,
                     metadata_directory=None,
                     queue=None,
                     keep_models=True,
                     debug_mode=False,
                     resampling_strategy='holdout',
                     resampling_strategy_arguments=None)

modl.fit(X_train, y_train, dataset_name='728_Testing',
         task=BINARY_CLASSIFICATION)

X_test = np.load(dataset_dir + 'test.npy')
y_test = np.load(dataset_dir + 'test_labels.npy')

try:
    reproducible_ensemble = 'tmp/autosk_tmp/728_ensemble.txt'
    modl.run_ensemble_builder(0, 1, ensemble_size).wait()
    time.sleep(5)
    with open(reproducible_ensemble, "w") as fh:
        fh.write(modl.show_models())
except Exception as E:
    print("Ups, there is an error: %s" % E)
