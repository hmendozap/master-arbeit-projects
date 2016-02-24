# -*- encoding: utf-8 -*-

import numpy as np
from autosklearn.pipeline.components.classification import add_classifier
import autosklearn.automl as autosk
from component import DeepFeedNet

dataset_dir = '/home/hmendoza/workspace/master_arbeit/auto-deep/datasets/dataset_728/'

## Load our training data
X_train = np.load(dataset_dir + 'train.npy')
y_train = np.load(dataset_dir + 'train_labels.npy')

add_classifier(DeepFeedNet.DeepFeedNet)

## Create model
modl = autosk.AutoML(time_left_for_this_task=1800, per_run_time_limit=180,
                     delete_tmp_folder_after_terminate=False,
                     tmp_dir='/tmp/autosk_tmp', output_dir='/tmp/autosk_out',
                     log_dir='/tmp/autosk_log',
                     include_estimators=['DeepFeedNet'],
                     ensemble_size=1,
                     ensemble_nbest=1,
                     initial_configurations_via_metalearning=0,
                     seed=200,
                     ml_memory_limit=2048,
                     metadata_directory=None,
                     queue=None,
                     keep_models=True,
                     debug_mode=False,
                     include_preprocessors=None,
                     resampling_strategy='holdout',
                     resampling_strategy_arguments=None)

modl.fit(X_train, y_train)

X_test = np.load(dataset_dir + 'test.npy')
y_test = np.load(dataset_dir + 'test_labels.npy')

tot_score = modl.score(X_test, y_test)
print tot_score

## Only predict before getting scoring
y_pred = modl.predict(X_test)

## Comparison
accuracy = np.count_nonzero(y_test == y_pred)
print (float(accuracy) / X_test.shape[0])
