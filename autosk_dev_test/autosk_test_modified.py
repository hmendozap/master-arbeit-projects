# -*- encoding: utf-8 -*-

import numpy as np
import autosklearn.automl as autosk
import sklearn.metrics
from autosklearn.constants import BINARY_CLASSIFICATION,\
    MULTICLASS_CLASSIFICATION

dataset_dir = 

# Load our training data
X_train = np.loadtxt(dataset_dir + 'train.data')
y_train = np.loadtxt(dataset_dir + 'train.solution')
y_train = np.argmax(y_train, axis=1)

# Create model
modl = autosk.AutoML(time_left_for_this_task=1800, per_run_time_limit=600,
                     delete_tmp_folder_after_terminate=False,
                     tmp_dir='tmp/autosk_tmp', output_dir='tmp/autosk_out',
                     log_dir='tmp/autosk_log',
                     include_estimators=['DeepFeedNet'],
                     include_preprocessors=['no_preprocessing'],
                     ensemble_size=2,
                     ensemble_nbest=2,
                     initial_configurations_via_metalearning=0,
                     seed=50,
                     ml_memory_limit=4096,
                     metadata_directory=None,
                     queue=None,
                     keep_models=False,
                     debug_mode=False,
                     resampling_strategy='holdout',
                     resampling_strategy_arguments=None)

modl.fit(X_train, y_train, dataset_name='evita_Testing',
         task=MULTICLASS_CLASSIFICATION)

X_test = np.loadtxt(dataset_dir + 'test.data')
y_test = np.loadtxt(dataset_dir + 'test.solution')
y_test = np.argmax(y_test, axis=1)

print(modl.show_models())
predictions = modl.predict(X_test)
print("Accuracy score", sklearn.metrics.accuracy_score(y_test, predictions))
