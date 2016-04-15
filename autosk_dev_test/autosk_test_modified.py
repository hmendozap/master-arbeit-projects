# -*- encoding: utf-8 -*-

import numpy as np
import autosklearn.automl as autosk
import sklearn.metrics
from autosklearn.constants import BINARY_CLASSIFICATION,\
    MULTICLASS_CLASSIFICATION

dataset_dir = 

# Create model
modl = autosk.AutoML(time_left_for_this_task=300, per_run_time_limit=90,
                     delete_tmp_folder_after_terminate=False,
                     tmp_dir='tmp/evita_ask', output_dir='tmp/evita_ask',
                     log_dir='tmp/autosk_log',
                     include_classifiers=['DeepFeedNet', 'LogReg'],
                     include_regressors=['RegDeepNet', 'LinReg'],
                     include_preprocessors=['no_preprocessing'],
                     ensemble_size=1,
                     ensemble_nbest=1,
                     initial_configurations_via_metalearning=0,
                     seed=150,
                     ml_memory_limit=4096,
                     metadata_directory=None,
                     queue=None,
                     keep_models=True,
                     debug_mode=False,
                     resampling_strategy='holdout',
                     resampling_strategy_arguments=None)

modl.fit_automl_dataset(dataset_dir)

#print(modl.show_models())
#predictions = modl.predict(X_test)
#print("Accuracy score", sklearn.metrics.accuracy_score(y_test, predictions))
