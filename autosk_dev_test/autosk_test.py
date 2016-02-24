# -*- encoding: utf-8 -*-

## Because we need it
import numpy as np
import autosklearn.automl as autosk
# import autosklearn.classification.AutoSklearnClassifier
# import autosklearn.estimators as autoEstim

dataset_dir = '/home/hmendoza/workspace/master_arbeit/auto-deep/datasets/dataset_728/'

## Load our training data
X_train = np.load(dataset_dir + 'train.npy')
y_train = np.load(dataset_dir + 'train_labels.npy')


## Create model
modl = autosk.AutoML(time_left_for_this_task=180, seed=10, per_run_time_limit=30,
                     tmp_dir='/tmp/autosk_tmp', output_dir='/tmp/autosk_out',
                     ensemble_size=1, ensemble_nbest=1, keep_models=True,
                     ml_memory_limit=2048,
                     delete_tmp_folder_after_terminate=False,
                     initial_configurations_via_metalearning=0,
                     include_estimators=['feed_nn'])

modl.fit(X_train, y_train)

X_test = np.load(dataset_dir + 'test.npy')
y_test = np.load(dataset_dir + 'test_labels.npy')

# Only predict before getting scoring (Syntax changed from development-java@58b0e3)
y_pred = modl.predict(X_test)

# Try to calculate score
tot_score = modl.score(X_test, y_test)
print tot_score

# Comparison
accuracy = np.count_nonzero(y_test == y_pred)
print (float(accuracy) / X_test.shape[0])
