import numpy as np
from sklearn.datasets import load_svmlight_file as lsf
from autosklearn.pipeline.components.classification import add_classifier
import autosklearn.automl as autosk
from component import DeepFeedNet

aad_dataset_dir = '../datasets/dataset_243/'
automl_dataset_dir = '../datasets/dataset_1000/'
libsvm_dataset = '../datasets/covtype.libsvm.binary'


X, y = lsf(libsvm_dataset, n_features=54)
train_size = int(X.shape[0] * 0.9)
X_train = X[:train_size]
y_train = y[:train_size] - 1

add_classifier(DeepFeedNet.DeepFeedNet)

# Create model
modl = autosk.AutoML(time_left_for_this_task=1800, seed=20, per_run_time_limit=180,
                     ensemble_nbest=1, ensemble_size=1,
                     ml_memory_limit=2048, resampling_strategy='holdout',
                     tmp_dir='/tmp/sparse_tmp', output_dir='/tmp/sparse_out',
                     delete_tmp_folder_after_terminate=False,
                     initial_configurations_via_metalearning=None,
                     include_preprocessors=['no_preprocessing'],
                     include_estimators=['feed_nn'])

modl.fit(X_train, y_train)

X_test = X[train_size:]
y_test = y[train_size:] - 1

# Only predict before getting scorin'
y_pred = modl.predict(X_test)

tot_score = modl.score(X_test, y_test)
print tot_score

# Comparison
accuracy = np.count_nonzero(y_test == y_pred)
print (float(accuracy) / X_test.shape[0])
