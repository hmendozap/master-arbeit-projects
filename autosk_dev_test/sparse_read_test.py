import numpy as np
from sklearn.datasets import load_svmlight_file as lsf
from autosklearn.pipeline.components.classification import add_classifier
from autosklearn.data import competition_data_manager as askdata
import autosklearn.automl as autosk
from component import DeepFeedNet

aad_dataset_dir = '../datasets/dataset_243/'
automl_dataset_dir = '/data/aad/automl_data/openml/293_acc/293_acc_'
libsvm_dataset = '../datasets/covtype.libsvm.binary'

# Also one need to size of features
X_list = askdata.sparse_file_to_sparse_list(automl_dataset_dir + 'train.data')
X_train = askdata.sparse_list_to_csr_sparse(X_list, nbr_features=54)

y_train = np.loadtxt(automl_dataset_dir + 'train.solution')

#X, y = lsf(libsvm_dataset, n_features=54)
#train_size = int(X.shape[0] * 0.9)
#X_train = X[:train_size]
#y_train = y[:train_size] - 1

add_classifier(DeepFeedNet.DeepFeedNet)

# Create model
modl = autosk.AutoML(time_left_for_this_task=1800, seed=20, per_run_time_limit=180,
                     ensemble_nbest=1, ensemble_size=1,
                     ml_memory_limit=2048, resampling_strategy='holdout',
                     tmp_dir='tmp/sparse_tmp', output_dir='tmp/sparse_out',
                     delete_tmp_folder_after_terminate=False,
                     initial_configurations_via_metalearning=None,
                     include_preprocessors=['no_preprocessing'],
                     include_estimators=['DeepFeedNet'])

modl.fit(X_train, y_train)

# Also one need to size of features
X_test_list = askdata.sparse_file_to_sparse_list(automl_dataset_dir + 'test.data')
X_test = askdata.sparse_list_to_csr_sparse(X_list, nbr_features=54)

y_test = np.loadtxt(automl_dataset_dir + 'test.solution')
#X_test = X[train_size:]
#y_test = y[train_size:] - 1

# Only predict before getting scorin'
y_pred = modl.predict(X_test)

tot_score = modl.score(X_test, y_test)
print(tot_score)

# Comparison
accuracy = np.count_nonzero(y_test == y_pred)
print(float(accuracy) / X_test.shape[0])
