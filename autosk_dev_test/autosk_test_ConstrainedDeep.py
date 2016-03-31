# -*- encoding: utf-8 -*-

import numpy as np
# from autosklearn.pipeline.components.classification import add_classifier
import autosklearn.automl as autosk
# from component import AdamConstFeedNet
# from component import DeepFeedNet

# Use MNIST
dataset_dir = 

X_train = np.loadtxt(dataset_dir + 'train.data')
y_train = np.loadtxt(dataset_dir + 'train.solution')
y_train = np.argmax(y_train, axis=1)

# Load our training data
# X_train = np.load(dataset_dir + 'train.npy')
# y_train = np.load(dataset_dir + 'train_labels.npy')

# Create model
modl = autosk.AutoML(time_left_for_this_task=600, per_run_time_limit=90,
                     delete_tmp_folder_after_terminate=False,
                     tmp_dir='tmp/activation_nn', output_dir='tmp/autosk_out',
                     log_dir='tmp/autosk_log',
                     include_estimators=['DeepFeedNet'],
                     include_preprocessors=['NoPreprocessing'],
                     ensemble_size=0,
                     ensemble_nbest=0,
                     initial_configurations_via_metalearning=0,
                     seed=120,
                     ml_memory_limit=3072,
                     metadata_directory=None,
                     queue=None,
                     keep_models=False,
                     debug_mode=False,
                     resampling_strategy='partial-cv',
                     resampling_strategy_arguments={'folds': 10})

modl.fit(X_train, y_train, dataset_name='Activation_MNIST')
