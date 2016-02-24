# -*- enconding: utf-8 -*-
import unittest

import numpy as np
import pandas as pd
import ConfigReader as cr
from component.implementation import FeedForwardNet as ffnet


# Load our training data
dataset_dir = '/home/mendozah/workspace/datasets/dataset_1128/'

X_train = np.load(dataset_dir + 'train.npy')
y_train = np.load(dataset_dir + 'train_labels.npy')
X_test = np.load(dataset_dir + 'test.npy')
y_test = np.load(dataset_dir + 'test_labels.npy')

# Load all configurations from a dataset
data_directory = '/mhome/mendozah/nn_prepro_Experiment/results/experiment/'
dataset = '1128_bac'  # Only local available dataset
preprocessing_method = 'no_preprocessing'  # Avoid preprocessing differences

reader = cr.ConfigReader(data_dir=data_directory, scenario=dataset)
configs_df = reader.load_run_configs(preprocessor=preprocessing_method)

# Filter configurations
dif_configs = 10
selection_mask = configs_df['response'] == configs_df['response'].min()
best_df = configs_df[selection_mask].sample(n=dif_configs, replace=False)
print(best_df.index)
#best_df = configs_df.ix[599]

# Init the ffnet with the "best cv error configuration found by SMAC"
for i in range(dif_configs):
    configuration = best_df.iloc[i, 6:]
    #configuration = best_df.iloc[6:]
    print(configuration)
    input_shape = (configuration.batch_size, X_train.shape[1])
    dropout_layers = tuple(configuration.iloc[3:9].values.astype(np.float32))
    units_layers = tuple(configuration.iloc[13:19].values)
    std_layers = tuple(configuration.iloc[22:28].values.astype(np.float32))
    deepnn = ffnet.FeedForwardNet(input_shape=input_shape,
                                  batch_size=configuration.batch_size,
                                  beta1=configuration.beta1,
                                  beta2=configuration.beta2,
                                  momentum=configuration.momentum,
                                  num_epochs=configuration.number_epochs,
                                  solver=configuration.solver,
                                  learning_rate=configuration.learning_rate,
                                  rho=configuration.rho,
                                  num_layers=configuration.num_layers,
                                  num_units_per_layer=units_layers,
                                  dropout_per_layer=dropout_layers,
                                  dropout_output=configuration.dropout_output,
                                  std_per_layer=std_layers)

    deepnn.fit(X_train, y_train)

    # Only predict before getting scoring
    y_predicted = deepnn.predict(X_test)
    del deepnn

    # Comparison
    accuracy = np.count_nonzero(y_test == y_predicted)
    print("TEST ERROR IS: ")
    print(1.0 - (float(accuracy) / X_test.shape[0]))
