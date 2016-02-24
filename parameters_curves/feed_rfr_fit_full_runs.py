# -*- coding: utf-8 -*-
"""
Create 20-02-2016
@author: Hector Mendoza

Python script that loads configurations from all runs generated
by SMAC without auto-sklearn preprocessor methods
"""

import numpy as np
import matplotlib.pyplot as plt
import pyrfr.regression as rfreg

# Feature matrix contains one data point per row?
feats = np.array(np.load('full_runs_data.npy'), order='C')

# Response on 1D Array, so a vector
response = np.array(np.load('full_response.npy'), order='C')

full_data = rfreg.mostly_continuous_data_container(feats.shape[1])

# Add Solver & Number of layers as categorical features
full_data.set_type_of_feature(21, 6)
full_data.set_type_of_feature(12, 7)
full_data.import_numpy_arrays(feats, response)

if np.allclose(full_data.export_responses(), response) and np.allclose(full_data.export_features(), feats):
    print("Import data was successful")

print("Number of features: {}".format(full_data.num_features()))
print("number of data points: {}".format(full_data.num_data_points()))

# RSS = Residual Sums of Squares
forst = rfreg.binary_rss()
forst.num_trees = 512

forst.seed = 12
forst.do_bootstrapping = True
forst.num_data_points_per_tree = 0  # 0 -> same number as data points
forst.max_features = 0  #  feats.shape[1]//2  # 0 -> means all features
forst.min_samples_to_split = 0  # 0 -> means split until pure
forst.min_samples_in_leaf = 0  # 0 -> no restriction
forst.max_depth = 1024  # being a regression forest maybe we need more
forst.epsilon_purity = 1e-8  # when checking for purity, the points can differ by this epsilon

forst.fit(full_data)

#forst.save_latex_representation('tex_representantions/rfr_')

print(forst.predict(feats[1]))

# Now we save the forest to disk
file_to_save = b'fitted_forests/parameter_fitting_run.bin'
forst.save_to_binary_file(file_to_save)

# Matrix of one parameter variation. In this case learning rate, feat[10]
# We will use the configuration with the best cross-validation error
# and then change one parameter.
nsamples = 5000
prediction_window = np.linspace(1e-4, 5e-3, num=nsamples)
min_inx = 217
#min_inx = np.argmin(response)
best_config = np.tile(feats[min_inx], (nsamples, 1))
best_config[:, 10] = prediction_window

print("Best X-Validation error is {}".format(response[min_inx]))
print("Best predicted X-Validation error is {}".format(forst.predict(feats[min_inx])))

pred_performance = np.zeros((nsamples, 2))

for i in range(nsamples):
    pred_performance[i] = forst.predict(best_config[i])

mean_pred_performance = pred_performance[:, 0]
std_pred_performance = pred_performance[:, 1]

plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (12.0, 9.0)
plt.fill_between(prediction_window, mean_pred_performance+std_pred_performance,
                 mean_pred_performance-std_pred_performance,  alpha=0.4)
plt.plot(prediction_window, mean_pred_performance, 'r')
plt.scatter(feats[217, 10], response[217])
plt.xlabel('learning rate values')
plt.ylabel('Predicted X-Validation error')
plt.title(u'Expected validation error based on learning '
          u'rate variation\nwithout preprocessing')
plt.show()
plt.savefig('special_learning_rate_prediction.png')

