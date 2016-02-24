import sys
import time

import numpy as np
import matplotlib.pyplot as plt
import pyrfr.regression as rfreg

# Feature matrix contains one data point per row?
feats = np.array(np.load('data_matrix.npy'), order='C')

# Response on 1D Array, so a vector
errors = np.array(np.load('performance.npy'), order='C')

full_data = rfreg.mostly_continuous_data_container(feats.shape[1])

# Add Solver categorical feature (As of 19.2 the categorical feature is disabled)
full_data.set_type_of_feature(21, 6)

full_data.import_numpy_arrays(feats, errors)

#for i in range(full_data.num_features()):
#    print("feature {} is now of type {}".format(i, full_data.get_type_of_feature(i)))

if np.allclose(full_data.export_responses(), errors) and np.allclose(full_data.export_features(), feats):
    print("Import data was successful")

print("Number of features: {}".format(full_data.num_features()))
print("number of data points: {}".format(full_data.num_data_points()))

# RSS = Residual Sums of Squares
forst = rfreg.binary_rss()
forst.num_trees = 512

# mostly_continuous means around a half of feats to be continuous
# RF Parameters
forst.seed = 10
forst.do_bootstrapping = True
forst.num_data_points_per_tree = 0  # 0 -> same number as data points
# First try with all features, maybe having one third and more trees should work
forst.max_features = 6  # 0 -> means all features
forst.min_samples_to_split = 0  # 0 -> means split until pure
forst.min_samples_in_leaf = 0  # 0 -> no restriction
forst.max_depth = 1024  # being a regression forest maybe we need more
forst.epsilon_purity = 1e-8  # when checking for purity, the points can differ by this epsilon

forst.fit(full_data)

#forst.save_latex_representation('tex_representantions/rfr_')

# After fitting we predict the value of error based on the change of one hyperparameter
# Return a tuple with [0] -> mean, [1] -> std.
print(forst.predict(feats[0]))

# Now we save the forest to disk
# forst.save_to_binary_file(b'param_curves_allNumerical.bin')

# Matrix of one parameter variation. In this case learning rate, feat[10]
# We will use the configuration with the best cross-validation error
# and then change one parameter.
nsamples = 1000
prediction_window = np.linspace(0.001, 0.11, num=nsamples)
min_inx = np.argmin(errors)  # This assumes that responses and matrix are aligned
best_config = np.tile(feats[min_inx], (nsamples, 1))
best_config[:, 10] = prediction_window

print("Best X-Validation error is {}".format(errors[min_inx]))
print("Best predicted X-Validation error is {}".format(forst.predict(feats[min_inx])))
#print("Best configuration based on error is: {}".format(feats[min_inx]))

pred_performance = np.zeros((nsamples, 2))

for i in range(nsamples):
    pred_performance[i] = forst.predict(best_config[i])

mean_pred_performance = pred_performance[:, 0]
std_pred_performance = pred_performance[:, 1]

plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (12.0, 9.0)
#plt.fill_between(prediction_window, mean_pred_performance+std_pred_performance, mean_pred_performance-std_pred_performance,  alpha=0.4)
plt.plot(prediction_window, mean_pred_performance, 'r')
plt.scatter(feats[min_inx, 10], errors[min_inx] )
plt.xlabel('learning rate values')
plt.ylabel('Predicted X-Validation error')
plt.title('Expected error based on learning rate variation')
plt.show()
plt.savefig('fig_lr_cat.png')

