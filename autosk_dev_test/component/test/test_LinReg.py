import unittest

from autosklearn.pipeline.components.regression.LinReg import LinReg
from autosklearn.pipeline.implementations.LogisticRegression import LogisticRegression
from autosklearn.pipeline.util import _test_regressor

import sklearn.metrics
import numpy as np


class LinRegComponentTest(unittest.TestCase):
    def test_default_configuration(self):
        for i in range(10):
            predictions, targets =_test_regressor(LinReg)
            print(sklearn.metrics.r2_score(y_true=targets, y_pred=predictions))


    def test_regression_deep_implementation(self):
        """
        Test of regression implementation for feed forward networks
        """
        X_train = np.loadtxt('/home/mendozah/workspace/datasets/yolanda_set/yolo_data.txt')
        y_train = np.loadtxt('/home/mendozah/workspace/datasets/yolanda_set/yolo_sol.txt')
        y_train = y_train[:, np.newaxis]
        model = LogisticRegression(input_shape=(100, 100), batch_size=100, activation='linear',
                                   lambda2=0.0005, dropout_output=0.79399,
                                   learning_rate=5.6e-7, num_output_units=1, solver='adam',
                                   num_epochs=200, is_regression=True)
        model.fit(X_train, y_train)
        X_test = np.loadtxt('/home/mendozah/workspace/datasets/yolanda_set/test_data.txt')
        y_test = np.loadtxt('/home/mendozah/workspace/datasets/yolanda_set/test_sol.txt')
        y_test = y_test[:, np.newaxis]
        prediction = model.predict(X_test)
        import sklearn.metrics
        print(sklearn.metrics.r2_score(y_true=y_test, y_pred=prediction))
        print("Model fitted")