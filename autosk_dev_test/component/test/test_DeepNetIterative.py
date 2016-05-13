import unittest
import os
from component.DeepNetIterative import DeepNetIterative
from autosklearn.pipeline.util import _test_classifier_iterative_fit
import sklearn.metrics


class IterativeNetComponentTest(unittest.TestCase):
    def test_default_configuration(self):
        for i in range(10):
            predictions, targets = _test_classifier_iterative_fit(DeepNetIterative, dataset='iris')
            acc_score = sklearn.metrics.accuracy_score(y_pred=predictions, y_true=targets)
            print(acc_score)
            self.assertAlmostEqual(0.62, acc_score)

    def test_default_configuration_sparse(self):
        for i in range(10):
            predictions, targets = \
                _test_classifier_iterative_fit(DeepNetIterative, sparse=True)
            acc_score = sklearn.metrics.accuracy_score(y_pred=predictions, y_true=targets)
            print(acc_score)
            self.assertAlmostEqual(0.54, acc_score)
