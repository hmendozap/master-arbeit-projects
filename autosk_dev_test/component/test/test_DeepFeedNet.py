import unittest

from component.DeepFeedNet import DeepFeedNet
from component.ConstrainedFeedNet import ConstrainedFeedNet, AdamConstFeedNet
from autosklearn.pipeline.util import _test_classifier, _test_classifier_predict_proba

import sklearn.metrics


class NetComponentTest(unittest.TestCase):
    def test_default_configuration(self):
        for i in range(10):
            predictions, targets = _test_classifier(DeepFeedNet, dataset='iris')
            print sklearn.metrics.accuracy_score(predictions, targets)
            # self.assertAlmostEqual(0.96,
            #                       sklearn.metrics.accuracy_score(predictions, targets))

    def test_default_sparse_configuration(self):
        for i in range(10):
            predictions, targets = _test_classifier(DeepFeedNet, sparse='True')
            acc_score = sklearn.metrics.accuracy_score(y_pred=predictions,
                                                       y_true=targets)
            # self.assertAlmostEqual(0.3, acc_score, places=2)

    def test_constrained_default_configuration_space(self):
        for i in range(10):
            predictions, targets = _test_classifier(AdamConstFeedNet,
                                                    dataset='iris')
            acc_score = sklearn.metrics.accuracy_score(y_true=targets,
                                                       y_pred=predictions)
            print(acc_score)

    def test_constrained_individual_configspace(self):
        # TODO: Test for fixed cs
        pass
