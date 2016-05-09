import unittest
from component.DeepFeedNet import DeepFeedNet
from autosklearn.pipeline.util import _test_classifier
import sklearn.metrics


class NetComponentTest(unittest.TestCase):
    def test_default_configuration(self):
        for i in range(10):
            predictions, targets = _test_classifier(DeepFeedNet, dataset='iris')
            acc_score = sklearn.metrics.accuracy_score(y_pred=predictions, y_true=targets)
            print(acc_score)
            self.assertAlmostEqual(0.58, acc_score)

    def test_default_configuration_binary(self):
        for i in range(10):
            predictions, targets = _test_classifier(DeepFeedNet,
                                                    make_binary=True)
            acc_score = sklearn.metrics.accuracy_score(y_pred=predictions, y_true=targets)
            print(acc_score)
            self.assertAlmostEqual(0.98, acc_score)

    def test_default_configuration_sparse(self):
        for i in range(10):
            predictions, targets = \
                _test_classifier(DeepFeedNet, sparse=True)
            acc_score = sklearn.metrics.accuracy_score(y_pred=predictions, y_true=targets)
            print(acc_score)
            self.assertAlmostEqual(0.4, acc_score)

    def test_default_configuration_multilabel(self):
        for i in range(10):
            predictions, targets = _test_classifier(DeepFeedNet,
                                                    make_multilabel=True)
            self.assertEqual(predictions.shape, (50, 3))
            ave_precision_score = sklearn.metrics.average_precision_score(targets, predictions)
            print(ave_precision_score)
            self.assertAlmostEqual(0.767777777778, ave_precision_score)

    def test_constrained_individual_configspace(self):
        # TODO: Test for fixed cs
        pass
