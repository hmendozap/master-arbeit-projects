import unittest
from autosklearn.pipeline.components.classification.LogReg import LogReg
from autosklearn.pipeline.util import _test_classifier
import sklearn.metrics


class LogRegComponentTest(unittest.TestCase):

    def test_default_configuration(self):
        for i in range(10):
            predictions, targets = _test_classifier(LogReg, dataset='iris')
            print(sklearn.metrics.accuracy_score(y_pred=predictions, y_true=targets))

    def test_default_configuration_binary(self):
        for i in range(10):
            predictions, targets = _test_classifier(LogReg,
                                                    make_binary=True)
            print(sklearn.metrics.accuracy_score(y_true=targets,
                                                 y_pred=predictions))

    def test_default_configuration_multilabel(self):
        for i in range(10):
            predictions, targets = _test_classifier(LogReg,
                                                    make_multilabel=True)
            print(sklearn.metrics.average_precision_score(y_true=targets,
                                                          y_score=predictions))
