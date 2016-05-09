import unittest

from component.RegDeepNet import RegDeepNet
from autosklearn.pipeline.util import _test_regressor
import sklearn.metrics


class RegDeepNetComponentTest(unittest.TestCase):
    def test_default_configuration(self):
        for i in range(10):
            predictions, targets = _test_regressor(RegDeepNet)
            R2score = sklearn.metrics.r2_score(y_true=targets, y_pred=predictions)
            print(R2score)
            self.assertAlmostEqual(0.43, R2score)

