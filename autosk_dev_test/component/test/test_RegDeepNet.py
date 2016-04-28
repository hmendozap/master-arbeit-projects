import unittest

from component.RegDeepNet import RegDeepNet
from autosklearn.pipeline.util import _test_regressor
import sklearn.metrics


class RegDeepNetComponentTest(unittest.TestCase):
    def test_default_configuration(self):
        for i in range(10):
            predictions, targets = _test_regressor(RegDeepNet)
            score_r2 = sklearn.metrics.r2_score(y_true=targets, y_pred=predictions)
            print(score_r2)
            self.assertAlmostEqual(0.4164, score_r2, places=1)
