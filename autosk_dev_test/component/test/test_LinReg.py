import unittest

from component.LinReg import LinReg
from autosklearn.pipeline.util import _test_regressor
import sklearn.metrics


class LinRegComponentTest(unittest.TestCase):
    def test_default_configuration(self):
        for i in range(10):
            predictions, targets = _test_regressor(LinReg)
            score_r2 = sklearn.metrics.r2_score(y_true=targets, y_pred=predictions)
            print(score_r2)
            # self.assertAlmostEqual(0.1212, score_r2)
