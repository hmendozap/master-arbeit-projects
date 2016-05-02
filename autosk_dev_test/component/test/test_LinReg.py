import unittest

from component.LinReg import LinReg
from autosklearn.pipeline.util import _test_regressor
import sklearn.metrics


class LinRegComponentTest(unittest.TestCase):
    def test_default_configuration(self):
        for i in range(10):
            predictions, targets =_test_regressor(LinReg)
            R2score = sklearn.metrics.r2_score(y_true=targets, y_pred=predictions)
            print(R2score)
            self.assertAlmostEqual(0.1212, R2score, places=1)
