# -*- enconding: utf-8 -*-
from __future__ import print_function
import unittest
import numpy as np
import autosklearn
import os

from autosklearn.metrics import acc_metric


class BinaryClassificationScoreTest(unittest.TestCase):
    def test_binary_score(self):
        """
        Test fix for binary classification prediction
        taking the index 1 of second dimension in prediction matrix
        """
        if self.travis:
            self.skipTest('This test does currently not run on travis-ci. '
                          'Make sure it runs locally on your machine!')

        output = os.path.join(self.test_dir, '..', '.tmp_test_fit')
        self._setUp(output)

        X_train, Y_train, X_test, Y_test = putil.get_dataset('digits(n_class=2)')
        automl = autosklearn.automl.AutoML(output, output, 15, 15)
        automl.fit(X_train, Y_train)

        score = automl.score(X_test, Y_test)
        # self.assertGreaterEqual(score, 0.0)
        self.assertEqual(automl._task, BINARY_CLASSIFICATION)

        del automl
        self._tearDown(output)
