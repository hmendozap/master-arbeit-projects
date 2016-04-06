import unittest
import os

from component.DeepFeedNet import DeepFeedNet
from component.ConstrainedFeedNet import ConstrainedFeedNet, AdamConstFeedNet
from autosklearn.pipeline.util import _test_classifier, _test_classifier_predict_proba

import sklearn.metrics
import numpy as np
import theano.sparse as S


class NetComponentTest(unittest.TestCase):
    home_dir = os.environ['HOME']
    dataset_dir = 

    X_train = np.load(dataset_dir + 'train.npy')
    y_train = np.load(dataset_dir + 'train_labels.npy')
    X_test = np.load(dataset_dir + 'test.npy')
    y_test = np.load(dataset_dir + 'test_labels.npy')

    def test_default_configuration(self):
        for i in range(10):
            predictions, targets = _test_classifier(DeepFeedNet, dataset='iris')
            print sklearn.metrics.accuracy_score(predictions, targets)
            # self.assertAlmostEqual(0.96,
            #                       sklearn.metrics.accuracy_score(predictions, targets))

    def test_default_configuration_space(self):

        for i in range(10):
            configuration_space = DeepFeedNet.get_hyperparameter_search_space()
            default = configuration_space.get_default_configuration()

            cls = DeepFeedNet(**{hp_name: default[hp_name] for hp_name in
                               default if default[hp_name] is not None})

            cls = cls.fit(self.X_train, self.y_train)
            prediction = cls.predict_proba(self.X_test)
            print sklearn.metrics.log_loss(self.y_test, prediction)

            # TODO: Ask Matthias about the value
            # self.assertAlmostEqual(sklearn.metrics.log_loss(y_test, prediction),
            #                       0.68661222917147913)

    def test_default_configuration_binary(self):
        """
        Test of default config feed net in
        a binary classification problem
        """

        # Don't know if should be external loading
        for i in range(10):
            predictions, targets = _test_classifier(DeepFeedNet,
                                                    make_binary=True)
            self.assertTrue(all(targets == predictions))


    def test_default_sparse_configuration(self):

        dataset_name = '../../data/covtype.test'
        from sklearn.datasets import load_svmlight_file as lsf

        X, y = lsf(dataset_name, n_features=54)
        train_size = int(X.shape[0] * 0.9)
        X_train = X[:train_size]
        y_train = y[:train_size] - 1
        X_test = X[train_size:]
        y_test = y[train_size:] - 1

        for i in range(10):
            configuration_space = DeepFeedNet.get_hyperparameter_search_space()
            default = configuration_space.get_default_configuration()

            cls = DeepFeedNet(**{hp_name: default[hp_name] for hp_name in
                               default if default[hp_name] is not None})

            cls = cls.fit(X_train, y_train)

            # TODO: Review with Aaron this implementation
            #x_sp = S.basic.as_sparse_or_tensor_variable(X_test)
            prediction = cls.predict_proba(X_test)
            print sklearn.metrics.log_loss(y_test, prediction)

    def test_constrained_default_configuration_space(self):
        for i in range(10):
            configuration_space = AdamConstFeedNet.get_hyperparameter_search_space()
            default = configuration_space.get_default_configuration()

            kls = AdamConstFeedNet(**{hp_name: default[hp_name] for hp_name in
                                     default if default[hp_name] is not None})
            kls.fit(self.X_train, self.y_train)
            prediction = kls.predict(self.X_test)
            print sklearn.metrics.log_loss(self.y_test, prediction)

    def test_default_configuration_multilabel(self):
        for i in range(10):
            predictions, targets = _test_classifier(DeepFeedNet,
                                                    make_multilabel=True)
            print(sklearn.metrics.
                  average_precision_score(targets, predictions))

    def test_constrained_individual_configspace(self):
        # TODO: Test for fixed cs
        pass
