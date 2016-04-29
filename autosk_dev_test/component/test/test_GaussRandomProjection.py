import unittest

from sklearn.linear_model import RidgeClassifier
from autosklearn.pipeline.components.feature_preprocessing.truncatedSVD import \
    TruncatedSVD
from component.GaussRandomProjection import GaussRandomProjection
from autosklearn.pipeline.util import _test_preprocessing, PreprocessingTestCase, \
    get_dataset
import sklearn.metrics


class TestGaussRandomProjectionComponent(PreprocessingTestCase):
    def test_default_configuration(self):
        transformation, original = _test_preprocessing(GaussRandomProjection)
        self.assertEqual(transformation.shape[0], original.shape[0])
        self.assertFalse((transformation == 0).all())

    def test_default_configuration_classify(self):
        for i in range(2):
            X_train, Y_train, X_test, Y_test = get_dataset(dataset='digits',
                                                           make_sparse=True)
            configuration_space = GaussRandomProjection.get_hyperparameter_search_space()
            default = configuration_space.get_default_configuration()
            preprocessor = GaussRandomProjection(random_state=1,
                                                 **{hp_name: default[hp_name]
                                                     for hp_name in
                                                     default if default[
                                                      hp_name] is not None})
            preprocessor.fit(X_train, Y_train)
            X_train_trans = preprocessor.transform(X_train)
            X_test_trans = preprocessor.transform(X_test)

            # fit a classifier on top
            classifier = RidgeClassifier()
            predictor = classifier.fit(X_train_trans, Y_train)
            predictions = predictor.predict(X_test_trans)
            accuracy = sklearn.metrics.accuracy_score(predictions, Y_test)
            self.assertAlmostEqual(accuracy, 0.44201578627808136, places=2)
