import unittest
import numpy as np

from component.implementation.FeedForwardNet import FeedForwardNet
from component.implementation.BinaryFeedForwardNet import BinaryFeedForwardNet


class TestFeedForwardNet(unittest.TestCase):
    dataset_dir = '/home/mendozah/workspace/' \
                      'datasets/dataset_728/'

    X_train = np.load(dataset_dir + 'train.npy')
    y_train = np.load(dataset_dir + 'train_labels.npy')
    X_test = np.load(dataset_dir + 'test.npy')
    y_test = np.load(dataset_dir + 'test_labels.npy')

    def test_initial_feed_implementation(self):
        """
        Test of initial implementation of simple run in
        NN Feed Forward architecture on Theano and Lasagne
        """
        model = FeedForwardNet(input_shape=(100, 7), batch_size=100,
                               learning_rate=0.1,
                               num_epochs=20)

        model.fit(self.X_train, self.y_train)
        print("Model fitted")

        predicted_probability_matrix = model.predict_proba(self.X_test)
        expected_labels = np.argmax(predicted_probability_matrix, axis=1)
        predicted_labels = model.predict(self.X_test)
        accuracy = np.count_nonzero(self.y_test == predicted_labels)
        print(float(accuracy) / float(self.X_test.shape[0]))

        self.assertTrue((predicted_labels == expected_labels).all(), msg="Failed predicted probability")
        self.assertTrue((1 - predicted_probability_matrix.sum(axis=1) < 1e-3).all())

    def test_lr_policies(self):
        model = FeedForwardNet(input_shape=(100, 7), batch_size=100,
                               learning_rate=0.1,
                               solver='adam',
                               lr_policy='step',
                               gamma=0.1,
                               power=0.75,
                               epoch_step=4,
                               num_epochs=20)
        model.fit(self.X_train, self.y_train)
        print("Model fitted")

        predicted_probability_matrix = model.predict_proba(self.X_test)
        expected_labels = np.argmax(predicted_probability_matrix, axis=1)
        predicted_labels = model.predict(self.X_test)
        accuracy = np.count_nonzero(self.y_test == predicted_labels)
        print(float(accuracy) / float(self.X_test.shape[0]))

        # TODO: Add asserts and end lr calculations
        # self.assertTrue(0.1 == model.learning_rate)
        self.assertTrue((predicted_labels == expected_labels).all(), msg="Failed predicted probability")
        self.assertTrue((1 - predicted_probability_matrix.sum(axis=1) < 1e-3).all())

        pass

    def test_binary_feed_implementation(self):

        """
        Test of binary (logistic) implementation of
        feed forward neural network with sigmodial output
        and binary cross entropy (BEC)
        :return:
        """
        dataset_dir = '/home/hmendoza/workspace/master_arbeit/' \
                      'auto-deep/datasets/dataset_728/'

        # Take training and data from binary classification
        X_train = np.load(dataset_dir + 'train.npy')
        y_train = np.load(dataset_dir + 'train_labels.npy')
        y_train = y_train[:, np.newaxis]
        X_test = np.load(dataset_dir + 'test.npy')
        y_test = np.load(dataset_dir + 'test_labels.npy')
        y_test = y_test[:, np.newaxis]

        model = BinaryFeedForwardNet(input_shape=(105, 7),
                                     batch_size=105,
                                     num_epochs=27,
                                     num_layers=4,
                                     num_units_per_layer=(274, 882, 6095),
                                     num_output_units=1,
                                     dropout_per_layer=(0.29848005310479914, 0.9133299770027523, 0.5832905399945898),
                                     dropout_output=0.9831436299733602,
                                     learning_rate=0.6224122392867175,
                                     solver='adagrad')

        model.fit(X_train, y_train)
        print("Model fitted")

        predicted_probability_matrix = model.predict_proba(X_test)
        predicted_probability_matrix[predicted_probability_matrix >= 0.5] = 1
        predicted_probability_matrix[predicted_probability_matrix < 0.5] = 0
        expected_labels = predicted_probability_matrix
        predicted_labels = model.predict(X_test)[:, np.newaxis]
        accuracy = np.count_nonzero(y_test == predicted_labels)
        print(float(accuracy) / float(X_test.shape[0]))

        self.assertTrue((predicted_labels == expected_labels).all(), msg="Failed predicted probability")
        self.assertTrue((1 - predicted_probability_matrix.sum(axis=1) < 1e-3).all())
