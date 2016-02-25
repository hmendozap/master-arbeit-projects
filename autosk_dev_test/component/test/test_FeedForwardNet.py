import unittest
import numpy as np

from component.implementation.FeedForwardNet import FeedForwardNet


class TestFeedForwardNet(unittest.TestCase):
    def test_initial_feed_implementation(self):
        """
        Test of initial implementation of simple run in
        NN Feed Forward architecture on Theano and Lasagne
        """

        dataset_dir = '/home/mendozah/workspace/datasets/dataset_728/'

        # Take training and data from binary classification
        X_train = np.load(dataset_dir + 'train.npy')
        y_train = np.load(dataset_dir + 'train_labels.npy')
        X_test = np.load(dataset_dir + 'test.npy')
        y_test = np.load(dataset_dir + 'test_labels.npy')

        # The input shape is using the batch size
        model = FeedForwardNet(input_shape=(50, 7), batch_size=50, num_epochs=2)

        model.fit(X_train, y_train)
        print("Model fitted")

        predicted_probability_matrix = model.predict_proba(X_test)
        expected_labels = np.argmax(predicted_probability_matrix, axis=1)
        predicted_labels = model.predict(X_test)
        accuracy = np.count_nonzero(y_test == predicted_labels)
        print(float(accuracy) / float(X_test.shape[0]))

        self.assertTrue((predicted_labels == expected_labels).all(), msg="Failed predicted probability")
        self.assertTrue((1 - predicted_probability_matrix.sum(axis=1) < 1e-3).all())
