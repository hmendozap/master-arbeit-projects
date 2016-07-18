import unittest
import numpy as np

from component.implementation.FeedForwardNet import FeedForwardNet


class TestFeedForwardNet(unittest.TestCase):
    dataset_dir = '/home/mendozah/workspace/datasets'

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
        model = FeedForwardNet(input_shape=(50, 7), batch_size=50,
                               learning_rate=0.1,
                               solver='nesterov',
                               lr_policy='inv',
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

    def test_policy_solver_comparison(self):
        model = FeedForwardNet(input_shape=(299, 7),
                               batch_size=299,
                               learning_rate=5.2324630436085886E-5,
                               num_layers=3,
                               num_units_per_layer=(4881, 29),
                               dropout_per_layer=(0.3041125564340035,
                                                  0.329849488011273),
                               dropout_output=0.20938047258156572,
                               std_per_layer=(1.1113852279881537E-4,
                                              0.001961480122126592),
                               lambda2=0.0038423973048742816,
                               momentum=0.7984896750906607,
                               solver='adam',
                               beta1=0.030996996347756028,
                               beta2=3.624204945904676E-4,
                               lr_policy='inv',
                               gamma=0.045100726594474436,
                               power=0.7620354736863749,
                               epoch_step=10,
                               num_epochs=49)
        model.fit(self.X_train, self.y_train)
        print("Model fitted")

        predicted_probability_matrix = model.predict_proba(self.X_test)
        expected_labels = np.argmax(predicted_probability_matrix, axis=1)
        predicted_labels = model.predict(self.X_test)

        # TODO: Add asserts and end lr calculations
        print("lr is {:.4E}".format(model.learning_rate))
        # self.assertTrue(0.1 == model.learning_rate)
        self.assertTrue((predicted_labels == expected_labels).all(), msg="Failed predicted probability")
        self.assertTrue((1 - predicted_probability_matrix.sum(axis=1) < 1e-3).all())

    def test_ranges(self):
        for i in range(10):
            self.test_policy_solver_comparison()
        print("==Done==")
