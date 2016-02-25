"""
Created on Jul 22, 2015
Modified on Feb 1, 2016

@author: Aaron Klein
@modified: Hector Mendoza
"""
import numpy as np
import theano
import theano.tensor as T
import theano.sparse as S
import lasagne

DEBUG = True


def iterate_minibatches(inputs, targets, batchsize, num_updates, shuffle=False):
    assert inputs.shape[0] == targets.shape[0]
    if shuffle:
        indices = np.arange(inputs.shape[0])
        np.random.shuffle(indices)
    for start_idx in range(0, inputs.shape[0] - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]


class FeedForwardNet(object):
    def __init__(self, input_shape=(100, 28*28),
                 batch_size=100, num_layers=4, num_units_per_layer=(10, 10, 10),
                 dropout_per_layer=(0.5, 0.5, 0.5), std_per_layer=(0.005, 0.005, 0.005),
                 num_output_units=2, dropout_output=0.5, learning_rate=0.01,
                 momentum=0.9, beta1=0.9, beta2=0.9,
                 rho=0.95, solver="sgd", num_epochs=3, num_updates=10000,
                 is_sparse=False):

        self.batch_size = batch_size
        self.input_shape = input_shape
        self.num_layers = num_layers
        self.num_units_per_layer = num_units_per_layer
        self.dropout_per_layer = dropout_per_layer
        self.num_output_units = num_output_units
        self.dropout_output = dropout_output
        self.std_per_layer = std_per_layer
        self.momentum = momentum
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.rho = rho
        self.number_updates = num_updates
        self.num_epochs = num_epochs

        # TODO: Add correct theano shape constructor
        if is_sparse:
            input_var = S.csr_matrix('inputs', dtype='float32')
        else:
            input_var = T.dmatrix('inputs')
        target_var = T.lvector('targets')
        if DEBUG:
            print("... building network")
            print input_shape

        self.network = lasagne.layers.InputLayer(shape=input_shape,
                                                 input_var=input_var)
        # Define each layer
        for i in range(num_layers - 1):
            self.network = lasagne.layers.DenseLayer(
                 lasagne.layers.dropout(self.network,
                                        p=self.dropout_per_layer[i]),
                 num_units=self.num_units_per_layer[i],
                 W=lasagne.init.Normal(std=self.std_per_layer[i], mean=0),
                 b=lasagne.init.Constant(val=0.0),
                 nonlinearity=lasagne.nonlinearities.rectify)

        # Define output layer
        self.network = lasagne.layers.DenseLayer(
                 lasagne.layers.dropout(self.network, p=self.dropout_output),
                 num_units=self.num_output_units,
                 W=lasagne.init.GlorotNormal(),
                 b=lasagne.init.Constant(),
                 nonlinearity=lasagne.nonlinearities.softmax)

        prediction = lasagne.layers.get_output(self.network)
        loss = lasagne.objectives.categorical_crossentropy(prediction,
                                                           target_var)
        # Aggregate loss mean function
        loss = loss.mean()
        params = lasagne.layers.get_all_params(self.network, trainable=True)

        if solver == "nesterov":
            updates = lasagne.updates.nesterov_momentum(loss, params,
                                                        learning_rate=self.learning_rate,
                                                        momentum=self.momentum)
        elif solver == "adam":
            updates = lasagne.updates.adam(loss, params,
                                           learning_rate=self.learning_rate,
                                           beta1=self.beta1, beta2=self.beta2)
        elif solver == "adadelta":
            updates = lasagne.updates.adadelta(loss, params,
                                               learning_rate=self.learning_rate,
                                               rho=self.rho)
        elif solver == "adagrad":
            updates = lasagne.updates.adagrad(loss, params,
                                              learning_rate=self.learning_rate)
        elif solver == "sgd":
            updates = lasagne.updates.sgd(loss, params,
                                          learning_rate=self.learning_rate)
        elif solver == "momentum":
            updates = lasagne.updates.momentum(loss, params,
                                               learning_rate=self.learning_rate,
                                               momentum=self.momentum)
        else:
            updates = lasagne.updates.sgd(loss, params,
                                          learning_rate=self.learning_rate)

        # valid_prediction = lasagne.layers.get_output(self.network,
                                                     # deterministic=True)

        # valid_loss = lasagne.objectives.categorical_accuracy(
                                                        # valid_prediction,
                                                        # target_var)
        # valid_loss = valid_loss.mean()
        # valid_acc = T.mean(T.eq(T.argmax(valid_prediction, axis=1),
                                # target_var),
                                # dtype=theano.config.floatX)

        print("... compiling theano functions")
        self.train_fn = theano.function([input_var, target_var], loss,
                                        updates=updates,
                                        allow_input_downcast=True)

    def fit(self, X, y):
        for epoch in range(self.num_epochs):
            # TODO: Add exception RaiseError in shape
            train_err = 0
            train_batches = 0
            for batch in iterate_minibatches(X, y, self.batch_size, self.number_updates, shuffle=True):
                inputs, targets = batch
                train_err += self.train_fn(inputs, targets)
                train_batches += 1
            assert (train_batches == self.number_updates)
            print("  training error:\t\t{:.6f}".format(train_err))
            print("  training batches:\t\t{:.6f}".format(train_batches))
            print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        return self

    def predict(self, X, is_sparse=False):
        predictions = self.predict_proba(X, is_sparse)
        return np.argmax(predictions, axis=1)

    def predict_proba(self, X, is_sparse=False):
        # TODO: Add try-catch statements
        if is_sparse:
            X = S.basic.as_sparse_or_tensor_variable(X)
        predictions = lasagne.layers.get_output(self.network, X, deterministic=True).eval()
        return predictions
