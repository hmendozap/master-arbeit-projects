"""
Created on April 12, 2016

@modified: Hector Mendoza
"""
import numpy as np
import theano
import theano.tensor as T
import theano.sparse as S
import lasagne

DEBUG = True


def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert inputs.shape[0] == targets.shape[0],\
           "The number of training points is not the same"
    if shuffle:
        indices = np.arange(inputs.shape[0])
        np.random.shuffle(indices)
    for start_idx in range(0, inputs.shape[0] - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]


class LogisticRegression(object):
    def __init__(self, input_shape=(100, 28*28), batch_size=100,
                 num_output_units=2, dropout_output=0.5, learning_rate=0.01,
                 lambda2=1e-4, momentum=0.9, beta1=0.9, beta2=0.9,
                 rho=0.95, solver="sgd", num_epochs=10, activation='relu',
                 lr_policy="fixed", gamma=0.01, power=1.0, epoch_step=1,
                 is_sparse=False, is_binary=False, is_regression=False,
                 is_multilabel=False):

        self.batch_size = batch_size
        self.input_shape = input_shape
        self.num_output_units = num_output_units
        self.dropout_output = T.cast(dropout_output, dtype=theano.config.floatX)
        self.momentum = T.cast(momentum, dtype=theano.config.floatX)
        self.learning_rate = np.asarray(learning_rate, dtype=theano.config.floatX)
        self.lambda2 = T.cast(lambda2, dtype=theano.config.floatX)
        self.beta1 = T.cast(beta1, dtype=theano.config.floatX)
        self.beta2 = T.cast(beta2, dtype=theano.config.floatX)
        self.rho = T.cast(rho, dtype=theano.config.floatX)
        self.num_epochs = num_epochs
        self.lr_policy = lr_policy
        self.gamma = np.asarray(gamma, dtype=theano.config.floatX)
        if power > 1.0:
            print('hyperparameter must be between 0 and 1')
            self.power = np.asarray(1.0, dtype=theano.config.floatX)
        else:
            self.power = np.asarray(power, dtype=theano.config.floatX)
        self.epoch_step = np.asarray(epoch_step, dtype=theano.config.floatX)
        self.is_binary = is_binary
        self.is_regression = is_regression
        self.is_multilabel = is_multilabel
        self.solver = solver
        self.activation = activation

        if is_sparse:
            input_var = S.csr_matrix('inputs', dtype='float32')
        else:
            input_var = T.matrix('inputs')

        if self.is_binary or self.is_multilabel or self.is_regression:
            target_var = T.matrix('targets')
        else:
            target_var = T.ivector('targets')

        if DEBUG:
            if self.is_binary:
                print("... using binary loss")
            if self.is_multilabel:
                print("... using multilabel prediction")
            if self.is_regression:
                print("... using regression loss")
            print("... building regressor")
            print input_shape
            print("... with number of epochs")
            print(num_epochs)

        self.network = lasagne.layers.InputLayer(shape=input_shape,
                                                 input_var=input_var)
        # Define output layer
        if self.is_regression:
            output_activation = lasagne.nonlinearities.linear
        elif self.is_binary or self.is_multilabel:
            output_activation = lasagne.nonlinearities.sigmoid
        else:
            output_activation = lasagne.nonlinearities.softmax

        self.network = lasagne.layers.DenseLayer(
                 lasagne.layers.dropout(self.network,
                                        p=self.dropout_output),
                 num_units=self.num_output_units,
                 W=lasagne.init.GlorotNormal(),
                 b=lasagne.init.Constant(),
                 nonlinearity=output_activation)

        prediction = lasagne.layers.get_output(self.network)

        if self.is_regression:
            loss_function = lasagne.objectives.squared_error
        elif self.is_binary or self.is_multilabel:
            loss_function = lasagne.objectives.binary_hinge_loss
        else:
            loss_function = lasagne.objectives.categorical_crossentropy

        loss = loss_function(prediction, target_var)

        # Aggregate loss mean function with l2 Regularization on all layers' params
        if self.is_regression or self.is_binary:
            loss = lasagne.objectives.aggregate(loss, mode='sum')
        else:
            loss = loss.mean(dtype=theano.config.floatX)
        l2_penalty = self.lambda2 * lasagne.regularization.regularize_network_params(
            self.network, lasagne.regularization.l2)
        loss += l2_penalty
        params = lasagne.layers.get_all_params(self.network, trainable=True)

        # Create the symbolic scalar lr for loss & updates function
        lr_scalar = T.scalar('lr', dtype=theano.config.floatX)

        if solver == "nesterov":
            updates = lasagne.updates.nesterov_momentum(loss, params,
                                                        learning_rate=lr_scalar,
                                                        momentum=self.momentum)
        elif solver == "adam":
            updates = lasagne.updates.adam(loss, params,
                                           learning_rate=lr_scalar,
                                           beta1=self.beta1, beta2=self.beta2)
        elif solver == "adadelta":
            updates = lasagne.updates.adadelta(loss, params,
                                               learning_rate=lr_scalar,
                                               rho=self.rho)
        elif solver == "adagrad":
            updates = lasagne.updates.adagrad(loss, params,
                                              learning_rate=lr_scalar)
        elif solver == "sgd":
            updates = lasagne.updates.sgd(loss, params,
                                          learning_rate=lr_scalar)
        elif solver == "momentum":
            updates = lasagne.updates.momentum(loss, params,
                                               learning_rate=lr_scalar,
                                               momentum=self.momentum)
        else:
            updates = lasagne.updates.sgd(loss, params,
                                          learning_rate=lr_scalar)

        if DEBUG:
            print("... compiling theano functions")
        self.train_fn = theano.function([input_var, target_var, lr_scalar],
                                        loss,
                                        updates=updates,
                                        allow_input_downcast=True,
                                        profile=False,
                                        on_unused_input='warn')
        self.update_function = self._policy_function()

    def _policy_function(self):
        epoch, gm, powr, step = T.scalars('epoch', 'gm', 'powr', 'step')
        if self.lr_policy == 'inv':
            decay = T.power(1+gm*epoch, -powr)
        elif self.lr_policy == 'exp':
            decay = gm ** epoch
        elif self.lr_policy == 'step':
            decay = T.switch(T.eq(T.mod_check(epoch, step), 0),
                             T.power(gm, T.floor_div(epoch, step)),
                             1.0)
        elif self.lr_policy == 'fixed':
            decay = T.constant(1.0, name='fixed', dtype=theano.config.floatX)

        return theano.function([gm, epoch, powr, step],
                               decay,
                               allow_input_downcast=True,
                               on_unused_input='ignore')

    def fit(self, X, y):
        # TODO: If batch size is bigger than available points
        # training is not executed.
        X = np.asarray(X, dtype=theano.config.floatX)
        y = np.asarray(y, dtype=theano.config.floatX)
        for epoch in range(self.num_epochs):
            train_err = 0
            train_batches = 0
            for inputs, targets in iterate_minibatches(X, y, self.batch_size, shuffle=True):
                train_err += self.train_fn(inputs, targets, self.learning_rate)
                train_batches += 1
            decay = self.update_function(self.gamma, epoch+1,
                                         self.power, self.epoch_step)
            self.learning_rate *= decay
            print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        return self

    def predict(self, X, is_sparse=False):
        predictions = self.predict_proba(X, is_sparse)
        if self.is_multilabel:
            return np.round(predictions)
        elif self.is_regression:
            return predictions
        else:
            return np.argmax(predictions, axis=1)

    def predict_proba(self, X, is_sparse=False):
        # TODO: Add try-except statements
        if is_sparse:
            X = S.basic.as_sparse_or_tensor_variable(X)
        predictions = lasagne.layers.get_output(self.network, X, deterministic=True).eval()
        if self.is_binary:
            return np.append(1 - predictions, predictions, axis=1)
        else:
            return predictions
