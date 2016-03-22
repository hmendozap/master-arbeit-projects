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


class FeedForwardNet(object):
    def __init__(self, input_shape=(100, 28*28),
                 batch_size=100, num_layers=4, num_units_per_layer=(10, 10, 10),
                 dropout_per_layer=(0.5, 0.5, 0.5), std_per_layer=(0.005, 0.005, 0.005),
                 num_output_units=2, dropout_output=0.5, learning_rate=0.01,
                 lambda2=1e-4, momentum=0.9, beta1=0.9, beta2=0.9,
                 rho=0.95, solver="sgd", num_epochs=2,
                 lr_policy="fixed", gamma=0.01, power=1.0, epoch_step=1,
                 is_sparse=False, is_binary=False):

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
        self.lambda2 = lambda2
        self.beta1 = beta1
        self.beta2 = beta2
        self.rho = rho
        # self.number_updates = number_updates
        self.num_epochs = num_epochs
        self.lr_policy = lr_policy
        self.gamma = gamma
        if power > 1.0:
            print('hyperparameter must be between 0 and 1')
            self.power = 1.0
        else:
            self.power = power
        self.epoch_step = epoch_step
        self.is_binary = is_binary
        self.solver = solver

        if is_sparse:
            input_var = S.csr_matrix('inputs', dtype='float32')
        else:
            input_var = T.matrix('inputs')

        if self.is_binary:
            target_var = T.lmatrix('targets')
        else:
            target_var = T.lvector('targets')

        if DEBUG:
            if self.is_binary:
                print("... using binary loss")
            print("... building network")
            print input_shape
            print("... with number of epochs")
            print(num_epochs)

        self.network = lasagne.layers.InputLayer(shape=input_shape,
                                                 input_var=input_var)

        # Choose hidden activation function

        # Define each layer
        for i in range(num_layers - 1):
            self.network = lasagne.layers.DenseLayer(
                 lasagne.layers.dropout(self.network,
                                        p=self.dropout_per_layer[i]),
                 num_units=self.num_units_per_layer[i],
                 W=lasagne.init.Normal(std=self.std_per_layer[i], mean=0),
                 b=lasagne.init.Constant(val=0.0),
                 nonlinearity=lasagne.nonlinearities.rectify)

        # Define output layer and nonlinearity of last layer
        if self.is_binary:
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

        if self.is_binary:
            loss_function = lasagne.objectives.binary_hinge_loss
        else:
            loss_function = lasagne.objectives.categorical_crossentropy

        loss = loss_function(prediction, target_var)

        # Aggregate loss mean function with l2 Regularization on all layers' params
        loss = loss.mean()
        l2_penalty = self.lambda2 * lasagne.regularization.regularize_network_params(
            self.network, lasagne.regularization.l2)
        loss += l2_penalty
        params = lasagne.layers.get_all_params(self.network, trainable=True)

        # Create the symbolic scalar lr for loss & updates function
        # EXCEPT for adam as it creates its own lr steps (only pass initial lr)
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

        # Validation was removed, as auto-sklearn handles that, if this net
        # is to be used independently, validation accuracy has to be included
        if DEBUG:
            print("... compiling theano functions")
        self.train_fn = theano.function([input_var, target_var, lr_scalar],
                                        loss,
                                        updates=updates,
                                        allow_input_downcast=True,
                                        profile=False,
                                        on_unused_input='warn')
        self.update_function = self._policy_function()

    def _choose_activation(self):
        pass

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
                               on_unused_input='ignore')

    def fit(self, X, y):
        for epoch in range(self.num_epochs):
            train_err = 0
            train_batches = 0
            for batch in iterate_minibatches(X, y, self.batch_size, shuffle=True):
                inputs, targets = batch
                train_err += self.train_fn(inputs, targets, self.learning_rate)
                train_batches += 1
            decay = self.update_function(self.gamma, epoch+1,
                                         self.power, self.epoch_step)
            self.learning_rate *= decay
            print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        return self

    def predict(self, X, is_sparse=False):
        predictions = self.predict_proba(X, is_sparse)
        if not self.is_binary:
            return np.argmax(predictions, axis=1)
        else:
            return np.rint(predictions)

    def predict_proba(self, X, is_sparse=False):
        # TODO: Add try-except statements
        if is_sparse:
            X = S.basic.as_sparse_or_tensor_variable(X)
        predictions = lasagne.layers.get_output(self.network, X, deterministic=True).eval()
        return predictions
