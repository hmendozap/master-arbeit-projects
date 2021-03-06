import numpy as np
import scipy.sparse as sp

from HPOlibConfigSpace.configuration_space import ConfigurationSpace
from HPOlibConfigSpace.conditions import EqualsCondition, InCondition, AndConjunction
from HPOlibConfigSpace.hyperparameters import UniformFloatHyperparameter, \
    UniformIntegerHyperparameter, CategoricalHyperparameter, Constant

from autosklearn.pipeline.components.base import AutoSklearnClassificationAlgorithm
from autosklearn.pipeline.constants import *


class DeepFeedNet(AutoSklearnClassificationAlgorithm):

    def __init__(self, number_updates, batch_size, num_layers, num_units_layer_1,
                 dropout_layer_1, dropout_output, std_layer_1,
                 learning_rate, solver, lambda2,
                 num_units_layer_2=10, num_units_layer_3=10, num_units_layer_4=10,
                 num_units_layer_5=10, num_units_layer_6=10,
                 dropout_layer_2=0.5, dropout_layer_3=0.5, dropout_layer_4=0.5,
                 dropout_layer_5=0.5, dropout_layer_6=0.5,
                 weight_init_1='he_normal', weight_init_2='he_normal', weight_init_3='he_normal',
                 weight_init_4='he_normal', weight_init_5='he_normal', weight_init_6='he_normal',
                 activation_layer_1='relu', activation_layer_2='relu', activation_layer_3='relu',
                 activation_layer_4='relu', activation_layer_5='relu', activation_layer_6='relu',
                 leakiness_layer_1=1./3., leakiness_layer_2=1./3., leakiness_layer_3=1./3.,
                 leakiness_layer_4=1./3., leakiness_layer_5=1./3., leakiness_layer_6=1./3.,
                 tanh_alpha_layer_1=2./3., tanh_alpha_layer_2=2./3., tanh_alpha_layer_3=2./3.,
                 tanh_alpha_layer_4=2./3., tanh_alpha_layer_5=2./3., tanh_alpha_layer_6=2./3.,
                 tanh_beta_layer_1=1.7159, tanh_beta_layer_2=1.7159, tanh_beta_layer_3=1.7159,
                 tanh_beta_layer_4=1.7159, tanh_beta_layer_5=1.7159, tanh_beta_layer_6=1.7159,
                 std_layer_2=0.005, std_layer_3=0.005, std_layer_4=0.005,
                 std_layer_5=0.005, std_layer_6=0.005,
                 momentum=0.99, beta1=0.9, beta2=0.99, rho=0.95,
                 lr_policy='fixed', gamma=0.01, power=1.0, epoch_step=1,
                 random_state=None):
        self.number_updates = number_updates
        self.batch_size = batch_size
        # Hacky implementation of condition on number of layers
        self.num_layers = ord(num_layers) - ord('a')
        self.dropout_output = dropout_output
        self.learning_rate = learning_rate
        self.lr_policy = lr_policy
        self.lambda2 = lambda2
        self.momentum = momentum
        self.beta1 = 1-beta1
        self.beta2 = 1-beta2
        self.rho = rho
        self.solver = solver
        self.gamma = gamma
        self.power = power
        self.epoch_step = epoch_step

        # Empty features and shape
        self.n_features = None
        self.input_shape = None
        self.m_issparse = False
        self.m_isbinary = False
        self.m_ismultilabel = False

        # To avoid eval call. Could be done with **karws
        args = locals()
        self.num_units_per_layer = []
        self.dropout_per_layer = []
        self.activation_per_layer = []
        self.weight_init_layer = []
        self.std_per_layer = []
        self.leakiness_per_layer = []
        self.tanh_alpha_per_layer = []
        self.tanh_beta_per_layer = []
        for i in range(1, self.num_layers):
            self.num_units_per_layer.append(int(args.get("num_units_layer_" + str(i))))
            self.dropout_per_layer.append(float(args.get("dropout_layer_" + str(i))))
            self.activation_per_layer.append(args.get("activation_layer_" + str(i)))
            self.weight_init_layer.append(args.get("weight_init_" + str(i)))
            self.std_per_layer.append(float(args.get("std_layer_" + str(i))))
            self.leakiness_per_layer.append(float(args.get("leakiness_layer_" + str(i))))
            self.tanh_alpha_per_layer.append(float(args.get("tanh_alpha_layer_" + str(i))))
            self.tanh_beta_per_layer.append(float(args.get("tanh_beta_layer_" + str(i))))
        self.estimator = None
        self.random_state = random_state

    def _prefit(self, X, y):
        self.batch_size = int(self.batch_size)
        self.n_features = X.shape[1]
        self.input_shape = (self.batch_size, self.n_features)

        assert len(self.num_units_per_layer) == self.num_layers - 1,\
            "Number of created layers is different than actual layers"
        assert len(self.dropout_per_layer) == self.num_layers - 1,\
            "Number of created layers is different than actual layers"

        if len(y.shape) == 2 and y.shape[1] > 1:  # Multilabel
            self.m_ismultilabel = True
            self.num_output_units = y.shape[1]
        else:
            number_classes = len(np.unique(y.astype(int)))
            if number_classes == 2:  # Make it binary
                self.m_isbinary = True
                self.num_output_units = 1
                if len(y.shape) == 1:
                    y = y[:, np.newaxis]
            else:
                self.num_output_units = number_classes

        self.m_issparse = sp.issparse(X)

        return X, y

    def fit(self, X, y):

        Xf, yf = self._prefit(X, y)

        epoch = (self.number_updates * self.batch_size)//X.shape[0]
        number_epochs = min(max(2, epoch), 80)  # Capping of epochs

        from implementation import FeedForwardNet
        self.estimator = FeedForwardNet.FeedForwardNet(batch_size=self.batch_size,
                                                       input_shape=self.input_shape,
                                                       num_layers=self.num_layers,
                                                       num_units_per_layer=self.num_units_per_layer,
                                                       dropout_per_layer=self.dropout_per_layer,
                                                       activation_per_layer=self.activation_per_layer,
                                                       weight_init_per_layer=self.weight_init_layer,
                                                       std_per_layer=self.std_per_layer,
                                                       leakiness_per_layer=self.leakiness_per_layer,
                                                       tanh_alpha_per_layer=self.tanh_alpha_per_layer,
                                                       tanh_beta_per_layer=self.tanh_beta_per_layer,
                                                       num_output_units=self.num_output_units,
                                                       dropout_output=self.dropout_output,
                                                       learning_rate=self.learning_rate,
                                                       lr_policy=self.lr_policy,
                                                       lambda2=self.lambda2,
                                                       momentum=self.momentum,
                                                       beta1=self.beta1,
                                                       beta2=self.beta2,
                                                       rho=self.rho,
                                                       solver=self.solver,
                                                       num_epochs=number_epochs,
                                                       gamma=self.gamma,
                                                       power=self.power,
                                                       epoch_step=self.epoch_step,
                                                       is_sparse=self.m_issparse,
                                                       is_binary=self.m_isbinary,
                                                       is_multilabel=self.m_ismultilabel,
                                                       random_state=self.random_state)
        self.estimator.fit(Xf, yf)
        return self

    def predict(self, X):
        if self.estimator is None:
            raise NotImplementedError
        return self.estimator.predict(X, self.m_issparse)

    def predict_proba(self, X):
        if self.estimator is None:
            raise NotImplementedError()
        return self.estimator.predict_proba(X, self.m_issparse)

    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'feed_nn',
                'name': 'Feed Forward Neural Network',
                'handles_regression': False,
                'handles_classification': True,
                'handles_multiclass': True,
                'handles_multilabel': True,
                'is_deterministic': True,
                'input': (DENSE, SPARSE, UNSIGNED_DATA),
                'output': (PREDICTIONS,)}

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):

        max_num_layers = 7  # Maximum number of layers coded

        # Hacky way to condition layers params based on the number of layers
        # 'c'=1, 'd'=2, 'e'=3 ,'f'=4', g ='5', h='6' + output_layer
        layer_choices = [chr(i) for i in range(ord('c'), ord('b')+max_num_layers)]

        batch_size = UniformIntegerHyperparameter("batch_size",
                                                  32, 4096,
                                                  log=True,
                                                  default=32)

        number_updates = UniformIntegerHyperparameter("number_updates",
                                                      50, 3500,
                                                      log=True,
                                                      default=200)

        num_layers = CategoricalHyperparameter("num_layers",
                                               choices=layer_choices,
                                               default='c')

        lr = UniformFloatHyperparameter("learning_rate", 1e-6, 1.0,
                                        log=True,
                                        default=0.01)

        l2 = UniformFloatHyperparameter("lambda2", 1e-7, 1e-2,
                                        log=True,
                                        default=1e-4)

        dropout_output = UniformFloatHyperparameter("dropout_output",
                                                    0.0, 0.99,
                                                    default=0.5)

        # Define basic hyperparameters and define the config space
        # basic means that are independent from the number of layers

        cs = ConfigurationSpace()
        # cs.add_hyperparameter(number_epochs)
        cs.add_hyperparameter(number_updates)
        cs.add_hyperparameter(batch_size)
        cs.add_hyperparameter(num_layers)
        cs.add_hyperparameter(lr)
        cs.add_hyperparameter(l2)
        cs.add_hyperparameter(dropout_output)

        #  Define parameters with different child parameters and conditions
        solver_choices = ["adam", "adadelta", "adagrad",
                          "sgd", "momentum", "nesterov",
                          "smorm3s"]

        solver = CategoricalHyperparameter(name="solver",
                                           choices=solver_choices,
                                           default="smorm3s")

        beta1 = UniformFloatHyperparameter("beta1", 1e-4, 0.1,
                                           log=True,
                                           default=0.1)

        beta2 = UniformFloatHyperparameter("beta2", 1e-4, 0.1,
                                           log=True,
                                           default=0.01)

        rho = UniformFloatHyperparameter("rho", 0.05, 0.99,
                                         log=True,
                                         default=0.95)

        momentum = UniformFloatHyperparameter("momentum", 0.3, 0.999,
                                              default=0.9)

        # TODO: Add policy based on this sklearn sgd
        policy_choices = ['fixed', 'inv', 'exp', 'step']

        lr_policy = CategoricalHyperparameter(name="lr_policy",
                                              choices=policy_choices,
                                              default='fixed')

        gamma = UniformFloatHyperparameter(name="gamma",
                                           lower=1e-3, upper=1e-1,
                                           default=1e-2)

        power = UniformFloatHyperparameter("power",
                                           0.0, 1.0,
                                           default=0.5)

        epoch_step = UniformIntegerHyperparameter("epoch_step",
                                                  2, 20,
                                                  default=5)

        cs.add_hyperparameter(solver)
        cs.add_hyperparameter(beta1)
        cs.add_hyperparameter(beta2)
        cs.add_hyperparameter(momentum)
        cs.add_hyperparameter(rho)
        cs.add_hyperparameter(lr_policy)
        cs.add_hyperparameter(gamma)
        cs.add_hyperparameter(power)
        cs.add_hyperparameter(epoch_step)

        # Define parameters that are needed it for each layer
        output_activation_choices = ['softmax', 'sigmoid', 'softplus', 'tanh']

        activations_choices = ['sigmoid', 'tanh', 'scaledTanh', 'elu', 'relu', 'leaky', 'linear']

        # http://lasagne.readthedocs.io/en/latest/modules/nonlinearities.html
        # #lasagne.nonlinearities.ScaledTanH
        # For normalized inputs, tanh_alpha = 2./3. and tanh_beta = 1.7159,
        # according to http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf
        # TODO: Review the bounds on  tanh_alpha and tanh_beta

        weight_choices = ['constant', 'normal', 'uniform',
                          'glorot_normal', 'glorot_uniform',
                          'he_normal', 'he_uniform',
                          'ortogonal', 'sparse']

        # Iterate over parameters that are used in each layer
        for i in range(1, max_num_layers):
            layer_units = UniformIntegerHyperparameter("num_units_layer_" + str(i),
                                                       64, 4096,
                                                       log=True,
                                                       default=128)
            cs.add_hyperparameter(layer_units)
            layer_dropout = UniformFloatHyperparameter("dropout_layer_" + str(i),
                                                       0.0, 0.99,
                                                       default=0.5)
            cs.add_hyperparameter(layer_dropout)
            weight_initialization = CategoricalHyperparameter('weight_init_' + str(i),
                                                              choices=weight_choices,
                                                              default='he_normal')
            cs.add_hyperparameter(weight_initialization)
            layer_std = UniformFloatHyperparameter("std_layer_" + str(i),
                                                   1e-6, 0.1,
                                                   log=True,
                                                   default=0.005)
            cs.add_hyperparameter(layer_std)
            layer_activation = CategoricalHyperparameter("activation_layer_" + str(i),
                                                         choices=activations_choices,
                                                         default="relu")
            cs.add_hyperparameter(layer_activation)
            layer_leakiness = UniformFloatHyperparameter('leakiness_layer_' + str(i),
                                                         0.01, 0.99,
                                                         default=0.3)

            cs.add_hyperparameter(layer_leakiness)
            layer_tanh_alpha = UniformFloatHyperparameter('tanh_alpha_layer_' + str(i),
                                                          0.5, 1.0,
                                                          default=2./3.)
            cs.add_hyperparameter(layer_tanh_alpha)
            layer_tanh_beta = UniformFloatHyperparameter('tanh_beta_layer_' + str(i),
                                                         1.1, 3.0,
                                                         log=True,
                                                         default=1.7159)
            cs.add_hyperparameter(layer_tanh_beta)

        # TODO: Could be in a function in a new module
        for i in range(2, max_num_layers):
            # Condition layers parameter on layer choice
            layer_unit_param = cs.get_hyperparameter("num_units_layer_" + str(i))
            layer_cond = InCondition(child=layer_unit_param, parent=num_layers,
                                     values=[l for l in layer_choices[i-1:]])
            cs.add_condition(layer_cond)
            # Condition dropout parameter on layer choice
            layer_dropout_param = cs.get_hyperparameter("dropout_layer_" + str(i))
            layer_cond = InCondition(child=layer_dropout_param, parent=num_layers,
                                     values=[l for l in layer_choices[i-1:]])
            cs.add_condition(layer_cond)
            # Condition weight initialization on layer choice
            layer_weight_param = cs.get_hyperparameter("weight_init_" + str(i))
            layer_cond = InCondition(child=layer_weight_param, parent=num_layers,
                                     values=[l for l in layer_choices[i-1:]])
            cs.add_condition(layer_cond)
            # Condition std parameter on weight layer initialization choice
            layer_std_param = cs.get_hyperparameter("std_layer_" + str(i))
            weight_cond = EqualsCondition(child=layer_std_param,
                                          parent=layer_weight_param,
                                          value='normal')
            cs.add_condition(weight_cond)
            # Condition activation parameter on layer choice
            layer_activation_param = cs.get_hyperparameter("activation_layer_" + str(i))
            layer_cond = InCondition(child=layer_activation_param, parent=num_layers,
                                     values=[l for l in layer_choices[i-1:]])
            cs.add_condition(layer_cond)
            # Condition leakiness on activation choice
            layer_leakiness_param = cs.get_hyperparameter("leakiness_layer_" + str(i))
            activation_cond = EqualsCondition(child=layer_leakiness_param,
                                              parent=layer_activation_param,
                                              value='leaky')
            cs.add_condition(activation_cond)
            # Condition tanh on activation choice
            layer_tanh_alpha_param = cs.get_hyperparameter("tanh_alpha_layer_" + str(i))
            activation_cond = EqualsCondition(child=layer_tanh_alpha_param,
                                              parent=layer_activation_param,
                                              value='scaledTanh')
            cs.add_condition(activation_cond)
            layer_tanh_beta_param = cs.get_hyperparameter("tanh_beta_layer_" + str(i))
            activation_cond = EqualsCondition(child=layer_tanh_beta_param,
                                              parent=layer_activation_param,
                                              value='scaledTanh')
            cs.add_condition(activation_cond)

        # Conditioning on solver
        momentum_depends_on_solver = InCondition(momentum, solver,
                                                 values=["momentum", "nesterov"])
        beta1_depends_on_solver = EqualsCondition(beta1, solver, "adam")
        beta2_depends_on_solver = EqualsCondition(beta2, solver, "adam")
        rho_depends_on_solver = EqualsCondition(rho, solver, "adadelta")

        cs.add_condition(momentum_depends_on_solver)
        cs.add_condition(beta1_depends_on_solver)
        cs.add_condition(beta2_depends_on_solver)
        cs.add_condition(rho_depends_on_solver)

        # Conditioning on learning rate policy
        lr_policy_depends_on_solver = InCondition(lr_policy, solver,
                                                  ["adadelta", "adagrad", "sgd",
                                                   "momentum", "nesterov"])
        gamma_depends_on_policy = InCondition(child=gamma, parent=lr_policy,
                                              values=["inv", "exp", "step"])
        power_depends_on_policy = EqualsCondition(power, lr_policy, "inv")
        epoch_step_depends_on_policy = EqualsCondition(epoch_step, lr_policy, "step")

        cs.add_condition(lr_policy_depends_on_solver)
        cs.add_condition(gamma_depends_on_policy)
        cs.add_condition(power_depends_on_policy)
        cs.add_condition(epoch_step_depends_on_policy)

        return cs
