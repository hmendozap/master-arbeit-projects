
import numpy as np
import scipy.sparse as sp

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.conditions import EqualsCondition, InCondition
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, \
    UniformIntegerHyperparameter, CategoricalHyperparameter, \
    UnParametrizedHyperparameter

from autosklearn.pipeline.components.base import AutoSklearnRegressionAlgorithm
from autosklearn.pipeline.constants import *
from implementation import FeedForwardNet


class RegDeepNet(AutoSklearnRegressionAlgorithm):

    def __init__(self, number_updates, batch_size, num_layers, num_units_layer_1,
                 dropout_layer_1, dropout_output, std_layer_1,
                 learning_rate, solver, lambda2, activation,
                 num_units_layer_2=10, num_units_layer_3=10, num_units_layer_4=10,
                 num_units_layer_5=10, num_units_layer_6=10,
                 dropout_layer_2=0.5, dropout_layer_3=0.5, dropout_layer_4=0.5,
                 dropout_layer_5=0.5, dropout_layer_6=0.5,
                 std_layer_2=0.005, std_layer_3=0.005, std_layer_4=0.005,
                 std_layer_5=0.005, std_layer_6=0.005,
                 momentum=0.99, beta1=0.9, beta2=0.9, rho=0.95,
                 lr_policy='fixed', gamma=0.01, power=1.0, epoch_step=2,
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
        self.activation = activation
        self.gamma = gamma
        self.power = power
        self.epoch_step = epoch_step

        # Empty features and shape
        self.n_features = None
        self.input_shape = None
        self.m_issparse = False
        self.m_isregression = True

        # To avoid eval call. Could be done with **karws
        args = locals()

        self.num_units_per_layer = []
        self.dropout_per_layer = []
        self.std_per_layer = []
        for i in range(1, self.num_layers):
            self.num_units_per_layer.append(int(args.get("num_units_layer_" + str(i))))
            self.dropout_per_layer.append(float(args.get("dropout_layer_" + str(i))))
            self.std_per_layer.append(float(args.get("std_layer_" + str(i))))
        self.estimator = None

    def _prefit(self, X, y):
        self.batch_size = int(self.batch_size)
        self.n_features = X.shape[1]
        self.input_shape = (self.batch_size, self.n_features)
        number_classes = len(np.unique(y.astype(int)))

        assert len(self.num_units_per_layer) == self.num_layers - 1,\
            "Number of created layers is different than actual layers"
        assert len(self.dropout_per_layer) == self.num_layers - 1,\
            "Number of created layers is different than actual layers"

        self.num_output_units = 1  # Regression, actually no true
        self.m_issparse = sp.issparse(X)

        return X, y

    def fit(self, X, y):

        Xf, yf = self._prefit(X, y)

        epoch = (self.number_updates * self.batch_size)//X.shape[0]
        # number_epochs = max(2, epoch)
        number_epochs = min(max(2, epoch), 50)  # Cap the max number of possible epochs

        self.estimator = FeedForwardNet.FeedForwardNet(batch_size=self.batch_size,
                                                       input_shape=self.input_shape,
                                                       num_layers=self.num_layers,
                                                       num_units_per_layer=self.num_units_per_layer,
                                                       dropout_per_layer=self.dropout_per_layer,
                                                       std_per_layer=self.std_per_layer,
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
                                                       activation=self.activation,
                                                       num_epochs=number_epochs,
                                                       gamma=self.gamma,
                                                       power=self.power,
                                                       epoch_step=self.epoch_step,
                                                       is_sparse=self.m_issparse,
                                                       is_binary=False,
                                                       is_regression=self.m_isregression)
        self.estimator.fit(Xf, yf)
        return self

    def predict(self, X):
        if self.estimator is None:
            raise NotImplementedError
        if sp.issparse(X):
            is_sparse = True
        else:
            is_sparse = False
        return self.estimator.predict(X, is_sparse)

    def predict_proba(self, X):
        if self.estimator is None:
            raise NotImplementedError()
        if sp.issparse(X):
            is_sparse = True
        else:
            is_sparse = False
        return self.estimator.predict_proba(X, is_sparse)

    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'feed_nn',
                'name': 'Feed Forward Neural Network',
                'handles_regression': False,
                'handles_classification': True,
                'handles_multiclass': True,
                'handles_multilabel': False,
                'is_deterministic': True,
                'input': (DENSE, SPARSE, UNSIGNED_DATA),
                'output': (PREDICTIONS,)}

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):

        solver_choices = ["adam", "adadelta", "adagrad", "sgd", "momentum", "nesterov"]

        policy_choices = ['fixed', 'inv', 'exp', 'step']

        # TODO: Add ScaledTanh hyperparamteres and Leakyrectify
        binary_activations = ['sigmoid', 'tanh', 'scaledTanh', 'softplus',
                              'elu', 'relu']

        multiclass_activations = ['relu', 'leaky', 'very_leaky', 'elu',
                                  'softplus', 'softmax', 'linear', 'scaledTanh']

        # Hacky way to condition layers params based on the number of layers
        # 'c'=2, 'd'=3, 'e'=4 ,'f'=5, 'g'=6, 'h'=7
        layer_choices = [chr(i) for i in xrange(ord('c'), ord('i'))]

        batch_size = UniformIntegerHyperparameter("batch_size", 100, 1000,
                                                  log=True,
                                                  default=100)

        number_updates = UniformIntegerHyperparameter("number_updates",
                                                      50, 2500,
                                                      log=True,
                                                      default=150)

        # number_epochs = UniformIntegerHyperparameter("number_epochs", 2, 20,
        #                                             default=3)

        num_layers = CategoricalHyperparameter("num_layers",
                                               choices=layer_choices,
                                               default='e')

        # <editor-fold desc="Number of units in layers 1-6">
        num_units_layer_1 = UniformIntegerHyperparameter("num_units_layer_1",
                                                         10, 6144,
                                                         log=True,
                                                         default=10)

        num_units_layer_2 = UniformIntegerHyperparameter("num_units_layer_2",
                                                         10, 6144,
                                                         log=True,
                                                         default=10)

        num_units_layer_3 = UniformIntegerHyperparameter("num_units_layer_3",
                                                         10, 6144,
                                                         log=True,
                                                         default=10)

        num_units_layer_4 = UniformIntegerHyperparameter("num_units_layer_4",
                                                         10, 6144,
                                                         log=True,
                                                         default=10)

        num_units_layer_5 = UniformIntegerHyperparameter("num_units_layer_5",
                                                         10, 6144,
                                                         log=True,
                                                         default=10)

        num_units_layer_6 = UniformIntegerHyperparameter("num_units_layer_6",
                                                         10, 6144,
                                                         log=True,
                                                         default=10)
        # </editor-fold>

        # <editor-fold desc="Dropout in layers 1-6">
        dropout_layer_1 = UniformFloatHyperparameter("dropout_layer_1",
                                                     0.0, 0.99,
                                                     default=0.5)

        dropout_layer_2 = UniformFloatHyperparameter("dropout_layer_2",
                                                     0.0, 0.99,
                                                     default=0.5)

        dropout_layer_3 = UniformFloatHyperparameter("dropout_layer_3",
                                                     0.0, 0.99,
                                                     default=0.5)

        dropout_layer_4 = UniformFloatHyperparameter("dropout_layer_4",
                                                     0.0, 0.99,
                                                     default=0.5)

        dropout_layer_5 = UniformFloatHyperparameter("dropout_layer_5",
                                                     0.0, 0.99,
                                                     default=0.5)

        dropout_layer_6 = UniformFloatHyperparameter("dropout_layer_6",
                                                     0.0, 0.99,
                                                     default=0.5)
        # </editor-fold>

        dropout_output = UniformFloatHyperparameter("dropout_output", 0.0, 0.99,
                                                    default=0.5)

        lr = UniformFloatHyperparameter("learning_rate", 1e-6, 1, log=True,
                                        default=0.01)

        # Todo: Check lambda2 parameter bounds
        l2 = UniformFloatHyperparameter("lambda2", 1e-6, 1e-2, log=True,
                                        default=1e-3)

        momentum = UniformFloatHyperparameter("momentum", 0.3, 0.999,
                                              default=0.9)

        # <editor-fold desc="Std for layers 1-6">
        std_layer_1 = UniformFloatHyperparameter("std_layer_1", 1e-6, 0.1,
                                                 log=True,
                                                 default=0.005)

        std_layer_2 = UniformFloatHyperparameter("std_layer_2", 1e-6, 0.1,
                                                 log=True,
                                                 default=0.005)

        std_layer_3 = UniformFloatHyperparameter("std_layer_3", 1e-6, 0.1,
                                                 log=True,
                                                 default=0.005)

        std_layer_4 = UniformFloatHyperparameter("std_layer_4", 1e-6, 0.1,
                                                 log=True,
                                                 default=0.005)

        std_layer_5 = UniformFloatHyperparameter("std_layer_5", 1e-6, 0.1,
                                                 log=True,
                                                 default=0.005)

        std_layer_6 = UniformFloatHyperparameter("std_layer_6", 1e-6, 0.1,
                                                 log=True,
                                                 default=0.005)
        # </editor-fold>

        solver = CategoricalHyperparameter(name="solver",
                                           choices=solver_choices,
                                           default="sgd")

        beta1 = UniformFloatHyperparameter("beta1", 1e-4, 0.1,
                                           log=True,
                                           default=0.1)
        beta2 = UniformFloatHyperparameter("beta2", 1e-4, 0.1,
                                           log=True,
                                           default=0.1)
        rho = UniformFloatHyperparameter("rho", 0.0, 1.0, default=0.95)

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
                                                  2, 10,
                                                  default=2)

        if (dataset_properties is not None and
                dataset_properties.get('multiclass') is False):

            non_linearities = CategoricalHyperparameter(name='activation',
                                                        choices=binary_activations,
                                                        default='sigmoid')
        else:
            non_linearities = CategoricalHyperparameter(name='activation',
                                                        choices=multiclass_activations,
                                                        default='relu')

        cs = ConfigurationSpace()
        # cs.add_hyperparameter(number_epochs)
        cs.add_hyperparameter(number_updates)
        cs.add_hyperparameter(batch_size)
        cs.add_hyperparameter(num_layers)
        cs.add_hyperparameter(num_units_layer_1)
        cs.add_hyperparameter(num_units_layer_2)
        cs.add_hyperparameter(num_units_layer_3)
        cs.add_hyperparameter(num_units_layer_4)
        cs.add_hyperparameter(num_units_layer_5)
        cs.add_hyperparameter(num_units_layer_6)
        cs.add_hyperparameter(dropout_layer_1)
        cs.add_hyperparameter(dropout_layer_2)
        cs.add_hyperparameter(dropout_layer_3)
        cs.add_hyperparameter(dropout_layer_4)
        cs.add_hyperparameter(dropout_layer_5)
        cs.add_hyperparameter(dropout_layer_6)
        cs.add_hyperparameter(dropout_output)
        cs.add_hyperparameter(std_layer_1)
        cs.add_hyperparameter(std_layer_2)
        cs.add_hyperparameter(std_layer_3)
        cs.add_hyperparameter(std_layer_4)
        cs.add_hyperparameter(std_layer_5)
        cs.add_hyperparameter(std_layer_6)
        cs.add_hyperparameter(lr)
        cs.add_hyperparameter(l2)
        cs.add_hyperparameter(solver)
        cs.add_hyperparameter(momentum)
        cs.add_hyperparameter(beta1)
        cs.add_hyperparameter(beta2)
        cs.add_hyperparameter(rho)
        cs.add_hyperparameter(lr_policy)
        cs.add_hyperparameter(gamma)
        cs.add_hyperparameter(power)
        cs.add_hyperparameter(epoch_step)
        cs.add_hyperparameter(non_linearities)

        # TODO: Put this conditioning in a for-loop
        # Condition layers parameter on layer choice
        layer_2_condition = InCondition(num_units_layer_2, num_layers, ['d', 'e', 'f', 'g', 'h'])
        layer_3_condition = InCondition(num_units_layer_3, num_layers, ['e', 'f', 'g', 'h'])
        layer_4_condition = InCondition(num_units_layer_4, num_layers, ['f', 'g', 'h'])
        layer_5_condition = InCondition(num_units_layer_5, num_layers, ['g', 'h'])
        layer_6_condition = InCondition(num_units_layer_6, num_layers, ['h'])
        cs.add_condition(layer_2_condition)
        cs.add_condition(layer_3_condition)
        cs.add_condition(layer_4_condition)
        cs.add_condition(layer_5_condition)
        cs.add_condition(layer_6_condition)

        # Condition dropout parameter on layer choice
        dropout_2_condition = InCondition(dropout_layer_2, num_layers, ['d', 'e', 'f', 'g', 'h'])
        dropout_3_condition = InCondition(dropout_layer_3, num_layers, ['e', 'f', 'g', 'h'])
        dropout_4_condition = InCondition(dropout_layer_4, num_layers, ['f', 'g', 'h'])
        dropout_5_condition = InCondition(dropout_layer_5, num_layers, ['g', 'h'])
        dropout_6_condition = InCondition(dropout_layer_6, num_layers, ['h'])
        cs.add_condition(dropout_2_condition)
        cs.add_condition(dropout_3_condition)
        cs.add_condition(dropout_4_condition)
        cs.add_condition(dropout_5_condition)
        cs.add_condition(dropout_6_condition)

        # Condition std parameter on layer choice
        std_2_condition = InCondition(std_layer_2, num_layers, ['d', 'e', 'f', 'g', 'h'])
        std_3_condition = InCondition(std_layer_3, num_layers, ['e', 'f', 'g', 'h'])
        std_4_condition = InCondition(std_layer_4, num_layers, ['f', 'g', 'h'])
        std_5_condition = InCondition(std_layer_5, num_layers, ['g', 'h'])
        std_6_condition = InCondition(std_layer_6, num_layers, ['h'])
        cs.add_condition(std_2_condition)
        cs.add_condition(std_3_condition)
        cs.add_condition(std_4_condition)
        cs.add_condition(std_5_condition)
        cs.add_condition(std_6_condition)

        momentum_depends_on_solver = InCondition(momentum, solver,
                                                 values=["sgd", "momentum", "nesterov"])
        beta1_depends_on_solver = EqualsCondition(beta1, solver, "adam")
        beta2_depends_on_solver = EqualsCondition(beta2, solver, "adam")
        rho_depends_on_solver = EqualsCondition(rho, solver, "adadelta")
        lr_policy_depends_on_solver = InCondition(lr_policy, solver,
                                                  ["adadelta", "adagrad", "sgd",
                                                   "momentum", "nesterov"])
        gamma_depends_on_policy = InCondition(child=gamma, parent=lr_policy,
                                              values=['inv', 'exp', 'step'])
        power_depends_on_policy = EqualsCondition(power, lr_policy, 'inv')
        epoch_step_depends_on_policy = EqualsCondition(epoch_step, lr_policy, 'step')

        cs.add_condition(momentum_depends_on_solver)
        cs.add_condition(beta1_depends_on_solver)
        cs.add_condition(beta2_depends_on_solver)
        cs.add_condition(rho_depends_on_solver)
        cs.add_condition(lr_policy_depends_on_solver)
        cs.add_condition(gamma_depends_on_policy)
        cs.add_condition(power_depends_on_policy)
        cs.add_condition(epoch_step_depends_on_policy)

        return cs
