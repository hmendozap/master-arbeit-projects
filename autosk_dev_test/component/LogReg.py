import numpy as np
import scipy.sparse as sp

from HPOlibConfigSpace.configuration_space import ConfigurationSpace
from HPOlibConfigSpace.conditions import EqualsCondition, InCondition
from HPOlibConfigSpace.hyperparameters import UniformFloatHyperparameter, \
    UniformIntegerHyperparameter, CategoricalHyperparameter, Constant

from autosklearn.pipeline.components.base import AutoSklearnClassificationAlgorithm
from autosklearn.pipeline.constants import *


class LogReg(AutoSklearnClassificationAlgorithm):

    def __init__(self, number_updates, batch_size, dropout_output,
                 learning_rate, solver, lambda2, activation,
                 momentum=0.99, beta1=0.9, beta2=0.9, rho=0.95,
                 lr_policy='fixed', gamma=0.01, power=1.0, epoch_step=2,
                 random_state=None):
        self.number_updates = number_updates
        self.batch_size = batch_size
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
        self.m_isregression = False
        self.m_isbinary = False
        self.m_ismultilabel = False

        self.estimator = None

    def _prefit(self, X, y):
        self.batch_size = int(self.batch_size)
        self.n_features = X.shape[1]
        self.input_shape = (self.batch_size, self.n_features)

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
        number_epochs = min(max(2, epoch), 210)  # Cap the max number of possible epochs

        from ...implementations import LogisticRegression
        self.estimator = LogisticRegression.LogisticRegression(batch_size=self.batch_size,
                                                               input_shape=self.input_shape,
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
                                                               is_binary=self.m_isbinary,
                                                               is_multilabel=self.m_ismultilabel)
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
        return {'shortname': 'log_reg',
                'name': 'Logistic Regression',
                'handles_regression': False,
                'handles_classification': True,
                'handles_multiclass': True,
                'handles_multilabel': True,
                'is_deterministic': True,
                'input': (DENSE, SPARSE, UNSIGNED_DATA),
                'output': (PREDICTIONS,)}

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):

        policy_choices = ['fixed', 'inv', 'exp', 'step']

        batch_size = UniformIntegerHyperparameter("batch_size",
                                                  32, 2048,
                                                  log=True,
                                                  default=100)

        number_updates = UniformIntegerHyperparameter("number_updates",
                                                      500, 10500,
                                                      log=True,
                                                      default=1050)

        dropout_output = UniformFloatHyperparameter("dropout_output", 0.0, 0.99,
                                                    default=0.5)

        lr = UniformFloatHyperparameter("learning_rate", 1e-6, 0.1, log=True,
                                        default=1e-6)

        l2 = UniformFloatHyperparameter("lambda2", 1e-6, 1e-2, log=True,
                                        default=1e-6)

        solver = CategoricalHyperparameter(name="solver", choices=["sgd", "adam"],
                                           default="adam")

        beta1 = UniformFloatHyperparameter("beta1", 1e-4, 0.1,
                                           log=True,
                                           default=0.1)

        beta2 = UniformFloatHyperparameter("beta2", 1e-4, 0.1,
                                           log=True,
                                           default=0.01)

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

        if (dataset_properties is not None and
                dataset_properties.get('multiclass') is False):
            non_linearities = Constant(name='activation', value='sigmoid')
        else:
            non_linearities = Constant(name='activation', value='softmax')

        cs = ConfigurationSpace()
        cs.add_hyperparameter(number_updates)
        cs.add_hyperparameter(batch_size)
        cs.add_hyperparameter(dropout_output)
        cs.add_hyperparameter(lr)
        cs.add_hyperparameter(l2)
        cs.add_hyperparameter(solver)
        cs.add_hyperparameter(beta1)
        cs.add_hyperparameter(beta2)
        cs.add_hyperparameter(lr_policy)
        cs.add_hyperparameter(gamma)
        cs.add_hyperparameter(power)
        cs.add_hyperparameter(epoch_step)
        cs.add_hyperparameter(non_linearities)

        gamma_depends_on_policy = InCondition(child=gamma, parent=lr_policy,
                                              values=['inv', 'exp', 'step'])
        power_depends_on_policy = EqualsCondition(power, lr_policy, 'inv')
        epoch_step_depends_on_policy = EqualsCondition(epoch_step,
                                                       lr_policy, 'step')

        cs.add_condition(gamma_depends_on_policy)
        cs.add_condition(power_depends_on_policy)
        cs.add_condition(epoch_step_depends_on_policy)

        return cs
