# -*- encoding: utf-8 -*-

from HPOlibConfigSpace.hyperparameters import UniformIntegerHyperparameter, Constant, \
                                              CategoricalHyperparameter,\
                                              UniformFloatHyperparameter
from HPOlibConfigSpace.configuration_space import ConfigurationSpace

from autosklearn.pipeline.components.base import AutoSklearnClassificationAlgorithm
from component.DeepFeedNet import DeepFeedNet


class ConstrainedFeedNet(DeepFeedNet, AutoSklearnClassificationAlgorithm):

    # Weak subtyping
    def __init__(self, number_updates, batch_size, num_layers, num_units_layer_1,
                 num_units_layer_2, dropout_layer_1, dropout_layer_2, dropout_output,
                 std_layer_1, std_layer_2, learning_rate, solver, beta1, beta2,
                 momentum=0.99, random_state=None):
        DeepFeedNet.__init__(self, number_updates, batch_size, num_layers, num_units_layer_1,
                             num_units_layer_2, num_units_layer_3=0, num_units_layer_4=0,
                             num_units_layer_5=0, num_units_layer_6=0,
                             dropout_layer_1=dropout_layer_1, dropout_layer_2=dropout_layer_2,
                             dropout_layer_3=0, dropout_layer_4=0,
                             dropout_layer_5=0, dropout_layer_6=0, dropout_output=dropout_output,
                             std_layer_1=std_layer_1, std_layer_2=std_layer_2, std_layer_3=0, std_layer_4=0,
                             std_layer_5=0, std_layer_6=0, learning_rate=learning_rate, solver=solver,
                             momentum=momentum, beta1=beta1, beta2=beta2, rho=0.95, random_state=random_state)

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        # Constrain the search space of the DeepFeedNet

        # Fixed Architecture
        batch_size = Constant(name='batch_size', value=206)
        solver = Constant(name='solver', value='adam')
        # TODO: Decimal library to work around floating point issue
        dropout_layer_1 = Constant(name='dropout_layer_1', value=0.02540390796)
        dropout_layer_2 = Constant(name='dropout_layer_2', value=0.07463115112)
        dropout_output = Constant(name='dropout_output', value=0.911359961918)
        num_layers = Constant(name='num_layers', value=2)
        num_units_layer_1 = Constant(name='num_units_layer_1', value=34)
        num_units_layer_2 = Constant(name='num_units_layer_2', value=71)
        number_updates = Constant(name='number_updates', value=188)
        std_layer_1 = Constant(name='std_layer_1', value=6.56573567729E-4)
        std_layer_2 = Constant(name='std_layer_2', value=1.12307705271E-6)

        # To Optimize
        lr = UniformFloatHyperparameter("learning_rate", 1e-6, 2e-1,
                                        log=True,
                                        default=0.01)
        momentum = UniformFloatHyperparameter("momentum", 0.3, 0.999,
                                              default=0.9)
        beta1 = UniformFloatHyperparameter("beta1", 1e-4, 0.1,
                                           log=True,
                                           default=0.1)
        beta2 = UniformFloatHyperparameter("beta2", 1e-4, 0.1,
                                           log=True,
                                           default=0.1)

        cs = ConfigurationSpace()
        cs.add_hyperparameter(batch_size)
        cs.add_hyperparameter(number_updates)
        cs.add_hyperparameter(solver)
        cs.add_hyperparameter(num_layers)
        cs.add_hyperparameter(num_units_layer_1)
        cs.add_hyperparameter(num_units_layer_2)
        cs.add_hyperparameter(dropout_layer_1)
        cs.add_hyperparameter(dropout_layer_2)
        cs.add_hyperparameter(dropout_output)
        cs.add_hyperparameter(std_layer_1)
        cs.add_hyperparameter(std_layer_2)
        cs.add_hyperparameter(lr)
        cs.add_hyperparameter(momentum)
        cs.add_hyperparameter(beta1)
        cs.add_hyperparameter(beta2)

        return cs
