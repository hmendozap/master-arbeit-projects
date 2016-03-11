# -*- encoding: utf-8 -*-

from HPOlibConfigSpace.configuration_space import ConfigurationSpace
from HPOlibConfigSpace.hyperparameters import Constant, \
    UniformFloatHyperparameter

from component.DeepFeedNet import DeepFeedNet


class ConstrainedFeedNet(DeepFeedNet):

    # Weak subtyping
    def __init__(self, number_updates, batch_size, num_layers, num_units_layer_1,
                 num_units_layer_2, dropout_layer_1, dropout_layer_2, dropout_output,
                 std_layer_1, std_layer_2, learning_rate, solver, beta1=0.9, beta2=0.9,
                 lambda2=1e-4, rho=0.96, momentum=0.99, random_state=None):
        DeepFeedNet.__init__(self, number_updates, batch_size, num_layers, num_units_layer_1,
                             num_units_layer_2, num_units_layer_3=0, num_units_layer_4=0,
                             num_units_layer_5=0, num_units_layer_6=0,
                             dropout_layer_1=dropout_layer_1, dropout_layer_2=dropout_layer_2,
                             dropout_layer_3=0, dropout_layer_4=0,
                             dropout_layer_5=0, dropout_layer_6=0, dropout_output=dropout_output,
                             std_layer_1=std_layer_1, std_layer_2=std_layer_2, std_layer_3=0, std_layer_4=0,
                             std_layer_5=0, std_layer_6=0, learning_rate=learning_rate, solver='sgd', lambda2=lambda2,
                             momentum=momentum, beta1=0.9, beta2=0.9, rho=0.95, random_state=random_state)

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        # Constrain the search space of the DeepFeedNet

        # Fixed Architecture
        batch_size = Constant(name='batch_size', value=206)
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

        # To Optimize for all cases
        l2 = UniformFloatHyperparameter("lambda2", 1e-6, 1e-2,
                                        log=True,
                                        default=1e-3)
        lr = UniformFloatHyperparameter("learning_rate", 1e-6, 2e-1,
                                        log=True,
                                        default=0.01)
        momentum = UniformFloatHyperparameter("momentum", 0.3, 0.999,
                                              default=0.9)

        cs = ConfigurationSpace()
        cs.add_hyperparameter(batch_size)
        cs.add_hyperparameter(number_updates)
        cs.add_hyperparameter(num_layers)
        cs.add_hyperparameter(num_units_layer_1)
        cs.add_hyperparameter(num_units_layer_2)
        cs.add_hyperparameter(dropout_layer_1)
        cs.add_hyperparameter(dropout_layer_2)
        cs.add_hyperparameter(dropout_output)
        cs.add_hyperparameter(std_layer_1)
        cs.add_hyperparameter(std_layer_2)
        cs.add_hyperparameter(lr)
        cs.add_hyperparameter(l2)
        cs.add_hyperparameter(momentum)

        return cs


class AdamConstFeedNet(ConstrainedFeedNet):

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        cs = ConstrainedFeedNet.get_hyperparameter_search_space()
        solver = Constant(name='solver', value='adam')
        cs.add_hyperparameter(solver)
        beta1 = UniformFloatHyperparameter("beta1", 1e-4, 0.1,
                                           log=True,
                                           default=0.1)
        beta2 = UniformFloatHyperparameter("beta2", 1e-4, 0.1,
                                           log=True,
                                           default=0.1)
        cs.add_hyperparameter(beta1)
        cs.add_hyperparameter(beta2)
        return cs


class SGDConstFeedNet(ConstrainedFeedNet):

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        cs = ConstrainedFeedNet.get_hyperparameter_search_space()
        solver = Constant(name='solver', value='sgd')
        cs.add_hyperparameter(solver)

        return cs


class AdadeltaConstFeedNet(ConstrainedFeedNet):

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        cs = ConstrainedFeedNet.get_hyperparameter_search_space()
        solver = Constant(name='solver', value='adadelta')
        cs.add_hyperparameter(solver)

        rho = UniformFloatHyperparameter('rho', 0.0, 1.0, default=0.95)
        cs.add_hyperparameter(rho)
        return cs


class AdagradConstFeedNet(ConstrainedFeedNet):

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        cs = ConstrainedFeedNet.get_hyperparameter_search_space()
        solver = Constant(name='solver', value='adagrad')
        cs.add_hyperparameter(solver)

        return cs


class NesterovConstFeedNet(ConstrainedFeedNet):

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        cs = ConstrainedFeedNet.get_hyperparameter_search_space()
        solver = Constant(name='solver', value='nesterov')
        cs.add_hyperparameter(solver)

        return cs


class MomentumConstFeedNet(ConstrainedFeedNet):

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        cs = ConstrainedFeedNet.get_hyperparameter_search_space()
        solver = Constant(name='solver', value='momentum')
        cs.add_hyperparameter(solver)

        return cs