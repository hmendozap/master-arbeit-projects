# -*- encoding: utf-8 -*-

from HPOlibConfigSpace.configuration_space import ConfigurationSpace
from HPOlibConfigSpace.hyperparameters import Constant, CategoricalHyperparameter, \
    UniformFloatHyperparameter, UniformIntegerHyperparameter

from .DeepFeedNet import DeepFeedNet


def _lr_policy_configuration_space(cs, policy=None):
    if policy == 'inv':
        lr_policy = Constant(name='lr_policy', value='inv')
        gamma = UniformFloatHyperparameter(name="gamma",
                                           lower=1e-2, upper=1.0,
                                           log=True,
                                           default=1e-2)
        power = UniformFloatHyperparameter("power",
                                           0.0, 1.0,
                                           default=0.5)

        cs.add_hyperparameter(lr_policy)
        cs.add_hyperparameter(gamma)
        cs.add_hyperparameter(power)

    elif policy == 'exp':
        lr_policy = Constant(name='lr_policy', value='exp')
        gamma = UniformFloatHyperparameter(name="gamma",
                                           lower=0.7, upper=1.0,
                                           default=0.79)
        cs.add_hyperparameter(lr_policy)
        cs.add_hyperparameter(gamma)

    elif policy == 'step':
        lr_policy = Constant(name='lr_policy', value='step')
        gamma = UniformFloatHyperparameter(name="gamma",
                                           lower=1e-2, upper=1.0,
                                           log=True,
                                           default=1e-2)
        epoch_step = CategoricalHyperparameter("epoch_step",
                                               [6, 8, 12],
                                               default=8)
        cs.add_hyperparameter(lr_policy)
        cs.add_hyperparameter(gamma)
        cs.add_hyperparameter(epoch_step)

    else:
        lr_policy = Constant(name='lr_policy', value='fixed')
        cs.add_hyperparameter(lr_policy)

    return cs


class ConstrainedFeedNet(DeepFeedNet):

    # Weak subtyping
    def __init__(self, number_updates, batch_size, num_layers, num_units_layer_1,
                 dropout_layer_1, dropout_output,
                 std_layer_1, learning_rate, lambda2,
                 solver, momentum=0.99,
                 beta1=0.9, beta2=0.9, rho=0.96,
                 lr_policy='fixed', gamma=0.01, power=1.0, epoch_step=2,
                 random_state=None):
        DeepFeedNet.__init__(self, number_updates, batch_size, num_layers, num_units_layer_1,
                             num_units_layer_2=0, num_units_layer_3=0, num_units_layer_4=0,
                             num_units_layer_5=0, num_units_layer_6=0,
                             dropout_layer_1=dropout_layer_1, dropout_layer_2=0,
                             dropout_layer_3=0, dropout_layer_4=0,
                             dropout_layer_5=0, dropout_layer_6=0, dropout_output=dropout_output,
                             std_layer_1=std_layer_1, std_layer_2=0, std_layer_3=0, std_layer_4=0,
                             std_layer_5=0, std_layer_6=0, learning_rate=learning_rate,
                             solver=solver, lambda2=lambda2,
                             momentum=momentum, beta1=beta1, beta2=beta2, rho=rho,
                             lr_policy=lr_policy, gamma=gamma, power=power, epoch_step=epoch_step,
                             random_state=random_state)

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        # Constrain the search space of the DeepFeedNet

        # Fixed Architecture for MNIST
        batch_size = Constant(name='batch_size', value=963)
        # TODO: Decimal library to work around floating point issue
        dropout_layer_1 = Constant(name='dropout_layer_1', value=0.39426633933)
        dropout_output = Constant(name='dropout_output', value=0.085813712701)
        num_layers = Constant(name='num_layers', value='c')
        num_units_layer_1 = Constant(name='num_units_layer_1', value=1861)
        number_updates = Constant(name='number_updates', value=1105)
        std_layer_1 = Constant(name='std_layer_1', value=0.00351015701)

        # To Optimize for all cases
        l2 = UniformFloatHyperparameter("lambda2", 1e-6, 1e-2,
                                        log=True,
                                        default=1e-3)
        lr = UniformFloatHyperparameter("learning_rate", 1e-4, 1e-1,
                                        log=True,
                                        default=1e-2)

        cs = ConfigurationSpace()
        cs.add_hyperparameter(batch_size)
        cs.add_hyperparameter(number_updates)
        cs.add_hyperparameter(num_layers)
        cs.add_hyperparameter(num_units_layer_1)
        cs.add_hyperparameter(dropout_layer_1)
        cs.add_hyperparameter(dropout_output)
        cs.add_hyperparameter(std_layer_1)
        cs.add_hyperparameter(lr)
        cs.add_hyperparameter(l2)

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


class AdamInvConstFeedNet(AdamConstFeedNet):

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        cs = AdamConstFeedNet.get_hyperparameter_search_space()
        cs = _lr_policy_configuration_space(cs=cs, policy='inv')
        return cs


class AdamExpConstFeedNet(AdamConstFeedNet):

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        cs = AdamConstFeedNet.get_hyperparameter_search_space()
        cs = _lr_policy_configuration_space(cs=cs, policy='exp')
        return cs


class AdamStepConstFeedNet(AdamConstFeedNet):

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        cs = AdamConstFeedNet.get_hyperparameter_search_space()
        cs = _lr_policy_configuration_space(cs=cs, policy='step')
        return cs


class SGDConstFeedNet(ConstrainedFeedNet):

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        cs = ConstrainedFeedNet.get_hyperparameter_search_space()
        solver = Constant(name='solver', value='sgd')
        cs.add_hyperparameter(solver)
        # lr policy is fixed by default

        return cs


class SGDInvConstFeedNet(SGDConstFeedNet):

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        cs = SGDConstFeedNet.get_hyperparameter_search_space()
        cs = _lr_policy_configuration_space(cs=cs, policy='inv')
        return cs


class SGDExpConstFeedNet(SGDConstFeedNet):

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        cs = SGDConstFeedNet.get_hyperparameter_search_space()
        cs = _lr_policy_configuration_space(cs=cs, policy='exp')
        return cs


class SGDStepConstFeedNet(SGDConstFeedNet):

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        cs = SGDConstFeedNet.get_hyperparameter_search_space()
        cs = _lr_policy_configuration_space(cs=cs, policy='step')
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


class AdadeltaInvConstFeedNet(AdadeltaConstFeedNet):

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        cs = AdadeltaConstFeedNet.get_hyperparameter_search_space()
        cs = _lr_policy_configuration_space(cs=cs, policy='inv')
        return cs


class AdadeltaExpConstFeedNet(AdadeltaConstFeedNet):

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        cs = AdadeltaConstFeedNet.get_hyperparameter_search_space()
        cs = _lr_policy_configuration_space(cs=cs, policy='exp')

        return cs


class AdadeltaStepConstFeedNet(AdadeltaConstFeedNet):

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        cs = AdadeltaConstFeedNet.get_hyperparameter_search_space()
        cs = _lr_policy_configuration_space(cs=cs, policy='step')
        return cs


class AdagradConstFeedNet(ConstrainedFeedNet):

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        cs = ConstrainedFeedNet.get_hyperparameter_search_space()
        solver = Constant(name='solver', value='adagrad')
        cs.add_hyperparameter(solver)

        return cs


class AdagradInvConstFeedNet(AdagradConstFeedNet):

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        cs = AdagradConstFeedNet.get_hyperparameter_search_space()
        cs = _lr_policy_configuration_space(cs=cs, policy='inv')
        return cs


class AdagradExpConstFeedNet(AdagradConstFeedNet):

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        cs = AdagradConstFeedNet.get_hyperparameter_search_space()
        cs = _lr_policy_configuration_space(cs=cs, policy='exp')
        return cs


class AdagradStepConstFeedNet(AdagradConstFeedNet):

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        cs = AdagradConstFeedNet.get_hyperparameter_search_space()
        cs = _lr_policy_configuration_space(cs=cs, policy='step')
        return cs


class NesterovConstFeedNet(ConstrainedFeedNet):

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        cs = ConstrainedFeedNet.get_hyperparameter_search_space()
        solver = Constant(name='solver', value='nesterov')
        cs.add_hyperparameter(solver)

        momentum = UniformFloatHyperparameter("momentum", 0.3, 0.999,
                                              default=0.9)
        cs.add_hyperparameter(momentum)

        return cs


class NesterovInvConstFeedNet(NesterovConstFeedNet):

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        cs = NesterovConstFeedNet.get_hyperparameter_search_space()
        cs = _lr_policy_configuration_space(cs=cs, policy='inv')
        return cs


class NesterovExpConstFeedNet(NesterovConstFeedNet):

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        cs = NesterovConstFeedNet.get_hyperparameter_search_space()
        cs = _lr_policy_configuration_space(cs=cs, policy='exp')
        return cs


class NesterovStepConstFeedNet(NesterovConstFeedNet):

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        cs = NesterovConstFeedNet.get_hyperparameter_search_space()
        cs = _lr_policy_configuration_space(cs=cs, policy='step')
        return cs


class MomentumConstFeedNet(ConstrainedFeedNet):

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        cs = ConstrainedFeedNet.get_hyperparameter_search_space()
        solver = Constant(name='solver', value='momentum')
        cs.add_hyperparameter(solver)

        momentum = UniformFloatHyperparameter("momentum", 0.3, 0.999,
                                              default=0.9)
        cs.add_hyperparameter(momentum)

        return cs


class MomentumInvConstFeedNet(MomentumConstFeedNet):

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        cs = MomentumConstFeedNet.get_hyperparameter_search_space()
        cs = _lr_policy_configuration_space(cs=cs, policy='inv')
        return cs


class MomentumExpConstFeedNet(MomentumConstFeedNet):

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        cs = MomentumConstFeedNet.get_hyperparameter_search_space()
        cs = _lr_policy_configuration_space(cs=cs, policy='exp')
        return cs


class MomentumStepConstFeedNet(MomentumConstFeedNet):

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        cs = MomentumConstFeedNet.get_hyperparameter_search_space()
        cs = _lr_policy_configuration_space(cs=cs, policy='step')
        return cs
