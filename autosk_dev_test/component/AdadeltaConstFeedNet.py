from HPOlibConfigSpace.hyperparameters import Constant, UniformFloatHyperparameter

from component import ConstrainedFeedNet


class AdadeltaConstFeedNet(ConstrainedFeedNet):

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        cs = ConstrainedFeedNet.get_hyperparameter_search_space()
        solver = Constant(name='solver', value='adadelta')
        cs.add_hyperparameter(solver)

        rho = UniformFloatHyperparameter('rho', 0.0, 1.0, default=0.95)
        cs.add_hyperparameter(rho)
        return cs