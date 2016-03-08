from HPOlibConfigSpace.hyperparameters import Constant, UniformFloatHyperparameter

from component import ConstrainedFeedNet


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
