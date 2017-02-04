import numpy
from .. import layers
from .. import backends as B
from ..models.initialize import init_visible as init
from .. import constraints
from .. import penalties

#---- MODEL CLASSES ----#

class VisibleModel(object):
    """VisibleModel
       Abstract class for a single-layer (fully visibile) neural network.

    """
    def __init__(self):
        self.layers = {}
        self.params = {}
        self.constraints = {}
        self.penalty = {}

    # placeholder function -- defined in each model
    def sample_visible(self, hidden, beta=None):
        pass

    def add_constraints(self, cons):
        for key in cons:
            assert key in self.params
            self.constraints[key] = cons[key]

    def enforce_constraints(self):
        for key in self.constraints:
            getattr(constraints, self.constraints[key])(self.params[key])

    def add_weight_decay(self, penalty, method='l2_penalty'):
        self.penalty.update({'weights': getattr(penalties, method)(penalty)})

    def mcstep(self, vis, beta=None):
        """mcstep(v):
           v -> v'
           return v'

        """
        return self.sample_visible(vis, beta)

    def markov_chain(self, vis, steps, beta=None):
        """gibbs_chain(v, n):
           v -> v_1 -> ... -> v_n
           return v_n

        """
        new_vis = vis.astype(vis.dtype)
        for t in range(steps):
            new_vis = self.mcstep(new_vis, beta)
        return new_vis

    def mean_field_step(self, vis, beta=None):
        """mean_field_step(v):
           v -> v'
           return v'
        """
        return self.visible_mean(vis, beta)

    def mean_field_iteration(self, vis, steps, beta=None):
        """mean_field_iteration(v, n):
           v -> v_1 -> ... -> v_n
           return v_n

        """
        new_vis = vis.astype(vis.dtype)
        for t in range(steps):
            new_vis = self.mean_field_step(new_vis, beta)
        return new_vis

    def deterministic_step(self, vis, beta=None):
        """deterministic_step(v):
           v -> v'
           return v'

        """
        return self.visible_mode(vis, beta)

    def deterministic_iteration(self, vis, steps, beta=None):
        """mean_field_iteration(v, n):
           v -> v_1 -> ... -> v_n
           return v_n

        """
        new_vis = vis.astype(vis.dtype)
        for t in range(steps):
            new_vis = self.deterministic_step(new_vis, beta)
        return new_vis

    def random(self, visible):
        return self.layers['visible'].random(visible)


class IsingModel(VisibleModel):
    """

    """
    def __init__(self, nvis, vis_type='ising'):
        assert vis_type in ['ising', 'bernoulli']

        super().__init__()

        self.nvis = nvis
        self.layers['visible'] = layers.get(vis_type)
        self.params['visible_bias'] = numpy.zeros(nvis, dtype=numpy.float32)
        self.params['weights'] = numpy.random.normal(loc=0.0, scale=0.01,
                                size=(nvis, nvis)).astype(dtype=numpy.float32)

    def initialize(self, data, method='hinton'):
        try:
            func = getattr(init, method)
        except AttributeError:
            print('{} is not a valid initialization method for latent models'.format(method))
        func(data, self)
        self.enforce_constraints()

    def _effective_field(self, vis):
        return self.params['visible_bias'] + numpy.dot(vis, self.params['weights'])




# ----- ALIASES ----- #

BinaryMarkovRandomField = BinaryMRF = IsingModel
