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
        assert vis_type == 'ising'

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

    def _visible_field(self, vis, beta=None):
        self.visible_field = B.dot(vis, self.params['weights'])
        if isinstance(beta, numpy.ndarray):
            self.visible_field *= beta
        self.visible_field += self.params['visible_bias']
        return self.visible_field

    def visible_mean(self, vis, beta=None):
        return self.layers['visible'].mean(self._visible_field(vis, beta))

    def visible_mode(self, vis, beta=None):
        return self.layers['visible'].prox(self._visible_field(vis, beta))

    def sample_visible(self, vis, beta=None):
        new_vis = vis.astype(vis.dtype)
        self._visible_field(new_vis, beta)
        for j in range(self.nvis):
            # try to flip spin j (i.e., column j of the matrix)
            new_col = self.layers['visible'].sample_state(self.visible_field[:,j])
            diff = new_vis[:, j] - new_col
            # update the field and the spins
            if isinstance(beta, numpy.ndarray):
                self.visible_field -= beta * B.outer(diff, self.params['weights'][:,j])
            else:
                self.visible_field -= B.outer(diff, self.params['weights'][:,j])
            new_vis[:, j] = new_col
        return new_vis

    def derivatives(self, visible):
        derivs = {}
        if len(visible.shape) == 2:
            derivs['visible_bias'] = -B.mean(visible, axis=0)
            derivs['weights'] = -B.dot(visible.T, visible) / len(visible)
        else:
            derivs['visible_bias'] = -visible
            derivs['weights'] = -B.outer(visible, visible)
        return derivs

    def marginal_free_energy(self, visible, beta=None):
        energy = -B.batch_dot(visible, self.params['weights'], visible)
        if isinstance(beta, numpy.ndarray):
            energy *= numpy.ravel(beta)**2
        energy -= B.dot(visible, self.params['visible_bias'])
        return energy


# ----- ALIASES ----- #

BinaryMarkovRandomField = BinaryMRF = IsingModel
