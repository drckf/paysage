from .. import layers
from .. import backends as be
from ..models.initialize import init_hidden as init
from .. import constraints
from .. import penalties

class Weights(object):

    def __init__(self, shape):
        self.shape = shape
        self.val = 0.01 * be.randn(shape)
        self.derivs = {'val': None}
        self.decay = None
        self.constraint = None

    def add_constraint(self, constraint):
        self.constraint = constraint

    def enforce_constraints(self):
        if self.constraint is not None:
            getattr(constraints, self.constraint)(self.val)

    def add_decay(self, penalty, method='l2_penalty'):
        self.decay = getattr(penalties, method)(penalty)


class Model(object):

    def __init__(self, vis_layer, hid_layer):
        self.layers = {
        'visible': vis_layer,
        'hidden': hid_layer
        }
        self.weights = Weights((vis_layer.len, hid_layer.len))

    def add_weight_constraint(self, constraint):
        self.weights.add_constraint(constraint)

    def enforce_constraints(self):
        self.weights.enforce_constraints()

    def add_weight_decay(self, penalty, method='l2_penalty'):
        self.weights.add_decay(penalty, method)

    def initialize(self, data, method='hinton'):
        try:
            func = getattr(init, method)
        except AttributeError:
            print(method + ' is not a valid initialization method for latent models')
        func(data, self)
        self.enforce_constraints()

    def random(self, visible):
        return self.layers['visible'].random(visible)

    def mcstep(self, vis, beta=None):
        """mcstep(v):
           v -> h -> v'
           return v'

        """
        self.layers['hidden'].update(vis, self.weights.val, beta)
        h = self.layers['hidden'].sample_state()
        self.layers['visible'].update(h, be.transpose(self.weights.val), beta)
        return self.layers['visible'].sample_state()

    def markov_chain(self, vis, steps, beta=None):
        """markov_chain(v, n):
           v -> h -> v_1 -> h_1 -> ... -> v_n
           return v_n

        """
        new_vis = be.float_tensor(vis)
        for t in range(steps):
            new_vis = self.mcstep(new_vis, beta)
        return new_vis

    def mean_field_step(self, vis, beta=None):
        """mean_field_step(v):
           v -> h -> v'
           return v'

        """
        self.layers['hidden'].update(vis, self.weights.val, beta)
        h = self.layers['hidden'].mean()
        self.layers['visible'].update(h, be.transpose(self.weights.val), beta)
        return self.layers['visible'].mean()

    def mean_field_iteration(self, vis, steps, beta=None):
        """mean_field_iteration(v, n):
           v -> h -> v_1 -> h_1 -> ... -> v_n
           return v_n

        """
        new_vis = be.float_tensor(vis)
        for t in range(steps):
            new_vis = self.mean_field_step(new_vis, beta)
        return new_vis

    def deterministic_step(self, vis, beta=None):
        """deterministic_step(v):
           v -> h -> v'
           return v'

        """
        self.layers['hidden'].update(vis, self.weights.val, beta)
        h = self.layers['hidden'].mode()
        self.layers['visible'].update(h, be.transpose(self.weights.val), beta)
        return self.layers['visible'].mode()

    def deterministic_iteration(self, vis, steps, beta=None):
        """mean_field_iteration(v, n):
           v -> h -> v_1 -> h_1 -> ... -> v_n
           return v_n

        """
        new_vis = be.float_tensor(vis)
        for t in range(steps):
            new_vis = self.deterministic_step(new_vis, beta)
        return new_vis

    def derivatives(self, visible):
        # calling self.layers['visible'].derivatives has two effects:
        # 1) it updates the parameters of the hidden layer using the visible
        # observations.
        # 2) it updates the derivs attribute of the visible layer
        self.layers['visible'].derivatives(visible,
                                           self.layers['hidden'],
                                           self.weights.val,
                                           beta=None
                                           )
        # calling self.layers['hidden'].mean after computing the derivatives
        # of the visible layer ensures that the ext_parameters of the hidden
        # layer are already up to date
        hid = self.layers['hidden'].mean()
        # calling self.layers['hidden'].derivatives has two effects:
        # 1) it updates the parameters of the visible layer using the hidden
        # observations (hid).
        # 2) it updates the derivs attribute of the hidden layer
        self.layers['hidden'].derivatives(hid,
                                          self.layers['visible'],
                                          be.transpose(self.weights.val),
                                          beta=None
                                          )
        # we need rescaled visible and hidden observations to compute the
        # derivatives of the weights -- this only has an effect for layers
        # that have a scale parameter
        hid = self.layers['hidden'].rescale(hid)
        vis = self.layers['visible'].rescale(visible)
        # return the derivative taken with respect to the model weights
        self.weights.derivs['val'] = -be.batch_outer(vis, hid) / len(visible)
