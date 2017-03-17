from .. import layers
from .. import backends as be
from ..models.initialize import init_hidden as init


class Model(object):

    def __init__(self, vis_layer, hid_layer):

        # the layers are stored in a list with the visible units
        # as the zeroth element
        self.layers = [
        vis_layer,
        hid_layer
        ]

        # adjacent layers are connected by weights
        # therefore, if there are len(layers) = n then len(weights) = n - 1
        self.weights = [
        layers.Weights((vis_layer.len, hid_layer.len))
        ]

    def initialize(self, data, method='hinton'):
        try:
            func = getattr(init, method)
        except AttributeError:
            print(method + ' is not a valid initialization method for latent models')
        func(data, self)
        for l in self.layers:
            l.enforce_constraints()
        for w in self.weights:
            w.enforce_constraints()

    def random(self, visible):
        return self.layers[0].random(visible)

    def mcstep(self, vis, beta=None):
        """mcstep(v):
           v -> h -> v'
           return v'

        """
        i = 0
        self.layers[i+1].update(vis, self.weights[i].val, beta)
        h = self.layers[i+1].sample_state()
        self.layers[i].update(h, be.transpose(self.weights[i].val), beta)
        return self.layers[i].sample_state()

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
        i = 0
        self.layers[i+1].update(vis, self.weights[i].val, beta)
        h = self.layers[i+1].mean()
        self.layers[i].update(h, be.transpose(self.weights[i].val), beta)
        return self.layers[i].mean()

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
        i = 0
        self.layers[i + 1].update(vis, self.weights[i].val, beta)
        h = self.layers[i + 1].mode()
        self.layers[i].update(h, be.transpose(self.weights[i].val), beta)
        return self.layers[i].mode()

    def deterministic_iteration(self, vis, steps, beta=None):
        """mean_field_iteration(v, n):
           v -> h -> v_1 -> h_1 -> ... -> v_n
           return v_n

        """
        new_vis = be.float_tensor(vis)
        for t in range(steps):
            new_vis = self.deterministic_step(new_vis, beta)
        return new_vis

    def gradient(self, observed, sampled):
        i = 0

        grad = {
        'layers': [None for l in self.layers],
        'weights': [None for w in self.weights]
        }


        # POSITIVE PHASE (using observed)

        # calling self.layers['visible'].derivatives has two effects:
        # 1) it updates the parameters of the hidden layer using the visible
        # observations.
        # 2) it updates the derivs attribute of the visible layer
        grad['layers'][i] = self.layers[i].derivatives(observed,
                                           self.layers[i + 1],
                                           self.weights[i].val,
                                           beta=None
                                           )
        # calling self.layers['hidden'].mean after computing the derivatives
        # of the visible layer ensures that the ext_parameters of the hidden
        # layer are already up to date
        hid = self.layers[i + 1].mean()
        # calling self.layers['hidden'].derivatives has two effects:
        # 1) it updates the parameters of the visible layer using the hidden
        # observations (hid).
        # 2) it updates the derivs attribute of the hidden layer
        grad['layers'][i+1] = self.layers[i + 1].derivatives(hid,
                                                 self.layers[i],
                                                 be.transpose(
                                                    self.weights[i].val),
                                                 beta=None
                                                 )
        # we need rescaled visible and hidden observations to compute the
        # derivatives of the weights -- this only has an effect for layers
        # that have a scale parameter
        hid = self.layers[i + 1].rescale(hid)
        vis = self.layers[i].rescale(observed)
        grad['weights'][i] = self.weights[i].derivatives(vis, hid)

        # NEGATIVE PHASE (using sampled)

        # calling self.layers['visible'].derivatives has two effects:
        # 1) it updates the parameters of the hidden layer using the visible
        # observations.
        # 2) it updates the derivs attribute of the visible layer
        grad['layers'][i] -= self.layers[i].derivatives(sampled,
                                            self.layers[i + 1],
                                            self.weights[i].val,
                                            beta=None
                                            )
        # calling self.layers['hidden'].mean after computing the derivatives
        # of the visible layer ensures that the ext_parameters of the hidden
        # layer are already up to date
        hid = self.layers[i + 1].mean()
        # calling self.layers['hidden'].derivatives has two effects:
        # 1) it updates the parameters of the visible layer using the hidden
        # observations (hid).
        # 2) it updates the derivs attribute of the hidden layer
        grad['layers'] -= self.layers[i + 1].derivatives(hid,
                                             self.layers[i],
                                             be.transpose(
                                                self.weights[i].val),
                                             beta=None
                                             )
        # we need rescaled visible and hidden observations to compute the
        # derivatives of the weights -- this only has an effect for layers
        # that have a scale parameter
        hid = self.layers[i + 1].rescale(hid)
        vis = self.layers[i].rescale(sampled)
        grad['weights'][i] = self.weights[i].derivatives(vis, hid)

        return grad
