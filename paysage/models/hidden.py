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
        layers.Weights((self.layers[i].len, self.layers[i+1].len))
        for i in range(len(self.layers) - 1)
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
        self.layers[i+1].update(vis, self.weights[i].W(), beta)
        h = self.layers[i+1].sample_state()
        self.layers[i].update(h, be.transpose(self.weights[i].W()), beta)
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
        self.layers[i+1].update(vis, self.weights[i].W(), beta)
        h = self.layers[i+1].mean()
        self.layers[i].update(h, be.transpose(self.weights[i].W()), beta)
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
        self.layers[i + 1].update(vis, self.weights[i].W(), beta)
        h = self.layers[i + 1].mode()
        self.layers[i].update(h, be.transpose(self.weights[i].W()), beta)
        return self.layers[i].mode()

    def deterministic_iteration(self, vis, steps, beta=None):
        """mean_field_iteration(v, n):
           v -> h -> v_1 -> h_1 -> ... -> v_n
           return v_n

        """
        new_vis = be.float_tensor(vis)
        for _ in range(steps):
            new_vis = self.deterministic_step(new_vis, beta)
        return new_vis

    def gradient(self, observed, sampled):
        i = 0

        grad = {
        'layers': [None for l in self.layers],
        'weights': [None for w in self.weights]
        }

        # POSITIVE PHASE (using observed)

        # update hidden layer external parameters
        self.layers[i+1].update(observed, self.weights[0].W(), beta=None)

        # (gaussian only) compute scaled mean of hidden layer
        # compute visible layer gradient
        grad['layers'][i] = self.layers[i].derivatives(observed,
                                           self.layers[i + 1],
                                           self.weights[i].W(),
                                           beta=None
                                           )

        # store hidden layer mean
        hid = self.layers[i + 1].mean()

        # update visible layer external parameters
        self.layers[i].update(hid, be.transpose(self.weights[0].W()), beta=None)

        # (gaussian only) compute scaled mean of visible layer
        # compute hidden layer gradient
        grad['layers'][i+1] = self.layers[i + 1].derivatives(hid,
                                                 self.layers[i],
                                                 be.transpose(
                                                    self.weights[i].W()),
                                                 beta=None
                                                 )

        # store the scaled mean of the hidden layer
        # store the scaled visible observations
        hid = self.layers[i + 1].rescale(hid)
        vis = self.layers[i].rescale(observed)

        # compute the gradient of the weights
        grad['weights'][i] = self.weights[i].derivatives(vis, hid)

        # NEGATIVE PHASE (using sampled)

        # update hidden layer external parameters
        self.layers[i+1].update(sampled, self.weights[0].W(), beta=None)

        # (gaussian only) compute scaled mean of hidden layer
        # compute visible layer gradient
        be.subtract_dicts_inplace(grad['layers'][i],
                                  self.layers[i].derivatives(sampled,
                                                 self.layers[i + 1],
                                                 self.weights[i].W(),
                                                 beta=None
                                                 )

                                  )
        # store hidden layer mean
        hid = self.layers[i + 1].mean()

        # (gaussian only) compute scaled mean of visible layer
        # compute hidden layer gradient
        be.subtract_dicts_inplace(grad['layers'][i+1],
                                  self.layers[i + 1].derivatives(hid,
                                                     self.layers[i],
                                                     be.transpose(
                                                        self.weights[i].W()),
                                                     beta=None
                                                     )
                                  )
        # store the scaled mean of the hidden layer
        # store the scaled visible observations
        hid = self.layers[i + 1].rescale(hid)
        vis = self.layers[i].rescale(sampled)

        # compute the gradient of the weights
        be.subtract_dicts_inplace(grad['weights'][i],
                                  self.weights[i].derivatives(vis, hid)
                                  )

        return grad

    def parameter_update(self, deltas):
        for i in range(len(self.layers)):
            self.layers[i].parameter_step(deltas['layers'][i])
        for i in range(len(self.weights)):
            self.weights[i].parameter_step(deltas['weights'][i])

    def joint_energy(self, visible, hidden):
        energy = 0
        i = 0
        energy += self.layers[i].energy(visible)
        energy += self.weights[i].energy(visible, hidden)
        energy += self.layers[i+1].energy(hidden)
        energy += self.weights[i+1].energy(hidden, visible)
        return energy

    def marginal_free_energy(self, visible):
        i = 0
        log_Z_hidden = self.layers[i+1].log_partition_function()
        energy = - be.tsum(log_Z_hidden, axis=1)
        energy -= be.dot(visible, self.layers[i].int_params['loc'])
        return energy
