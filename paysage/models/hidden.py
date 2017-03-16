from .. import layers
from .. import backends as be
from ..models.initialize import init_hidden as init
from .. import constraints
from .. import penalties


# ---- MODEL CLASSES ---- #

class Model(object):

    def __init__(self, vis_layer, hid_layer):
        self.layers = {
        'visible': vis_layer,
        'hidden': hid_layer
        }
        self.weights = 0.01 * be.randn((vis_layer.len, hid_layer.len))
        self.weight_constraint = None
        self.weight_decay = None

    def add_weight_constraint(self, constraint):
        self.weight_constraint = constraint

    def enforce_constraints(self):
        getattr(constraints, self.weight_constraint)(self.weights)

    def add_weight_decay(self, penalty, method='l2_penalty'):
        self.weight_decay = getattr(penalties, method)(penalty)

    def initialize(self, data, method='hinton'):
        try:
            func = getattr(init, method)
        except AttributeError:
            print('{} is not a valid initialization method for latent models'.format(method))
        func(data, self)
        self.enforce_constraints()

    def mcstep(self, vis, beta=None):
        """mcstep(v):
           v -> h -> v'
           return v'

        """
        self.layers['hidden'].update(vis, self.weights, beta)
        h = self.layers['hidden'].sample_state()
        self.layers['visible'].update(h, be.transpose(self.weights), beta)
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
        self.layers['hidden'].update(vis, self.weights, beta)
        h = self.layers['hidden'].mean()
        self.layers['visible'].update(h, be.transpose(self.weights), beta)
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
        self.layers['hidden'].update(vis, self.weights, beta)
        h = self.layers['hidden'].mode()
        self.layers['visible'].update(h, be.transpose(self.weights), beta)
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

    def random(self, visible):
        return self.layers['visible'].random(visible)


'''
class LatentModel(object):
    """LatentModel
       Abstract class for a 2-layer neural network.

    """
    def __init__(self):
        self.layers = {}
        self.params = {}
        self.constraints = {}
        self.penalty = {}

    # placeholder function -- defined in each model
    def sample_hidden(self, visible, beta=None):
        pass

    # placeholder function -- defined in each model
    def sample_visible(self, hidden, beta=None):
        pass

    # placeholder function -- defined in each model
    def marginal_free_energy(self, visible, beta=None):
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
           v -> h -> v'
           return v'

        """
        hid = self.sample_hidden(vis, beta)
        return self.sample_visible(hid, beta)

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

           It may be worth looking into extended approaches:
           Gabrié, Marylou, Eric W. Tramel, and Florent Krzakala.
           "Training Restricted Boltzmann Machine via the￼
           Thouless-Anderson-Palmer free energy."
           Advances in Neural Information Processing Systems. 2015.

        """
        hid = self.hidden_mean(vis, beta)
        return self.visible_mean(hid, beta)

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
        hid = self.hidden_mode(vis, beta)
        return self.visible_mode(hid, beta)

    def deterministic_iteration(self, vis, steps, beta=None):
        """mean_field_iteration(v, n):
           v -> h -> v_1 -> h_1 -> ... -> v_n
           return v_n

        """
        new_vis = be.float_tensor(vis)
        for t in range(steps):
            new_vis = self.deterministic_step(new_vis, beta)
        return new_vis

    def random(self, visible):
        return self.layers['visible'].random(visible)



class RestrictedBoltzmannMachine(LatentModel):
    """RestrictedBoltzmanMachine

       Hinton, Geoffrey.
       "A practical guide to training restricted Boltzmann machines."
       Momentum 9.1 (2010): 926.

    """
    def __init__(self, nvis, nhid, vis_type='ising', hid_type='bernoulli'):
        assert vis_type in ['ising', 'bernoulli']
        assert hid_type in ['ising', 'bernoulli']

        super().__init__()

        self.nvis = nvis
        self.nhid = nhid

        self.layers['visible'] = layers.get(vis_type)(nvis)
        self.layers['hidden'] = layers.get(hid_type)(nhid)
        self.params['weights'] = 0.01 * be.randn((nvis, nhid))

    def initialize(self, data, method='hinton'):
        try:
            func = getattr(init, method)
        except AttributeError:
            print('{} is not a valid initialization method for latent models'.format(method))
        func(data, self)
        self.enforce_constraints()

    def _hidden_field(self, visible, beta=None):
        result = be.dot(visible, self.params['weights'])
        if beta is not None:
            result *= be.broadcast(beta, result)
        result += be.broadcast(self.params['hidden_bias'], result)
        return result

    def _visible_field(self, hidden, beta=None):
        result = be.dot(hidden, be.transpose(self.params['weights']))
        if beta is not None:
            result *= be.broadcast(beta, result)
        result += be.broadcast(self.params['visible_bias'], result)
        return result

    def sample_hidden(self, visible, beta=None):
        return self.layers['hidden'].sample_state(self._hidden_field(visible, beta))

    def hidden_mean(self, visible, beta=None):
        return self.layers['hidden'].mean(self._hidden_field(visible, beta))

    def hidden_mode(self, visible, beta=None):
        return self.layers['hidden'].prox(self._hidden_field(visible, beta))

    def sample_visible(self, hidden, beta=None):
        return self.layers['visible'].sample_state(self._visible_field(hidden, beta))

    def visible_mean(self, hidden, beta=None):
        return self.layers['visible'].mean(self._visible_field(hidden, beta))

    def visible_mode(self, hidden, beta=None):
        return self.layers['visible'].prox(self._visible_field(hidden, beta))

    def derivatives(self, visible):
        mean_hidden = self.hidden_mean(visible, beta=None)
        derivs = {}
        derivs['visible_bias'] = -be.mean(visible, axis=0)
        derivs['hidden_bias'] = -be.mean(mean_hidden, axis=0)
        derivs['weights'] = -be.batch_outer(visible, mean_hidden) / len(visible)
        return derivs

    def joint_energy(self, visible, hidden, beta=None):
        energy = -be.batch_dot(visible, self.params['weights'], hidden)
        if beta is not None:
            energy *= be.flatten(beta)
        energy -= be.dot(visible, self.params['visible_bias'])
        energy -= be.dot(hidden, self.params['hidden_bias'])
        return be.mean(energy)

    def marginal_free_energy(self, visible, beta=None):
        log_Z_hidden = (self.layers['hidden']
             .log_partition_function(self._hidden_field(visible, beta=beta)))
        energy = - be.tsum(log_Z_hidden, axis=1)
        energy -= be.dot(visible, self.params['visible_bias'])
        return energy


class HopfieldModel(LatentModel):
    """HopfieldModel
       A model of associative memory with binary visible units and
       Gaussian hidden units.

       Hopfield, John J.
       "Neural networks and physical systems with emergent collective
       computational abilities."
       Proceedings of the national academy of sciences 79.8 (1982): 2554-2558.

    """
    def __init__(self, nvis, nhid, vis_type='ising'):
        assert vis_type in ['ising', 'bernoulli']

        super().__init__()

        self.nvis = nvis
        self.nhid = nhid

        self.layers['visible'] = layers.get(vis_type)(nvis)
        self.layers['hidden'] = layers.get('gaussian')(nhid)

        self.params['weights'] = 0.1 * be.randn((nvis, nhid))
        self.params['visible_bias'] = be.zeros(nvis)

        # the parameters of the hidden layer are not trainable
        self.hidden_bias = be.zeros(nhid)
        self.hidden_scale = be.ones(nhid)

    def initialize(self, data, method='hinton'):
        try:
            func = getattr(init, method)
        except AttributeError:
            print('{} is not a valid initialization method for latent models'.format(method))
        func(data, self)
        self.enforce_constraints()

    def _hidden_loc(self, visible, beta=None):
        result = be.dot(visible, self.params['weights'])
        if beta is not None:
            result *= be.broadcast(beta, result)
        result += be.broadcast(self.hidden_bias, result)
        return result

    def _visible_field(self, hidden, beta=None):
        result = be.dot(hidden, be.transpose(self.params['weights']))
        if beta is not None:
            result *= be.broadcast(beta, result)
        result += be.broadcast(self.params['visible_bias'], result)
        return result

    def sample_hidden(self, visible, beta=None):
        return self.layers['hidden'].sample_state(self._hidden_loc(visible, beta), self.hidden_scale)

    def hidden_mean(self, visible, beta=None):
        return self.layers['hidden'].mean(self._hidden_loc(visible, beta))

    def hidden_mode(self, visible, beta=None):
        return self.layers['hidden'].prox(self._hidden_loc(visible, beta))

    def sample_visible(self, hidden, beta=None):
        return self.layers['visible'].sample_state(self._visible_field(hidden, beta))

    def visible_mean(self, hidden, beta=None):
        return self.layers['visible'].mean(self._visible_field(hidden, beta))

    def visible_mode(self, hidden, beta=None):
        return self.layers['visible'].prox(self._visible_field(hidden,beta))

    def derivatives(self, visible):
        mean_hidden = self.hidden_mean(visible, beta=None)
        derivs = {}
        derivs['visible_bias'] = -be.mean(visible, axis=0)
        derivs['weights'] = -be.batch_outer(visible, mean_hidden) / len(visible)
        return derivs

    def joint_energy(self, visible, hidden, beta=None):
        energy = -be.batch_dot(visible, self.params['weights'], hidden)
        if beta is not None:
            energy *= be.flatten(beta)
        energy -= be.dot(visible, self.params['visible_bias'])
        energy -= be.tsum(hidden**2, axis=1)
        return be.mean(energy)

    def marginal_free_energy(self, visible, beta=None):
        J = be.dot(self.params['weights'], be.transpose(self.params['weights']))
        energy = -be.batch_dot(visible, J, visible)
        if beta is not None:
            energy *= be.flatten(beta)**2
        energy -= be.dot(visible, self.params['visible_bias'])
        return energy



class GaussianRestrictedBoltzmannMachine(LatentModel):
    """GaussianRestrictedBoltzmanMachine
       RBM with Gaussian visible units.

       Hinton, Geoffrey.
       "A practical guide to training restricted Boltzmann machines."
       Momentum 9.1 (2010): 926.

    """
    def __init__(self, nvis, nhid, hid_type='bernoulli'):
        assert hid_type in ['ising', 'bernoulli']

        super().__init__()

        self.nvis = nvis
        self.nhid = nhid

        self.layers['visible'] = layers.get('gaussian')(nvis)
        self.layers['hidden'] = layers.get(hid_type)(nhid)

        self.params['weights'] = 0.01 * be.randn((nvis, nhid))
        self.params['visible_bias'] = be.zeros(nvis)
        self.params['visible_scale'] = be.zeros(nvis)
        self.params['hidden_bias'] = be.zeros(nhid)

    def initialize(self, data, method='hinton'):
        try:
            func = getattr(init, method)
        except AttributeError:
            print('{} is not a valid initialization method for latent models'.format(method))
        func(data, self)
        self.enforce_constraints()

    def _hidden_field(self, visible, beta=None):
        scale = be.exp(self.params['visible_scale'])
        v_scaled = visible / be.broadcast(scale, visible)
        result = be.dot(v_scaled, self.params['weights'])
        if beta is not None:
            result *= be.broadcast(beta, result)
        result += be.broadcast(self.params['hidden_bias'], result)
        return result

    def _visible_loc(self, hidden, beta=None):
        result = be.dot(hidden, be.transpose(self.params['weights']))
        if beta is not None:
            result *= be.broadcast(beta, result)
        result += be.broadcast(self.params['visible_bias'], result)
        return result

    def sample_hidden(self, visible, beta=None):
        return self.layers['hidden'].sample_state(self._hidden_field(visible, beta))

    def hidden_mean(self, visible, beta=None):
        return self.layers['hidden'].mean(self._hidden_field(visible, beta))

    def hidden_mode(self, visible, beta=None):
        return self.layers['hidden'].prox(self._hidden_field(visible, beta))

    def sample_visible(self, hidden, beta=None):
        scale = be.exp(0.5 * self.params['visible_scale'])
        return self.layers['visible'].sample_state(self._visible_loc(hidden, beta), scale)

    def visible_mean(self, hidden, beta=None):
        return self.layers['visible'].mean(self._visible_loc(hidden, beta))

    def visible_mode(self, hidden, beta=None):
        return self.layers['visible'].prox(self._visible_loc(hidden,beta))

    def derivatives(self, visible):
        mean_hidden = self.hidden_mean(visible, beta=None)
        scale = be.exp(self.params['visible_scale'])
        v_scaled = visible / be.broadcast(scale, visible)
        derivs = {}
        derivs['visible_bias'] = -be.mean(v_scaled, axis=0)
        derivs['hidden_bias'] = -be.mean(mean_hidden, axis=0)
        derivs['weights'] = -be.batch_outer(v_scaled, mean_hidden) / len(visible)

        diff = (visible - be.broadcast(self.params['visible_bias'], visible))**2
        derivs['visible_scale'] = -0.5 * be.mean(diff, axis=0)
        derivs['visible_scale'] += be.batch_dot(mean_hidden, be.transpose(self.params['weights']), visible, axis=0) / len(visible)
        derivs['visible_scale'] /= scale
        return derivs

    def joint_energy(self, visible, hidden, beta=None):
        scale = be.exp(self.params['visible_scale'])
        v_scaled = visible / be.broadcast(scale, visible)
        energy = -be.batch_dot(v_scaled, self.params['weights'], hidden)
        if beta is not None:
            energy *= be.flatten(beta)
            diff = (visible - be.broadcast(self.params['visible_bias'], visible))**2
        diff /= be.broadcast(scale, diff)
        energy -= -0.5 * be.mean(diff, axis=1)
        energy -= be.dot(hidden, self.params['hidden_bias'])
        return be.mean(energy)

    def marginal_free_energy(self, visible, beta=None):
        raise NotImplementedError


# ----- ALIASES ----- #

BernoulliRBM = RBM = RestrictedBoltzmannMachine
GRBM = GaussianRBM = GaussianRestrictedBoltzmannMachine
'''
