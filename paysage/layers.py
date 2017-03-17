from . import backends as be


class Layer(object):

    def __init__(self, *args, **kwargs):
        self.int_params = {}
        self.penalties = {}
        self.constraints = {}

    def add_constraint(self, constraint):
        self.constraint.update(constraint)

    def enforce_constraints(self):
        for param_name in self.constraints:
            self.constraint[param_name](self.int_params[param_name])

    def add_penalty(self, penalty):
        self.penalties.update(penalty)

    def get_penalties(self):
        for param_name in self.penalties:
            self.penalties[param_name].value(self.int_params[param_name])

    def get_penalty_gradients(self):
        pen = {param_name:
            self.penalties[param_name].grad(self.int_params[param_name])
            for param_name in self.penalties}
        return pen

    def parameter_step(self, deltas):
        be.subtract_dicts_inplace(self.int_params, deltas)
        self.enforce_constraints()


class Weights(Layer):

    def __init__(self, shape):
        super().__init__()

        self.shape = shape

        # simple weight layers only have a single internal parameter matrix
        # they have no external parameters because they
        # do not depend on the state of anything else
        self.int_params = {
        'matrix': 0.01 * be.randn(shape)
        }

    def W(self):
        # the W method provides a reference to the weight matrix
        # it is just for convenience so that we don't have to
        # type out the whole thing every time
        return self.int_params['matrix']

    def derivatives(self, first_layer_scaled, second_layer_scaled):
        n = len(first_layer_scaled)
        derivs = {
        'matrix': -be.batch_outer(first_layer_scaled, second_layer_scaled) / n
        }
        be.add_dicts_inplace(derivs, self.get_penalty_gradients())
        return derivs



class GaussianLayer(Layer):

    def __init__(self, num_units):
        super().__init__()

        self.len = num_units
        self.sample_size = 0
        self.rand = be.randn

        self.int_params = {
        'loc': be.zeros(self.len),
        'log_var': be.zeros(self.len)
        }

        self.ext_params = {
        'mean': None,
        'variance': None
        }

    def online_param_update(self, data):
        n = len(data)
        new_sample_size = n + self.sample_size
        # compute the current value of the second moment
        x2 = be.exp(self.int_params['log_var'])
        x2 += self.int_params['loc']**2
        # update the first moment / location parameter
        self.int_params['loc'] *= self.sample_size / new_sample_size
        self.int_params['loc'] += n * be.mean(data, axis=0) / new_sample_size
        # update the second moment
        x2 *= self.sample_size / new_sample_size
        x2 += n * be.mean(be.square(data), axis=0) / new_sample_size
        # update the log_var parameter from the second moment
        self.int_params['log_var'] = be.log(x2 - self.int_params['loc']**2)
        # update the sample size
        self.sample_size = new_sample_size

    def update(self, units, weights, beta=None):
        self.ext_params['mean'] = be.dot(units, weights)
        if beta is not None:
            self.ext_params['mean'] *= be.broadcast(
                                       beta,
                                       self.ext_params['mean']
                                       )
        self.ext_params['mean'] += be.broadcast(
                                   self.int_params['loc'],
                                   self.ext_params['mean']
                                   )
        self.ext_params['variance'] = be.broadcast(
                                      be.exp(self.int_params['log_var']),
                                      self.ext_params['mean']
                                      )

    def derivatives(self, observations, connected_layer, weights, beta=None):
        derivs = {
        'loc': be.zeros(self.len),
        'log_var': be.zeros(self.len)
        }

        connected_layer.update(observations, weights, beta)
        connected_mean_scaled = connected_layer.rescale(connected_layer.mean())

        self.update(connected_mean_scaled, weights, beta)
        v_scaled = self.rescale(observations)

        derivs['loc'] = -be.mean(v_scaled, axis=0)

        diff = be.square(
        observations - be.broadcast(self.int_params['loc'], observations)
        )
        derivs['log_var'] = -0.5 * be.mean(diff, axis=0)
        derivs['log_var'] += be.batch_dot(
                             connected_mean_scaled,
                             be.transpose(weights),
                             observations,
                             axis=0
                             ) / len(observations)
        derivs['log_var'] = self.rescale(derivs['log_var'])

        be.add_dicts_inplace(derivs, self.get_penalty_gradients())
        return derivs

    def rescale(self, observations):
        scale = be.exp(self.int_params['log_var'])
        return observations / be.broadcast(scale, observations)

    def mode(self):
        return self.ext_params['mean']

    def mean(self):
        return self.ext_params['mean']

    def sample_state(self):
        r = be.float_tensor(self.rand(be.shape(self.ext_params['loc'])))
        return self.ext_params['loc'] + be.sqrt(self.ext_params['variance'])*r

    def random(self, array_or_shape):
        try:
            r = be.float_tensor(self.rand(be.shape(array_or_shape)))
        except AttributeError:
            r = be.float_tensor(self.rand(array_or_shape))
        return r


class IsingLayer(Layer):

    def __init__(self, num_units):
        super().__init__()

        self.len = num_units
        self.sample_size = 0
        self.rand = be.rand

        self.int_params = {
        'loc': be.zeros(self.len)
        }

        self.ext_params = {
        'field': None
        }

    def online_param_update(self, data):
        n = len(data)
        new_sample_size = n + self.sample_size
        # update the first moment
        x = be.tanh(self.int_params['loc'])
        x *= self.sample_size / new_sample_size
        x += n * be.mean(data, axis=0) / new_sample_size
        # update the location parameter
        self.int_params['loc'] = be.atanh(x)
        # update the sample size
        self.sample_size = new_sample_size

    def update(self, units, weights, beta=None):
        self.ext_params['field'] = be.dot(units, weights)
        if beta is not None:
            self.ext_params['field'] *= be.broadcast(
                                        beta,
                                        self.ext_params['field']
                                        )
        self.ext_params['field'] += be.broadcast(
                                    self.int_params['loc'],
                                    self.ext_params['field']
                                    )

    def derivatives(self, observations, connected_layer, weights, beta=None):
        derivs = {
        'loc': be.zeros(self.len)
        }
        connected_layer.update(observations, weights, beta)
        connected_mean_scaled = connected_layer.rescale(connected_layer.mean())
        self.update(connected_mean_scaled, weights, beta)
        v_scaled = self.rescale(observations)
        derivs['loc'] = -be.mean(v_scaled, axis=0)
        be.add_dicts_inplace(derivs, self.get_penalty_gradients())
        return derivs

    def rescale(self, observations):
        return observations

    def mode(self):
        return 2 * be.float_tensor(self.ext_params['field'] > 0) - 1

    def mean(self):
        return be.tanh(self.ext_params['field'])

    def sample_state(self):
        p = be.expit(self.ext_params['field'])
        r = self.rand(be.shape(p))
        return 2 * be.float_tensor(r < p) - 1

    def random(self, array_or_shape):
        try:
            r = self.rand(be.shape(array_or_shape))
        except AttributeError:
            r = self.rand(array_or_shape)
        return 2 * be.float_tensor(r < 0.5) - 1


class BernoulliLayer(Layer):

    def __init__(self, num_units):
        super().__init__()

        self.len = num_units
        self.sample_size = 0
        self.rand = be.rand

        self.int_params = {
        'loc': be.zeros(self.len)
        }

        self.ext_params = {
        'field': None
        }

    def online_param_update(self, data):
        n = len(data)
        new_sample_size = n + self.sample_size
        # update the first moment
        x = be.expit(self.int_params['loc'])
        x *= self.sample_size / new_sample_size
        x += n * be.mean(data, axis=0) / new_sample_size
        # update the location parameter
        self.int_params['loc'] = be.logit(x)
        # update the sample size
        self.sample_size = new_sample_size

    def update(self, units, weights, beta=None):
        self.ext_params['field'] = be.dot(units, weights)
        if beta is not None:
            self.ext_params['field'] *= be.broadcast(
                                        beta,
                                        self.ext_params['field']
                                        )
        self.ext_params['field'] += be.broadcast(
                                    self.int_params['loc'],
                                    self.ext_params['field']
                                    )

    def derivatives(self, observations, connected_layer, weights, beta=None):
        derivs = {
        'loc': be.zeros(self.len)
        }
        connected_layer.update(observations, weights, beta)
        connected_mean_scaled = connected_layer.rescale(connected_layer.mean())
        self.update(connected_mean_scaled, be.transpose(weights), beta)
        scaled_observations = self.rescale(observations)
        derivs['loc'] = -be.mean(scaled_observations, axis=0)
        be.add_dicts_inplace(derivs, self.get_penalty_gradients())
        return derivs

    def rescale(self, observations):
        return observations

    def mode(self):
        return be.float_tensor(self.ext_params['field'] > 0.0)

    def mean(self):
        return be.expit(self.ext_params['field'])

    def sample_state(self):
        p = be.expit(self.ext_params['field'])
        r = self.rand(be.shape(p))
        return be.float_tensor(r < p)

    def random(self, array_or_shape):
        try:
            r = self.rand(be.shape(array_or_shape))
        except AttributeError:
            r = self.rand(array_or_shape)
        return be.float_tensor(r < 0.5)

class ExponentialLayer(Layer):

    def __init__(self, num_units):
        super().__init__()

        self.len = num_units
        self.sample_size = 0
        self.rand = be.rand

        self.int_params = {
        'loc': be.zeros(self.len)
        }

        self.ext_params = {
        'field': None
        }

    def online_param_update(self, data):
        n = len(data)
        new_sample_size = n + self.sample_size
        # update the first moment
        x = self.mean(self.int_params['loc'])
        x *= self.sample_size / new_sample_size
        x += n * be.mean(data, axis=0) / new_sample_size
        # update the location parameter
        self.int_params['loc'] = be.reciprocal(x)
        # update the sample size
        self.sample_size = new_sample_size

    def update(self, units, weights, beta=None):
        self.ext_params['field'] = be.dot(units, weights)
        if beta is not None:
            self.ext_params['field'] *= be.broadcast(
                                        beta,
                                        self.ext_params['field']
                                        )
        self.ext_params['field'] += be.broadcast(
                                    self.int_params['loc'],
                                    self.ext_params['field']
                                    )

    def derivatives(self, observations, connected_layer, weights, beta=None):
        derivs = {
        'loc': be.zeros(self.len)
        }
        connected_layer.update(observations, weights, beta)
        connected_mean_scaled = connected_layer.rescale(connected_layer.mean())
        self.update(connected_mean_scaled, weights, beta)
        v_scaled = self.rescale(observations)
        derivs['field'] = -be.mean(v_scaled, axis=0)
        be.add_dicts_inplace(derivs, self.get_penalty_gradients())
        return derivs

    def rescale(self, observations):
        return observations

    def mode(self):
        raise NotImplementedError("Exponential distribution has no mode.")

    def mean(self):
        return be.repicrocal(self.ext_params['field'])

    def sample_state(self):
        r = self.rand(be.shape(self.ext_params['field']))
        return -be.log(r) / self.ext_params['field']

    def random(self, array_or_shape):
        try:
            r = self.rand(be.shape(array_or_shape))
        except AttributeError:
            r = self.rand(array_or_shape)
        return -be.log(r)


# ---- FUNCTIONS ----- #

def get(key):
    if 'gauss' in key.lower():
        return GaussianLayer
    elif 'ising' in key.lower():
        return IsingLayer
    elif 'bern' in key.lower():
        return BernoulliLayer
    elif 'expo' in key.lower():
        return ExponentialLayer
    else:
        raise ValueError('Unknown layer type')
