from . import backends as be

#----- LAYER CLASSES -----#

class GaussianLayer(object):

    def __init__(self, num_units):
        self.len = num_units
        self.params = {
        'loc': be.zeros(self.len),
        'log_var': be.zeros(self.len)
        }
        self.rand = be.randn

    def prox(self, vis):
        return vis

    def mean(self, loc):
        return loc

    def inverse_mean(self, mean):
        return mean

    def log_partition_function(self, scale):
        return -be.log(scale)

    def sample_state(self, loc, scale):
        r = be.float_tensor(self.rand(be.shape(loc)))
        return loc + be.broadcast(scale, r) * r

    def random(self, array_or_shape):
        try:
            r = be.float_tensor(self.rand(be.shape(array_or_shape)))
        except AttributeError:
            r = be.float_tensor(self.rand(array_or_shape))
        return r


class IsingLayer(object):

    def __init__(self, num_units):
        self.len = num_units
        self.params = {
        'loc': be.zeros(self.len)
        }
        self.rand = be.rand

    def prox(self, vis):
        return 2.0 * be.float_tensor(vis > 0.0) - 1.0

    def mean(self, loc):
        return be.tanh(loc)

    def inverse_mean(self, mean):
        return be.atanh(mean)

    def log_partition_function(self, loc):
        return be.logcosh(loc)

    def sample_state(self, loc):
        p = be.expit(loc)
        r = self.rand(be.shape(p))
        return 2*be.float_tensor(r<p)-1

    def random(self, array_or_shape):
        try:
            r = self.rand(be.shape(array_or_shape))
        except AttributeError:
            r = self.rand(array_or_shape)
        return 2*be.float_tensor(r<0.5)-1


class BernoulliLayer(object):

    def __init__(self, num_units):
        self.len = num_units
        self.params = {
        'loc': be.zeros(self.len)
        }
        self.rand = be.rand

    def prox(self, vis):
        return be.float_tensor(vis > 0.0)

    def mean(self, loc):
        return be.expit(loc)

    def inverse_mean(self, mean):
        return be.logit(mean)

    def log_partition_function(self, loc):
        return be.softplus(loc)

    def sample_state(self, loc):
        p = be.expit(loc)
        r = self.rand(be.shape(p))
        return be.float_tensor(r<p)

    def random(self, array_or_shape):
        try:
            r = self.rand(be.shape(array_or_shape))
        except AttributeError:
            r = self.rand(array_or_shape)
        return be.float_tensor(r<0.5)

class ExponentialLayer(object):

    def __init__(self, num_units):
        self.len = num_units
        self.params = {
        'loc': be.zeros(self.len)
        }
        self.rand = be.rand

    def prox(self, vis):
        return be.clip(vis, a_min=be.EPSILON)

    def mean(self, loc):
        return be.repicrocal(loc)

    def inverse_mean(self, mean):
        return be.reciprocal(mean)

    def log_partition_function(self, loc):
        return -be.log(loc)

    def sample_state(self, loc):
        r = self.rand(be.shape(loc))
        return -be.log(r) / loc

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
