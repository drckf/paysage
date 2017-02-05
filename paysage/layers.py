import numpy
from . import backends as B

#----- LAYER CLASSES -----#

class GaussianLayer(object):

    def __init__(self):
        self.rand = numpy.random.randn

    def prox(self, vis):
        return vis

    def mean(self, loc):
        return loc

    def inverse_mean(self, mean):
        return mean

    def log_partition_function(self, scale):
        return -B.log(scale)

    def sample_state(self, loc, scale):
        r = numpy.float32(self.rand(*loc.shape))
        return loc + scale * r

    def random(self, array_or_shape):
        try:
            r = numpy.float32(self.rand(*array_or_shape.shape))
        except AttributeError:
            r = numpy.float32(self.rand(*array_or_shape))
        return r


class IsingLayer(object):

    def __init__(self):
        self.rand = numpy.random.rand

    def prox(self, vis):
        return 2.0 * (vis > 0.0).astype(numpy.float32) - 1.0

    def mean(self, loc):
        return B.tanh(loc)

    def inverse_mean(self, mean):
        return B.atanh(mean)

    def log_partition_function(self, loc):
        return B.logcosh(loc)

    def sample_state(self, loc):
        p = B.expit(loc)
        r = self.rand(*p.shape)
        return 2*numpy.float32(r<p)-1

    def random(self, array_or_shape):
        try:
            r = self.rand(*array_or_shape.shape)
        except AttributeError:
            r = self.rand(*array_or_shape)
        return 2*numpy.float32(r<0.5)-1


class BernoulliLayer(object):

    def __init__(self):
        self.rand = numpy.random.rand

    def prox(self, vis):
        return (vis > 0.0).astype(numpy.float32)

    def mean(self, loc):
        return B.expit(loc)

    def inverse_mean(self, mean):
        return B.logit(mean)

    def log_partition_function(self, loc):
        return B.softplus(loc)

    def sample_state(self, loc):
        p = B.expit(loc)
        r = self.rand(*p.shape)
        return numpy.float32(r<p)

    def random(self, array_or_shape):
        try:
            r = self.rand(*array_or_shape.shape)
        except AttributeError:
            r = self.rand(*array_or_shape)
        return numpy.float32(r<0.5)

class ExponentialLayer(object):

    def __init__(self):
        self.rand = numpy.random.rand

    def prox(self, vis):
        return vis.clip(min=B.EPSILON)

    def mean(self, loc):
        return B.elementwise_inverse(loc)

    def inverse_mean(self, mean):
        return B.elementwise_inverse(mean)

    def log_partition_function(self, loc):
        return -B.log(loc)

    def sample_state(self, loc):
        r = self.rand(*loc.shape)
        return -B.log(r) / loc

    def random(self, array_or_shape):
        try:
            r = self.rand(*array_or_shape.shape)
        except AttributeError:
            r = self.rand(*array_or_shape)
        return -B.log(r)


# ---- FUNCTIONS ----- #

def get(key):
    if 'gauss' in key.lower():
        return GaussianLayer()
    elif 'ising' in key.lower():
        return IsingLayer()
    elif 'bern' in key.lower():
        return BernoulliLayer()
    elif 'expo' in key.lower():
        return ExponentialLayer()
    else:
        raise ValueError('Unknown layer type')
