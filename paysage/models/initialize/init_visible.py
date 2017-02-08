import numpy
from ... import backends as B

# ----- FUNCTIONS ----- #

def hinton(batch, model):
    """
        Hinton says to initalize the weights from N(0, 0.001)
        visible_bias = inverse_mean( \< v_i \> )

        Hinton, Geoffrey.
        "A practical guide to training restricted Boltzmann machines."
        Momentum 9.1 (2010): 926.

    """
    nvis = len(model.params['visible_bias'])
    model.params['weights'] = numpy.random.normal(loc=0.0, scale=0.001,
                                size=(nvis, nvis)).astype(dtype=numpy.float32)

    x = numpy.zeros(batch.ncols, dtype=numpy.float32)
    nsamples = 0
    while True:
        try:
            v_data = batch.get(mode='train')
        except StopIteration:
            break
        x += B.msum(v_data, axis=0)
        nsamples += len(v_data)

    model.params['visible_bias'] = model.layers['visible'].inverse_mean(x/nsamples)

def mean_field(batch, model):
    """
    Yasser Roudi, Erik Aurell, John A. Hertz.
    "Statistical physics of pairwise probability models"
    https://arxiv.org/pdf/0905.1410.pdf
    """
    nvis = len(model.params['visible_bias'])
    x = numpy.zeros(nvis, dtype=numpy.float32)
    x2 = numpy.zeros((nvis,nvis), dtype=numpy.float32)
    nsamples = 0
    while True:
        try:
            v_data = batch.get(mode='train')
        except StopIteration:
            break
        x += B.msum(v_data, axis=0)
        x2 += B.dot(v_data.T, v_data)
        nsamples += len(v_data)

    cov = x2/nsamples - (x/nsamples)**2
    J = -numpy.linalg.pinv(cov)
    numpy.fill_diagonal(J, 0)
    model.params['weights'] = J
    model.params['visible_bias'] = model.layers['visible'].inverse_mean(x/nsamples)
    model.params['visible_bias'] -= B.dot(J, x/nsamples)

def jacquin_rancon(batch, model):
    """
    Jacquin, Hugo, and A. Rançon.
    "Resummed mean-field inference for strongly coupled data."
    Physical Review E 94.4 (2016): 042118.
    """
    nvis = len(model.params['visible_bias'])
    x = numpy.zeros(nvis, dtype=numpy.float32)
    x2 = numpy.zeros((nvis,nvis), dtype=numpy.float32)
    nsamples = 0
    while True:
        try:
            v_data = batch.get(mode='train')
        except StopIteration:
            break
        x += B.msum(v_data, axis=0)
        x2 += B.dot(v_data.T, v_data)
        nsamples += len(v_data)

    cov = x2/nsamples - (x/nsamples)**2
    J = -numpy.linalg.pinv(cov)
    numpy.fill_diagonal(J, 0)
    model.params['weights'] = J
    model.params['visible_bias'] = model.layers['visible'].inverse_mean(x/nsamples)
    model.params['visible_bias'] -= B.dot(J, x/nsamples)
