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
    numpy.fill_diagonal(model.params['weights'], 0)

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

def mean_field(batch, model,
               pseudocount=0.01):
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

    ave = (1 - pseudocount) * x / nsamples
    identity = numpy.identity(len(ave), ave.dtype)
    cov = pseudocount * identity + (1 - pseudocount) * x2 / nsamples
    cov -= B.outer(ave, ave)
    J = -numpy.linalg.inv(cov)
    numpy.fill_diagonal(J, 0)
    model.params['weights'] = J
    model.params['visible_bias'] = model.layers['visible'].inverse_mean(ave)
    model.params['visible_bias'] -= B.dot(J, ave)

def tap(batch, model,
        pseudocount=0.01,
        iterations=5):
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

    # compute the necessary moments
    ave = (1 - pseudocount) * x / nsamples
    out = B.outer(ave, ave)
    identity = numpy.identity(len(ave), ave.dtype)
    cov = pseudocount * identity + (1 - pseudocount) * x2 / nsamples
    cov -= out
    inv = numpy.linalg.inv(cov)

    # compute J_{ij} using Newton's method
    J = -inv
    for iteration in range(iterations):
        J -= (inv + J + 2 * out * J**2) / (1 + 4 * out * J)

    # remove the diagonal terms
    numpy.fill_diagonal(J, 0)
    model.params['weights'] = J

    # compute the fields
    model.params['visible_bias'] = model.layers['visible'].inverse_mean(ave)
    model.params['visible_bias'] -= B.dot(J, ave)
    model.params['visible_bias'] += B.dot(J**2, ave * (1-ave**2))

def jacquin_rancon(batch, model,
                   iterations=100,
                   pseudocount=numpy.float32(0.01),
                   damping=numpy.float32(0.9)):
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

    ave = (1 - pseudocount) * x / nsamples
    identity = numpy.identity(len(ave), ave.dtype)
    cov = pseudocount * identity + (1 - pseudocount) * x2 / nsamples
    cov -= B.outer(ave, ave)

    var = 1 - ave**2
    std = numpy.sqrt(var)
    corr = cov / B.outer(std, std)

    X = numpy.copy(identity)
    for i in range(iterations):
        F = numpy.diag(numpy.linalg.inv(identity + B.dot(X, corr))) - 1
        F *= numpy.diag(X)
        F = numpy.diag(1-F)
        B.mix_inplace(damping, X, F)
    D = numpy.diag( (1/numpy.diag(X) - 1)*var )
    J = -numpy.linalg.inv(cov + D)
    numpy.fill_diagonal(J, 0)

    model.params['weights'] = J
    model.params['visible_bias'] = model.layers['visible'].inverse_mean(ave)
    model.params['visible_bias'] -= B.dot(J, ave)
