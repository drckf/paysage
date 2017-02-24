from ... import backends as be

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
    model.params['weights'] = 0.001 * be.randn((nvis, nvis))
    be.fill_diagonal(model.params['weights'], 0)

    x = be.zeros(nvis)
    nsamples = 0
    while True:
        try:
            v_data = batch.get(mode='train')
        except StopIteration:
            break
        x += be.tsum(v_data, axis=0)
        nsamples += len(v_data)

    ave = x / nsamples
    model.params['visible_bias'] = model.layers['visible'].inverse_mean(ave)

def mean_field(batch, model,
               pseudocount=0.01):
    """
    Yasser Roudi, Erik Aurell, John A. Hertz.
    "Statistical physics of pairwise probability models"
    https://arxiv.org/pdf/0905.1410.pdf

    """
    nvis = len(model.params['visible_bias'])
    x = be.zeros(nvis)
    x2 = be.zeros((nvis,nvis))
    nsamples = 0
    while True:
        try:
            v_data = batch.get(mode='train')
        except StopIteration:
            break
        x += be.tsum(v_data, axis=0)
        x2 += be.dot(v_data.T, v_data)
        nsamples += len(v_data)

    ave = (1 - pseudocount) * x / nsamples
    identity = be.identity(len(ave))
    cov = pseudocount * identity + (1 - pseudocount) * x2 / nsamples
    cov -= be.outer(ave, ave)
    J = -be.inv(cov)
    be.fill_diagonal(J, 0)
    model.params['weights'] = J
    model.params['visible_bias'] = model.layers['visible'].inverse_mean(ave)
    model.params['visible_bias'] -= be.dot(J, ave)

def tap(batch, model,
        pseudocount=0.01,
        iterations=5):
    """
    Yasser Roudi, Erik Aurell, John A. Hertz.
    "Statistical physics of pairwise probability models"
    https://arxiv.org/pdf/0905.1410.pdf

    """
    nvis = len(model.params['visible_bias'])
    x = be.zeros(nvis)
    x2 = be.zeros((nvis,nvis))
    nsamples = 0
    while True:
        try:
            v_data = batch.get(mode='train')
        except StopIteration:
            break
        x += be.tsum(v_data, axis=0)
        x2 += be.dot(v_data.T, v_data)
        nsamples += len(v_data)

    # compute the necessary moments
    ave = (1 - pseudocount) * x / nsamples
    out = be.outer(ave, ave)
    identity = be.identity(len(ave))
    cov = pseudocount * identity + (1 - pseudocount) * x2 / nsamples
    cov -= out
    inv = be.inv(cov)

    # compute J_{ij} using Newton's method
    J = -inv
    for iteration in range(iterations):
        J -= (inv + J + 2 * out * J**2) / (1 + 4 * out * J)

    # remove the diagonal terms
    be.fill_diagonal(J, 0)
    model.params['weights'] = J

    # compute the fields
    model.params['visible_bias'] = model.layers['visible'].inverse_mean(ave)
    model.params['visible_bias'] -= be.dot(J, ave)
    model.params['visible_bias'] += be.dot(J**2, ave * (1-ave**2))
