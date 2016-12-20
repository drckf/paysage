import numpy
from ... import backends as B


# ----- FUNCTIONS ----- #


def hinton(batch, model):
    """
        Hinton says to initalize the weights from N(0, 0.01)
        hidden_bias = 0
        visible_bias = log(p_i / (1 - p_i))
        
        Hinton, Geoffrey. "A practical guide to training restricted Boltzmann machines." Momentum 9.1 (2010): 926.

    """
    nvis, nhid = model.params['weights'].shape
    model.params['weights'] = numpy.random.normal(loc=0.0, scale=0.01, size=(nvis, nhid)).astype(dtype=numpy.float32)
    model.params['hidden_bias'] = B.EPSILON * numpy.ones(nhid, dtype=numpy.float32)
    
    p = numpy.zeros(batch.ncols, dtype=numpy.float32)
    nbatches = 0
    while True:
        try:
            v_data = batch.get(mode='train')
        except StopIteration:
            break
        p += numpy.mean(v_data, axis=0).astype(numpy.float32)
        nbatches += 1
        
    model.params['visible_bias'] = model.layers['visible'].inverse_mean(p/nbatches)
