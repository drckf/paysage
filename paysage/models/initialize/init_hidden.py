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
        p += B.mean(v_data, axis=0).astype(numpy.float32)
        nbatches += 1
        
    model.params['visible_bias'] = model.layers['visible'].inverse_mean(p/nbatches)
    
def hinton_grbm(batch, model):
    """
        Hinton says to initalize the weights from N(0, 0.01)
        hidden_bias = 0
        visible_loc = mean(v_i)
        visible_scale = variance(v_i)
        
        Hinton, Geoffrey. "A practical guide to training restricted Boltzmann machines." Momentum 9.1 (2010): 926.

    """
    nvis, nhid = model.params['weights'].shape
    model.params['weights'] = numpy.random.normal(loc=0.0, scale=0.01, size=(nvis, nhid)).astype(dtype=numpy.float32)
    model.params['hidden_bias'] = B.EPSILON * numpy.ones(nhid, dtype=numpy.float32)
    
    x = numpy.zeros(batch.ncols, dtype=numpy.float32)
    x2 = numpy.zeros(batch.ncols, dtype=numpy.float32)
    nbatches = 0
    while True:
        try:
            v_data = batch.get(mode='train')
        except StopIteration:
            break
        x += B.mean(v_data, axis=0).astype(numpy.float32)
        x2 += B.mean(v_data**2, axis=0).astype(numpy.float32)
        nbatches += 1
        
    model.params['visible_loc'] = model.layers['visible'].inverse_mean(x/nbatches)
    model.params['visible_scale'] = x2 / nbatches - (x / nbatches)**2 + 1 / batch.batch_size
