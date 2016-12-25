import numpy
from ... import backends as B


# ----- FUNCTIONS ----- #


def hinton(batch, model):
    """
        Hinton says to initalize the weights from N(0, 0.01)
        hidden_bias = 0
        visible_bias = inverse_mean( \< v_i \> )
        if visible_scale:
            visible_scale = \< v_i^2 \> - \< v_i \>^2
        
        Hinton, Geoffrey. "A practical guide to training restricted Boltzmann machines." Momentum 9.1 (2010): 926.

    """
    nvis, nhid = model.params['weights'].shape
    model.params['weights'] = numpy.random.normal(loc=0.0, scale=0.01, size=(nvis, nhid)).astype(dtype=numpy.float32)
    model.params['hidden_bias'] = B.EPSILON * numpy.ones(nhid, dtype=numpy.float32)
    
    has_scale = 'visible_scale' in model.params
    
    x = numpy.zeros(batch.ncols, dtype=numpy.float32)
    if has_scale:
        x2 = numpy.zeros(batch.ncols, dtype=numpy.float32)
    nbatches = 0
    while True:
        try:
            v_data = batch.get(mode='train')
        except StopIteration:
            break
        x += B.mean(v_data, axis=0).astype(numpy.float32)
        if has_scale:
            x2 += B.mean(v_data**2, axis=0).astype(numpy.float32)
        nbatches += 1
        
    model.params['visible_bias'] = model.layers['visible'].inverse_mean(x/nbatches)
    if has_scale:
        model.params['visible_scale'] = x2 / nbatches - (x / nbatches)**2
        # apply some shrinkage towards one
        B.mix_inplace(numpy.float32(1 - 1/batch.batch_size), model.params['visible_scale'], numpy.ones_like(model.params['visible_scale'], dtype=numpy.float32))
        # scale parameters should be expressed in log-space    
        model.params['visible_scale'] = B.log(model.params['visible_scale'])
        
        model.params['visible_scale'] = numpy.zeros_like(model.params['visible_scale'])
    