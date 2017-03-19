from ... import backends as be


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
    nvis, nhid = be.shape(model.params['weights'])
    model.params['weights'] = 0.01 * be.randn((nvis, nhid))
    model.params['hidden_bias'] = be.EPSILON * be.ones(nhid)
    has_scale = 'visible_scale' in model.params

    x = be.zeros(nvis)
    if has_scale:
        x2 = be.zeros(nvis)
    nbatches = 0
    while True:
        try:
            v_data = batch.get(mode='train')
        except StopIteration:
            break
        x += be.mean(v_data, axis=0)
        if has_scale:
            x2 += be.mean(v_data**2, axis=0)
        nbatches += 1

    model.params['visible_bias'] = model.layers['visible'].inverse_mean(x/nbatches)
    if has_scale:
        model.params['visible_scale'] = x2 / nbatches - (x / nbatches)**2
        # apply some shrinkage towards one
        be.mix_inplace(be.float_scalar(1 - 1/batch.batch_size),
                      model.params['visible_scale'],
                      be.ones_like(model.params['visible_scale'])
                      )
        # scale parameters should be expressed in log-space
        model.params['visible_scale'] = be.log(model.params['visible_scale'])
