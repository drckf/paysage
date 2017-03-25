from ... import backends as be


# ----- FUNCTIONS ----- #


def hinton(batch, model):
    """
    Initialize the parameters of an RBM.

    Based on the method described in:

    Hinton, Geoffrey.
    "A practical guide to training restricted Boltzmann machines."
    Momentum 9.1 (2010): 926.

    Initalize the weights from N(0, 0.01)
    Set hidden_bias = 0
    Set visible_bias = inverse_mean( \< v_i \> )
    If visible_scale: set visible_scale = \< v_i^2 \> - \< v_i \>^2

    Notes:
        Modifies the model parameters in place.

    Args:
        batch: A batch object that provides minibatches of data.
        model: A model to inialize.

    Returns:
        None

    """
    i = 0
    model.weights[i].val = 0.01 * be.randn(model.weights[i].shape)
    while True:
        try:
            v_data = batch.get(mode='train')
        except StopIteration:
            break
        model.layers[i].online_param_update(v_data)
    model.layers[i].shrink_parameters(shrinkage=0.01)
