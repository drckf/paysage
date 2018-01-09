from .. import backends as be
from .. import math_utils as mu
import math

def hinton(batch, model):
    """
    Initialize the parameters of an RBM.

    Based on the method described in:

    Hinton, Geoffrey.
    "A practical guide to training restricted Boltzmann machines."
    Momentum 9.1 (2010): 926.

    Initialize the weights from N(0, \sigma)
    Set hidden_bias = 0
    Set visible_bias = inverse_mean( \< v_i \> )
    If visible_scale: set visible_scale = \< v_i^2 \> - \< v_i \>^2

    Notes:
        Modifies the model parameters in place.

    Args:
        batch: A batch object that provides minibatches of data.
        model: A model to initialize.

    Returns:
        None

    """
    for i in range(len(model.weights)):
        model.weights[i].params.matrix[:] = \
                        0.01 * be.randn(model.weights[i].shape)
    while True:
        try:
            v_data = batch.get(mode='train')
        except StopIteration:
            break
        model.layers[0].online_param_update(v_data)
    model.layers[0].shrink_parameters(shrinkage=0.01)

def glorot_normal(batch, model):
    """
    Initialize the parameters of an RBM.

    Identical to the 'hinton' method above
    with the variation that we initialize the weights according to
    the prescription of Glorot and Bengio from

    "Understanding the difficulty of training deep feedforward neural networks", 2010:

    Initialize the weights from N(0, \sigma)
    with \sigma = \sqrt(2 / (num_vis_units + num_hidden_units)).

    Set hidden_bias = 0
    Set visible_bias = inverse_mean( \< v_i \> )
    If visible_scale: set visible_scale = \< v_i^2 \> - \< v_i \>^2

    Notes:
        Modifies the model parameters in place.

    Args:
        batch: A batch object that provides minibatches of data.
        model: A model to initialize.

    Returns:
        None

    """
    for i in range(len(model.weights)):
        sigma = math.sqrt(2/(model.weights[i].shape[0]
                             + model.weights[i].shape[1]))
        model.weights[i].params.matrix[:] = \
                        sigma * be.randn(model.weights[i].shape)
    while True:
        try:
            v_data = batch.get(mode='train')
        except StopIteration:
            break
        model.layers[0].online_param_update(v_data)
    model.layers[0].shrink_parameters(shrinkage=0.01)
