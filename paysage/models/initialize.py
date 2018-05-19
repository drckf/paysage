from .. import backends as be
from .. import math_utils as mu
from .. import factorization
from .. import layers
import math

def hinton(batch, model, **kwargs):
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
    for i in range(len(model.connections)):
        model.connections[i].weights.set_params(layers.ParamsWeights(
                        0.01 * be.randn(model.connections[i].shape)))
    while True:
        try:
            v_data = batch.get(mode='train')
        except StopIteration:
            break
        model.layers[0].online_param_update(v_data)
    model.layers[0].shrink_parameters(shrinkage=0.01)

def glorot_normal(batch, model, **kwargs):
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
    for i in range(len(model.connections)):
        sigma = math.sqrt(2/(model.connections[i].shape[0]
                             + model.connections[i].shape[1]))
        model.connections[i].weights.set_params(layers.ParamsWeights(
                        sigma * be.randn(model.connections[i].shape)))
    while True:
        try:
            v_data = batch.get(mode='train')
        except StopIteration:
            break
        model.layers[0].online_param_update(v_data)
    model.layers[0].shrink_parameters(shrinkage=0.01)

def stddev(batch, model, **kwargs):
    """
    Initialize the parameters of an RBM. Set the rows of the weight matrix
    proportional to the standard deviations of the visible units.

    Notes:
        Modifies the model parameters in place.

    Args:
        batch: A batch object that provides minibatches of data.
        model: A model to initialize.

    Returns:
        None

    """
    moments = mu.MeanVarianceArrayCalculator()
    while True:
        try:
            v_data = batch.get(mode='train')
        except StopIteration:
            break
        moments.update(v_data)
        model.layers[0].online_param_update(v_data)
    model.layers[0].shrink_parameters(shrinkage=0.01)

    std = be.unsqueeze(be.sqrt(moments.var), axis=1)
    for i in range(len(model.connections)):
        glorot_multiplier = math.sqrt(2/(model.connections[i].shape[0]
        + model.connections[i].shape[1]))
        if i == 0:
            model.connections[i].weights.set_params(layers.ParamsWeights(
            glorot_multiplier * be.multiply(std, be.randn(model.connections[i].shape))))
        else:
            model.connections[i].weights.set_params(layers.ParamsWeights(
            glorot_multiplier * be.randn(model.connections[i].shape)))

def pca(batch, model, **kwargs):
    """
    Initialize the parameters of an RBM using the principal components
    to initialize the weights.

    Notes:
        Modifies the model parameters in place.

    Args:
        batch: A batch object that provides minibatches of data.
        model: A model to initialize.

    Returns:
        None

    """
    default_kwargs = {
            "epochs": 100,
            "grad_steps_per_minibatch": 1,
            "stepsize": 0.001,
            "convergence": 1e-5
            }

    # overwrite any kwargs passed in to the function
    for arg in kwargs:
        default_kwargs[arg] = kwargs[arg]

    # initialize the layer parameters as usual
    while True:
        try:
            v_data = batch.get(mode='train')
        except StopIteration:
            break
        model.layers[0].online_param_update(v_data)
    model.layers[0].shrink_parameters(shrinkage=0.01)

    # compute a pca
    num_visible_units, num_hidden_units = model.connections[0].shape
    assert num_visible_units >= num_hidden_units, "PCA initialization doesn't suppport num_units < num_components"

    pca = factorization.PCA.from_batch(batch, num_hidden_units, **default_kwargs)

    std = be.sqrt(be.EPSILON + pca.var)
    weights = std / be.norm(std)

    # intialize the model weights
    for i in range(len(model.connections)):
        n = model.connections[i].shape[0]
        m = model.connections[i].shape[1]
        glorot_multiplier = math.sqrt(2/(n+m))
        if i == 0:
            model.connections[i].weights.set_params(layers.ParamsWeights(
            glorot_multiplier * math.sqrt(n) * weights * pca.W))
        else:
            model.connections[i].weights.params.matrix[:] = \
            glorot_multiplier * be.randn(model.connections[i].shape)
