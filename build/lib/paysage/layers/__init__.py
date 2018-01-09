import sys
from .layer import *
from .bernoulli_layer import *
from .gaussian_layer import *
from .onehot_layer import *
from .weights import *

# ---- FUNCTIONS ----- #

def layer_from_config(config):
    """
    Construct a layer from a configuration.

    Args:
        A dictionary configuration of the layer metadata.

    Returns:
        An object which is a subclass of `Layer`.

    """
    layer_obj = getattr(sys.modules[__name__], config["layer_type"])
    return layer_obj.from_config(config)

def get(key):
    if 'gauss' in key.lower():
        return GaussianLayer
    elif 'bern' in key.lower():
        return BernoulliLayer
    elif 'onehot' in key.lower():
        return OneHotLayer
    else:
        raise ValueError('Unknown layer type')
