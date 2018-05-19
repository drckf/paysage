import sys

def layer_from_config(config):
    """
    Construct a layer from a configuration.

    Args:
        A dictionary configuration of the layer metadata.

    Returns:
        An object which is a subclass of `Layer`.

    """
    layer_config = dict(config)
    layer_obj = getattr(sys.modules['paysage.layers'], layer_config.pop("layer_type"))
    return layer_obj.from_config(layer_config)