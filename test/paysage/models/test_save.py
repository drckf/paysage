import tempfile
import pandas

from paysage import layers
from paysage.models import hidden

import pytest

# ----- MODEL CONSTRUCTION ----- #

def test_rmb_construction():
    vis_layer = layers.BernoulliLayer(8)
    hid_layer = layers.BernoulliLayer(5)
    rbm = hidden.Model([vis_layer, hid_layer])

def test_grbm_construction():
    vis_layer = layers.GaussianLayer(8)
    hid_layer = layers.BernoulliLayer(5)
    rbm = hidden.Model([vis_layer, hid_layer])

def test_hopfield_construction():
    vis_layer = layers.BernoulliLayer(8)
    hid_layer = layers.GaussianLayer(5)
    rbm = hidden.Model([vis_layer, hid_layer])


# ----- CONFIG CREATION ----- #

def test_grbm_config():
    vis_layer = layers.BernoulliLayer(8)
    hid_layer = layers.GaussianLayer(5)
    grbm = hidden.Model([vis_layer, hid_layer])
    grbm.get_config()

def test_grbm_from_config():
    vis_layer = layers.BernoulliLayer(8)
    hid_layer = layers.GaussianLayer(5)
    grbm = hidden.Model([vis_layer, hid_layer])
    config = grbm.get_config()

    rbm_from_config = hidden.Model.from_config(config)
    config_from_config = rbm_from_config.get_config()
    assert config == config_from_config

def test_grbm_save():
    vis_layer = layers.BernoulliLayer(8)
    hid_layer = layers.GaussianLayer(5)
    grbm = hidden.Model([vis_layer, hid_layer])
    with tempfile.NamedTemporaryFile() as file:
        store = pandas.HDFStore(file.name, mode='w')
        grbm.save(store)
        store.close()


if __name__ == "__main__":
    pytest.main([__file__])
