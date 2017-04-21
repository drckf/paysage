import tempfile
import pandas

from paysage import layers
from paysage.models import model
from paysage import backends as be

import pytest

num_vis = 8
num_hid = 5
num_samples = 10

# ----- MODEL CONSTRUCTION ----- #

def test_rmb_construction():
    vis_layer = layers.BernoulliLayer(num_vis)
    hid_layer = layers.BernoulliLayer(num_hid)
    rbm = model.Model([vis_layer, hid_layer])

def test_grbm_construction():
    vis_layer = layers.GaussianLayer(num_vis)
    hid_layer = layers.BernoulliLayer(num_hid)
    rbm = model.Model([vis_layer, hid_layer])

def test_hopfield_construction():
    vis_layer = layers.BernoulliLayer(num_vis)
    hid_layer = layers.GaussianLayer(num_hid)
    rbm = model.Model([vis_layer, hid_layer])


# ----- CONFIG CREATION ----- #

def test_grbm_config():
    vis_layer = layers.BernoulliLayer(num_vis)
    hid_layer = layers.GaussianLayer(num_hid)
    grbm = model.Model([vis_layer, hid_layer])
    grbm.get_config()

def test_grbm_from_config():
    vis_layer = layers.BernoulliLayer(num_vis)
    hid_layer = layers.GaussianLayer(num_hid)
    grbm = model.Model([vis_layer, hid_layer])
    config = grbm.get_config()

    rbm_from_config = model.Model.from_config(config)
    config_from_config = rbm_from_config.get_config()
    assert config == config_from_config

def test_grbm_save():
    vis_layer = layers.BernoulliLayer(num_vis)
    hid_layer = layers.GaussianLayer(num_hid)
    grbm = model.Model([vis_layer, hid_layer])
    with tempfile.NamedTemporaryFile() as file:
        store = pandas.HDFStore(file.name, mode='w')
        grbm.save(store)
        store.close()

def test_grbm_reload():
    vis_layer = layers.BernoulliLayer(num_vis)
    hid_layer = layers.GaussianLayer(num_hid)
    # create some extrinsics
    grbm = model.Model([vis_layer, hid_layer])
    with tempfile.NamedTemporaryFile() as file:
        # save the model
        store = pandas.HDFStore(file.name, mode='w')
        grbm.save(store)
        store.close()
        # reload
        store = pandas.HDFStore(file.name, mode='r')
        grbm_reload = model.Model.from_saved(store)
        store.close()
    # check the two models are consistent
    vis_state = vis_layer.random((num_samples, num_vis))

    hid_orig = grbm.layers[1].conditional_mode(
        [vis_layer.rescale(vis_state)],
        [grbm.weights[0].W()])

    hid_reload = grbm_reload.layers[1].conditional_mode(
        [vis_layer.rescale(vis_state)],
        [grbm_reload.weights[0].W()])

    assert be.allclose(hid_orig, hid_reload)


if __name__ == "__main__":
    pytest.main([__file__])
