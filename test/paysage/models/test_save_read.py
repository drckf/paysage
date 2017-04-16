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
    # create some extrinsics
    vis_layer.ext_params = layers.ExtrinsicParamsBernoulli(
        field=vis_layer.random((num_samples, num_vis))
    )
    hid_layer.ext_params = layers.ExtrinsicParamsGaussian(
        mean=hid_layer.random((num_samples, num_vis)),
        variance=hid_layer.random((num_samples, num_vis))
    )
    grbm = model.Model([vis_layer, hid_layer])
    with tempfile.NamedTemporaryFile() as file:
        store = pandas.HDFStore(file.name, mode='w')
        grbm.save(store)
        store.close()

def test_grbm_reload():
    vis_layer = layers.BernoulliLayer(num_vis)
    hid_layer = layers.GaussianLayer(num_hid)
    # create some extrinsics
    vis_layer.ext_params = layers.ExtrinsicParamsBernoulli(
        field=vis_layer.random((num_samples, num_vis))
    )
    hid_layer.ext_params = layers.ExtrinsicParamsGaussian(
        mean=hid_layer.random((num_samples, num_vis)),
        variance=hid_layer.random((num_samples, num_vis))
    )
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
    vis_data = vis_layer.sample_state()
    data_state = model.State.from_visible(vis_data, grbm)
    vis_orig = grbm.deterministic_step(data_state).units[0]
    vis_reload = grbm_reload.deterministic_step(data_state).units[0]
    assert be.allclose(vis_orig, vis_reload)



if __name__ == "__main__":
    pytest.main([__file__])
