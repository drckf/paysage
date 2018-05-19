import tempfile
import pandas
import numpy as np

from paysage import layers
from paysage.models import BoltzmannMachine
from paysage.models.state import State
from paysage import backends as be
from paysage import batch
from paysage import math_utils

import pytest

num_vis = 8
num_hid = 5
num_samples = 10

def test_rmb_construction():
    vis_layer = layers.BernoulliLayer(num_vis)
    hid_layer = layers.BernoulliLayer(num_hid)
    rbm = BoltzmannMachine([vis_layer, hid_layer])

def test_grbm_construction():
    vis_layer = layers.GaussianLayer(num_vis)
    hid_layer = layers.BernoulliLayer(num_hid)
    rbm = BoltzmannMachine([vis_layer, hid_layer])

def test_hopfield_construction():
    vis_layer = layers.BernoulliLayer(num_vis)
    hid_layer = layers.GaussianLayer(num_hid)
    rbm = BoltzmannMachine([vis_layer, hid_layer])

def test_grbm_config():
    vis_layer = layers.BernoulliLayer(num_vis)
    hid_layer = layers.GaussianLayer(num_hid)
    grbm = BoltzmannMachine([vis_layer, hid_layer])
    grbm.get_config()

def test_grbm_from_config():
    vis_layer = layers.BernoulliLayer(num_vis)
    hid_layer = layers.GaussianLayer(num_hid)

    grbm = BoltzmannMachine([vis_layer, hid_layer])
    config = grbm.get_config()

    rbm_from_config = BoltzmannMachine.from_config(config)
    config_from_config = rbm_from_config.get_config()
    assert config == config_from_config

def test_grbm_save():
    vis_layer = layers.BernoulliLayer(num_vis, center=True)
    hid_layer = layers.GaussianLayer(num_hid, center=True)
    grbm = BoltzmannMachine([vis_layer, hid_layer])
    data = batch.Batch(
        {'train': batch.InMemoryTable(be.randn((10*num_samples, num_vis)), num_samples)})
    grbm.initialize(data)
    with tempfile.NamedTemporaryFile() as file:
        store = pandas.HDFStore(file.name, mode='w')
        grbm.save(store)
        store.close()

def test_grbm_reload():
    vis_layer = layers.BernoulliLayer(num_vis, center=True)
    hid_layer = layers.GaussianLayer(num_hid, center=True)

    # create some extrinsics
    grbm = BoltzmannMachine([vis_layer, hid_layer])
    data = batch.Batch(
        {'train': batch.InMemoryTable(be.randn((10*num_samples, num_vis)), num_samples)})
    grbm.initialize(data)
    with tempfile.NamedTemporaryFile() as file:
        # save the model
        store = pandas.HDFStore(file.name, mode='w')
        grbm.save(store)
        store.close()
        # reload
        store = pandas.HDFStore(file.name, mode='r')
        grbm_reload = BoltzmannMachine.from_saved(store)
        store.close()
    # check the two models are consistent
    vis_data = vis_layer.random((num_samples, num_vis))
    data_state = State.from_visible(vis_data, grbm)
    vis_orig = grbm.deterministic_iteration(1, data_state)[0]
    vis_reload = grbm_reload.deterministic_iteration(1, data_state)[0]
    assert be.allclose(vis_orig, vis_reload)
    assert be.allclose(grbm.layers[0].moments.mean, grbm_reload.layers[0].moments.mean)
    assert be.allclose(grbm.layers[0].moments.var, grbm_reload.layers[0].moments.var)
    assert be.allclose(grbm.layers[1].moments.mean, grbm_reload.layers[1].moments.mean)
    assert be.allclose(grbm.layers[1].moments.var, grbm_reload.layers[1].moments.var)


if __name__ == "__main__":
    pytest.main([__file__])
