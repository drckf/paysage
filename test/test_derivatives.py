import os

from paysage import backends as be
from paysage import layers
from paysage.models import hidden

#import pytest

def test_bernoulli_update():
    num_visible_units = 100
    num_hidden_units = 50
    batch_size = 25

    # set a seed for the random number generator
    be.set_seed()

    # set up some layer and model objects
    vis_layer = layers.BernoulliLayer(num_visible_units)
    hid_layer = layers.BernoulliLayer(num_hidden_units)
    rbm = hidden.Model([vis_layer, hid_layer])

    # randomly set the intrinsic model parameters
    a = be.randn((num_visible_units,))
    b = be.randn((num_hidden_units,))
    W = be.randn((num_visible_units, num_hidden_units))

    rbm.layers[0].int_params['loc'] = a
    rbm.layers[1].int_params['loc'] = b
    rbm.weights[0].int_params['matrix'] = W

    # generate a random batch of data
    vdata = rbm.layers[0].random((batch_size, num_visible_units))
    hdata = rbm.layers[1].random((batch_size, num_hidden_units))

    # compute extrinsic parameters
    hidden_field = be.dot(vdata, W) # (batch_size, num_hidden_units)
    hidden_field += be.broadcast(b, hidden_field)

    visible_field = be.dot(hdata, be.transpose(W)) # (batch_size, num_visible_units)
    visible_field += be.broadcast(a, visible_field)

    # update the extrinsic parameter using the layer functions
    rbm.layers[1].update(vdata, rbm.weights[0].W())
    rbm.layers[0].update(hdata, be.transpose(rbm.weights[0].W()))

    assert be.allclose(hidden_field, rbm.layers[1].ext_params['field']), \
    "hidden field wrong in bernoulli-bernoulli rbm"

    assert be.allclose(visible_field, rbm.layers[0].ext_params['field']), \
    "hidden field wrong in bernoulli-bernoulli rbm"

if __name__ == "__main__":
    test_bernoulli_update()
