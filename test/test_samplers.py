from paysage import backends as be
from paysage import layers
from paysage.models import BoltzmannMachine
from paysage import math_utils as mu
from paysage.models.state import State
from paysage import samplers

import pytest

def test_independent():
    """
    Test sampling from an rbm with two layers connected by a weight matrix that
    contains all zeros, so that the layers are independent.

    Note:
        This test compares values estimated by *sampling* to values computed
        analytically. It can fail for small batch_size, or strict tolerances,
        even if everything is working propery.

    """
    num_visible_units = 20
    num_hidden_units = 10
    batch_size = 1000
    steps = 100
    mean_tol = 0.2
    corr_tol = 0.2

    # set a seed for the random number generator
    be.set_seed()

    layer_types = [
            layers.BernoulliLayer,
            layers.GaussianLayer]

    for layer_type in layer_types:
        # set up some layer and model objects
        vis_layer = layer_type(num_visible_units)
        hid_layer = layer_type(num_hidden_units)
        rbm = BoltzmannMachine([vis_layer, hid_layer])

        # randomly set the intrinsic model parameters
        a = be.rand((num_visible_units,))
        b = be.rand((num_hidden_units,))
        W = be.zeros((num_visible_units, num_hidden_units))

        rbm.layers[0].params.loc[:] = a
        rbm.layers[1].params.loc[:] = b
        rbm.connections[0].weights.params.matrix[:] = W

        if layer_type == layers.GaussianLayer:
            log_var_a = be.randn((num_visible_units,))
            log_var_b = be.randn((num_hidden_units,))
            rbm.layers[0].params.log_var[:] = log_var_a
            rbm.layers[1].params.log_var[:] = log_var_b

        # initialize a state
        state = State.from_model(batch_size, rbm)

        # run a markov chain to update the state
        state = rbm.markov_chain(steps, state)

        # compute the mean
        state_for_moments = State.from_model(1, rbm)
        sample_mean = [be.mean(state[i], axis=0) for i in range(state.len)]
        model_mean = [rbm.layers[i].conditional_mean(
                rbm._connected_rescaled_units(i, state_for_moments),
                rbm._connected_weights(i)) for i in range(rbm.num_layers)]

        # check that the means are roughly equal
        for i in range(rbm.num_layers):
            ave = sample_mean[i]
            close = be.allclose(ave, model_mean[i][0], rtol=mean_tol, atol=mean_tol)
            assert close, "{0} {1}: sample mean does not match model mean".format(layer_type, i)

        # check the cross correlation between the layers
        crosscov = be.cov(state[0], state[1])
        norm = be.outer(be.std(state[0], axis=0), be.std(state[1], axis=0))
        crosscorr = be.divide(norm, crosscov)
        assert be.tmax(be.tabs(crosscorr)) < corr_tol, "{} cross correlation too large".format(layer_type)


def test_conditional_sampling():
    """
    Test sampling from one layer conditioned on the state of another layer.

    Note:
        This test compares values estimated by *sampling* to values computed
        analytically. It can fail for small batch_size, or strict tolerances,
        even if everything is working propery.

    """
    num_visible_units = 20
    num_hidden_units = 10
    steps = 1000
    mean_tol = 0.1

    # set a seed for the random number generator
    be.set_seed()

    layer_types = [
            layers.BernoulliLayer,
            layers.GaussianLayer]

    for layer_type in layer_types:
        # set up some layer and model objects
        vis_layer = layer_type(num_visible_units)
        hid_layer = layer_type(num_hidden_units)
        rbm = BoltzmannMachine([vis_layer, hid_layer])

        # randomly set the intrinsic model parameters
        a = be.rand((num_visible_units,))
        b = be.rand((num_hidden_units,))
        W = 10 * be.rand((num_visible_units, num_hidden_units))

        rbm.layers[0].params.loc[:] = a
        rbm.layers[1].params.loc[:] = b
        rbm.connections[0].weights.params.matrix[:] = W

        if layer_type == layers.GaussianLayer:
            log_var_a = be.randn((num_visible_units,))
            log_var_b = be.randn((num_hidden_units,))
            rbm.layers[0].params.log_var[:] = log_var_a
            rbm.layers[1].params.log_var[:] = log_var_b

        # initialize a state
        state = State.from_model(1, rbm)

        # set up a calculator for the moments
        moments = mu.MeanVarianceArrayCalculator()

        for _ in range(steps):
            moments.update(rbm.layers[0].conditional_sample(
                    rbm._connected_rescaled_units(0, state),
                    rbm._connected_weights(0)))

        model_mean = rbm.layers[0].conditional_mean(
                rbm._connected_rescaled_units(0, state),
                rbm._connected_weights(0))


        ave = moments.mean

        close = be.allclose(ave, model_mean[0], rtol=mean_tol, atol=mean_tol)
        assert close, "{} conditional mean".format(layer_type)

        if layer_type == layers.GaussianLayer:
            model_mean, model_var = rbm.layers[0].conditional_params(
                rbm._connected_rescaled_units(0, state),
                rbm._connected_weights(0))

            close = be.allclose(be.sqrt(moments.var), be.sqrt(model_var[0]), rtol=mean_tol, atol=mean_tol)
            assert close, "{} conditional standard deviation".format(layer_type)


# ----- TEST SAMPLER CLASSES ----- #

def test_clamped_SequentialMC():
    num_visible_units = 100
    num_hidden_units = 50
    batch_size = 25
    steps = 1

    # set a seed for the random number generator
    be.set_seed()

    # set up some layer and model objects
    vis_layer = layers.BernoulliLayer(num_visible_units)
    hid_layer = layers.BernoulliLayer(num_hidden_units)
    rbm = BoltzmannMachine([vis_layer, hid_layer])

    # randomly set the intrinsic model parameters
    a = be.randn((num_visible_units,))
    b = be.randn((num_hidden_units,))
    W = be.randn((num_visible_units, num_hidden_units))

    rbm.layers[0].params.loc[:] = a
    rbm.layers[1].params.loc[:] = b
    rbm.connections[0].weights.params.matrix[:] = W

    # generate a random batch of data
    vdata = rbm.layers[0].random((batch_size, num_visible_units))
    data_state = State.from_visible(vdata, rbm)

    for u in ['markov_chain', 'mean_field_iteration', 'deterministic_iteration']:

        # set up the sampler with the visible layer clamped
        sampler = samplers.SequentialMC(rbm, updater=u, clamped=[0], beta_std=0)
        sampler.set_state(data_state)

        # update the sampler state and check the output
        sampler.update_state(steps)

        assert be.allclose(data_state[0], sampler.state[0]), \
        "visible layer is clamped, and shouldn't get updated: {}".format(u)

        assert not be.allclose(data_state[1], sampler.state[1]), \
        "hidden layer is not clamped, and should get updated: {}".format(u)


def test_unclamped_SequentialMC():
    num_visible_units = 100
    num_hidden_units = 50
    batch_size = 25
    steps = 1

    # set a seed for the random number generator
    be.set_seed()

    # set up some layer and model objects
    vis_layer = layers.BernoulliLayer(num_visible_units)
    hid_layer = layers.BernoulliLayer(num_hidden_units)
    rbm = BoltzmannMachine([vis_layer, hid_layer])

    # randomly set the intrinsic model parameters
    a = be.randn((num_visible_units,))
    b = be.randn((num_hidden_units,))
    W = be.randn((num_visible_units, num_hidden_units))

    rbm.layers[0].params.loc[:] = a
    rbm.layers[1].params.loc[:] = b
    rbm.connections[0].weights.params.matrix[:] = W

    # generate a random batch of data
    vdata = rbm.layers[0].random((batch_size, num_visible_units))
    data_state = State.from_visible(vdata, rbm)

    for u in ['markov_chain', 'mean_field_iteration', 'deterministic_iteration']:
        # set up the sampler with the visible layer clamped
        sampler = samplers.SequentialMC(rbm, updater=u, beta_std=0)
        sampler.set_state(data_state)

        # update the sampler state and check the output
        sampler.update_state(steps)

        assert not be.allclose(data_state[0], sampler.state[0]), \
        "visible layer is not clamped, and should get updated: {}".format(u)

        assert not be.allclose(data_state[1], sampler.state[1]), \
        "hidden layer is not clamped, and should get updated: {}".format(u)

def test_state_for_grad_SequentialMC():
    num_visible_units = 100
    num_hidden_units = 50
    batch_size = 25

    # set a seed for the random number generator
    be.set_seed()

    # set up some layer and model objects
    vis_layer = layers.BernoulliLayer(num_visible_units)
    hid_layer = layers.BernoulliLayer(num_hidden_units)
    rbm = BoltzmannMachine([vis_layer, hid_layer])

    # randomly set the intrinsic model parameters
    a = be.randn((num_visible_units,))
    b = be.randn((num_hidden_units,))
    W = be.randn((num_visible_units, num_hidden_units))

    rbm.layers[0].params.loc[:] = a
    rbm.layers[1].params.loc[:] = b
    rbm.connections[0].weights.params.matrix[:] = W

    # generate a random batch of data
    vdata = rbm.layers[0].random((batch_size, num_visible_units))
    data_state = State.from_visible(vdata, rbm)

    for u in ['markov_chain', 'mean_field_iteration', 'deterministic_iteration']:
        # set up the sampler
        sampler = samplers.SequentialMC(rbm, updater=u, clamped=[0], beta_std=0)
        sampler.set_state(data_state)

        # update the state of the hidden layer
        grad_state = sampler.state_for_grad(1)

        assert be.allclose(data_state[0], grad_state[0]), \
        "visible layer is clamped, and shouldn't get updated: {}".format(u)

        assert not be.allclose(data_state[1], grad_state[1]), \
        "hidden layer is not clamped, and should get updated: {}".format(u)

        # compute the conditional mean with the layer function
        ave = rbm.layers[1].conditional_mean(
                rbm._connected_rescaled_units(1, data_state),
                rbm._connected_weights(1))

        assert be.allclose(ave, grad_state[1]), \
        "hidden layer of grad_state should be conditional mean: {}".format(u)

def test_clamped_DrivenSequentialMC():
    num_visible_units = 100
    num_hidden_units = 50
    batch_size = 25
    steps = 1

    # set a seed for the random number generator
    be.set_seed()

    # set up some layer and model objects
    vis_layer = layers.BernoulliLayer(num_visible_units)
    hid_layer = layers.BernoulliLayer(num_hidden_units)
    rbm = BoltzmannMachine([vis_layer, hid_layer])

    # randomly set the intrinsic model parameters
    a = be.randn((num_visible_units,))
    b = be.randn((num_hidden_units,))
    W = be.randn((num_visible_units, num_hidden_units))

    rbm.layers[0].params.loc[:] = a
    rbm.layers[1].params.loc[:] = b
    rbm.connections[0].weights.params.matrix[:] = W

    # generate a random batch of data
    vdata = rbm.layers[0].random((batch_size, num_visible_units))
    data_state = State.from_visible(vdata, rbm)

    for u in ['markov_chain', 'mean_field_iteration', 'deterministic_iteration']:

        # set up the sampler with the visible layer clamped
        sampler = samplers.SequentialMC(rbm, updater=u, clamped=[0])
        sampler.set_state(data_state)

        # update the sampler state and check the output
        sampler.update_state(steps)

        assert be.allclose(data_state[0], sampler.state[0]), \
        "visible layer is clamped, and shouldn't get updated: {}".format(u)

        assert not be.allclose(data_state[1], sampler.state[1]), \
        "hidden layer is not clamped, and should get updated: {}".format(u)

def test_unclamped_DrivenSequentialMC():
    num_visible_units = 100
    num_hidden_units = 50
    batch_size = 25
    steps = 1

    # set a seed for the random number generator
    be.set_seed()

    # set up some layer and model objects
    vis_layer = layers.BernoulliLayer(num_visible_units)
    hid_layer = layers.BernoulliLayer(num_hidden_units)
    rbm = BoltzmannMachine([vis_layer, hid_layer])

    # randomly set the intrinsic model parameters
    a = be.randn((num_visible_units,))
    b = be.randn((num_hidden_units,))
    W = be.randn((num_visible_units, num_hidden_units))

    rbm.layers[0].params.loc[:] = a
    rbm.layers[1].params.loc[:] = b
    rbm.connections[0].weights.params.matrix[:] = W

    # generate a random batch of data
    vdata = rbm.layers[0].random((batch_size, num_visible_units))
    data_state = State.from_visible(vdata, rbm)

    for u in ['markov_chain', 'mean_field_iteration', 'deterministic_iteration']:
        # set up the sampler with the visible layer clamped
        sampler = samplers.SequentialMC(rbm, updater=u)
        sampler.set_state(data_state)

        # update the sampler state and check the output
        sampler.update_state(steps)

        assert not be.allclose(data_state[0], sampler.state[0]), \
        "visible layer is not clamped, and should get updated: {}".format(u)

        assert not be.allclose(data_state[1], sampler.state[1]), \
        "hidden layer is not clamped, and should get updated: {}".format(u)

def test_state_for_grad_DrivenSequentialMC():
    num_visible_units = 100
    num_hidden_units = 50
    batch_size = 25

    # set a seed for the random number generator
    be.set_seed()

    # set up some layer and model objects
    vis_layer = layers.BernoulliLayer(num_visible_units)
    hid_layer = layers.BernoulliLayer(num_hidden_units)
    rbm = BoltzmannMachine([vis_layer, hid_layer])

    # randomly set the intrinsic model parameters
    a = be.randn((num_visible_units,))
    b = be.randn((num_hidden_units,))
    W = be.randn((num_visible_units, num_hidden_units))

    rbm.layers[0].params.loc[:] = a
    rbm.layers[1].params.loc[:] = b
    rbm.connections[0].weights.params.matrix[:] = W

    # generate a random batch of data
    vdata = rbm.layers[0].random((batch_size, num_visible_units))
    data_state = State.from_visible(vdata, rbm)

    for u in ['markov_chain', 'mean_field_iteration', 'deterministic_iteration']:
        # set up the sampler
        sampler = samplers.SequentialMC(rbm, updater=u, clamped=[0])
        sampler.set_state(data_state)

        # update the state of the hidden layer
        grad_state = sampler.state_for_grad(1)

        assert be.allclose(data_state[0], grad_state[0]), \
        "visible layer is clamped, and shouldn't get updated: {}".format(u)

        assert not be.allclose(data_state[1], grad_state[1]), \
        "hidden layer is not clamped, and should get updated: {}".format(u)

        # compute the conditional mean with the layer function
        ave = rbm.layers[1].conditional_mean(
                rbm._connected_rescaled_units(1, data_state),
                rbm._connected_weights(1))

        assert be.allclose(ave, grad_state[1]), \
        "hidden layer of grad_state should be conditional mean: {}".format(u)


if __name__ == "__main__":
    pytest.main([__file__])
