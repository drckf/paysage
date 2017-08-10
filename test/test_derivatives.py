from paysage import backends as be
from paysage import layers
from paysage.models import model
from paysage.models import model_utils as mu
from paysage.models import gradient_util as gu
import pytest
from copy import deepcopy
from cytoolz import partial

# ----- Functional Programs with Gradients ----- #

def test_zero_grad():
    num_visible_units = 100
    num_hidden_units = 50

    # set a seed for the random number generator
    be.set_seed()

    # set up some layer and model objects
    vis_layer = layers.BernoulliLayer(num_visible_units)
    hid_layer = layers.BernoulliLayer(num_hidden_units)
    rbm = model.Model([vis_layer, hid_layer])

    # create a gradient object filled with zeros
    gu.zero_grad(rbm)

def test_random_grad():
    num_visible_units = 100
    num_hidden_units = 50

    # set a seed for the random number generator
    be.set_seed()

    # set up some layer and model objects
    vis_layer = layers.BernoulliLayer(num_visible_units)
    hid_layer = layers.BernoulliLayer(num_hidden_units)
    rbm = model.Model([vis_layer, hid_layer])

    # create a gradient object filled with random numbers
    gu.random_grad(rbm)

def test_grad_accumulate():
    num_visible_units = 100
    num_hidden_units = 50

    # set a seed for the random number generator
    be.set_seed()

    # set up some layer and model objects
    vis_layer = layers.BernoulliLayer(num_visible_units)
    hid_layer = layers.BernoulliLayer(num_hidden_units)
    rbm = model.Model([vis_layer, hid_layer])

    # create a gradient object filled with random numbers
    grad = gu.random_grad(rbm)
    gu.grad_accumulate(be.norm, grad)

def test_grad_apply():
    num_visible_units = 100
    num_hidden_units = 50

    # set a seed for the random number generator
    be.set_seed()

    # set up some layer and model objects
    vis_layer = layers.BernoulliLayer(num_visible_units)
    hid_layer = layers.BernoulliLayer(num_hidden_units)
    rbm = model.Model([vis_layer, hid_layer])

    # create a gradient object filled with random numbers
    grad = gu.random_grad(rbm)
    gu.grad_apply(be.square, grad)

def test_grad_apply_():
    num_visible_units = 100
    num_hidden_units = 50

    # set a seed for the random number generator
    be.set_seed()

    # set up some layer and model objects
    vis_layer = layers.BernoulliLayer(num_visible_units)
    hid_layer = layers.BernoulliLayer(num_hidden_units)
    rbm = model.Model([vis_layer, hid_layer])

    # create a gradient object filled with random numbers
    grad = gu.random_grad(rbm)
    gu.grad_apply_(be.square, grad)

def test_grad_mapzip():
    num_visible_units = 100
    num_hidden_units = 50

    # set a seed for the random number generator
    be.set_seed()

    # set up some layer and model objects
    vis_layer = layers.BernoulliLayer(num_visible_units)
    hid_layer = layers.BernoulliLayer(num_hidden_units)
    rbm = model.Model([vis_layer, hid_layer])

    # create a gradient object filled with random numbers
    grad_1 = gu.random_grad(rbm)
    grad_2 = gu.random_grad(rbm)
    gu.grad_mapzip(be.add, grad_1, grad_2)

def test_grad_mapzip_():
    num_visible_units = 100
    num_hidden_units = 50

    # set a seed for the random number generator
    be.set_seed()

    # set up some layer and model objects
    vis_layer = layers.BernoulliLayer(num_visible_units)
    hid_layer = layers.BernoulliLayer(num_hidden_units)
    rbm = model.Model([vis_layer, hid_layer])

    # create a gradient object filled with random numbers
    grad_1 = gu.random_grad(rbm)
    grad_2 = gu.random_grad(rbm)
    gu.grad_mapzip_(be.add_, grad_1, grad_2)

def test_grad_magnitude():
    num_visible_units = 100
    num_hidden_units = 50

    # set a seed for the random number generator
    be.set_seed()

    # set up some layer and model objects
    vis_layer = layers.BernoulliLayer(num_visible_units)
    hid_layer = layers.BernoulliLayer(num_hidden_units)
    rbm = model.Model([vis_layer, hid_layer])

    # create a gradient object filled with random numbers
    grad = gu.zero_grad(rbm)
    mag = gu.grad_magnitude(grad)
    assert mag == 0


# ----- Layer Methods ----- #

def test_bernoulli_conditional_params():
    num_visible_units = 100
    num_hidden_units = 50
    batch_size = 25

    # set a seed for the random number generator
    be.set_seed()

    # set up some layer and model objects
    vis_layer = layers.BernoulliLayer(num_visible_units)
    hid_layer = layers.BernoulliLayer(num_hidden_units)
    rbm = model.Model([vis_layer, hid_layer])

    # randomly set the intrinsic model parameters
    a = be.randn((num_visible_units,))
    b = be.randn((num_hidden_units,))
    W = be.randn((num_visible_units, num_hidden_units))

    rbm.layers[0].params.loc[:] = a
    rbm.layers[1].params.loc[:] = b
    rbm.weights[0].params.matrix[:] = W

    # generate a random batch of data
    vdata = rbm.layers[0].random((batch_size, num_visible_units))
    hdata = rbm.layers[1].random((batch_size, num_hidden_units))

    # compute conditional parameters
    hidden_field = be.dot(vdata, W) # (batch_size, num_hidden_units)
    hidden_field += b

    visible_field = be.dot(hdata, be.transpose(W)) # (batch_size, num_visible_units)
    visible_field += a

    # compute conditional parameters with layer funcitons
    hidden_field_layer = rbm.layers[1]._conditional_params(
        [vdata], [rbm.weights[0].W()])
    visible_field_layer = rbm.layers[0]._conditional_params(
        [hdata], [rbm.weights[0].W_T()])

    assert be.allclose(hidden_field, hidden_field_layer), \
    "hidden field wrong in bernoulli-bernoulli rbm"

    assert be.allclose(visible_field, visible_field_layer), \
    "visible field wrong in bernoulli-bernoulli rbm"


def test_bernoulli_derivatives():
    num_visible_units = 100
    num_hidden_units = 50
    batch_size = 25

    # set a seed for the random number generator
    be.set_seed()

    # set up some layer and model objects
    vis_layer = layers.BernoulliLayer(num_visible_units)
    hid_layer = layers.BernoulliLayer(num_hidden_units)
    rbm = model.Model([vis_layer, hid_layer])

    # randomly set the intrinsic model parameters
    a = be.randn((num_visible_units,))
    b = be.randn((num_hidden_units,))
    W = be.randn((num_visible_units, num_hidden_units))

    rbm.layers[0].params.loc[:] = a
    rbm.layers[1].params.loc[:] = b
    rbm.weights[0].params.matrix[:] = W

    # generate a random batch of data
    vdata = rbm.layers[0].random((batch_size, num_visible_units))
    vdata_scaled = rbm.layers[0].rescale(vdata)

    # compute the conditional mean of the hidden layer
    hid_mean = rbm.layers[1].conditional_mean([vdata], [rbm.weights[0].W()])
    hid_mean_scaled = rbm.layers[1].rescale(hid_mean)

    # compute the derivatives
    d_visible_loc = -be.mean(vdata, axis=0)
    d_hidden_loc = -be.mean(hid_mean_scaled, axis=0)
    d_W = -be.batch_outer(vdata, hid_mean_scaled) / len(vdata)

    # compute the derivatives using the layer functions
    vis_derivs = rbm.layers[0].derivatives(vdata, [hid_mean_scaled],
                                            [rbm.weights[0].W_T()])

    hid_derivs = rbm.layers[1].derivatives(hid_mean, [vdata_scaled],
                                          [rbm.weights[0].W()])

    weight_derivs = rbm.weights[0].derivatives(vdata, hid_mean_scaled)

    assert be.allclose(d_visible_loc, vis_derivs[0].loc), \
    "derivative of visible loc wrong in bernoulli-bernoulli rbm"

    assert be.allclose(d_hidden_loc, hid_derivs[0].loc), \
    "derivative of hidden loc wrong in bernoulli-bernoulli rbm"

    assert be.allclose(d_W, weight_derivs[0].matrix), \
    "derivative of weights wrong in bernoulli-bernoulli rbm"

def test_bernoulli_log_partition_gradient():
    lay = layers.BernoulliLayer(500)
    lay.params.loc[:] = be.rand_like(lay.params.loc) * 2.0 - 1.0
    A = be.rand((1,500))
    B = be.rand_like(A)
    grad = lay.grad_log_partition_function(A,B)
    logZ = be.mean(lay.log_partition_function(A,B), axis=0)
    lr = 0.01
    gogogo = True
    while gogogo:
        cop = deepcopy(lay)
        cop.params.loc[:] = lay.params.loc + lr * grad.loc
        logZ_next = be.mean(cop.log_partition_function(A,B), axis=0)
        regress = logZ_next - logZ < 0.0
        if True in regress:
            if lr < 1e-6:
                assert False, \
                "gradient of Bernoulli log partition function is wrong"
                break
            else:
                lr *= 0.5
        else:
            break

def test_bernoulli_GFE_magnetization_gradient():
    num_units = 500

    layer_1 = layers.BernoulliLayer(num_units)
    layer_2 = layers.BernoulliLayer(num_units)
    layer_3 = layers.BernoulliLayer(num_units)
    layer_4 = layers.BernoulliLayer(num_units)
    rbm = model.Model([layer_1, layer_2, layer_3, layer_4])
    for i in range(len(rbm.weights)):
        rbm.weights[i].params.matrix[:] = \
        0.01 * be.randn(rbm.weights[i].shape)

    for lay in rbm.layers:
        lay.params.loc[:] = be.rand_like(lay.params.loc)

    state = mu.StateTAP.from_model_rand(rbm)
    GFE = rbm.gibbs_free_energy(state)

    lr = 0.001
    gogogo = True
    grad = rbm._TAP_magnetization_grad(state)
    while gogogo:
        cop = deepcopy(state)
        for i in range(rbm.num_layers):
            cop.cumulants[i].mean[:] = state.cumulants[i].mean + lr * grad[i].mean

        GFE_next = rbm.gibbs_free_energy(cop)
        regress = GFE_next - GFE < 0.0
        if regress:
            if lr < 1e-6:
                assert False,\
                "Bernoulli GFE magnetization gradient is wrong"
                break
            else:
                lr *= 0.5
        else:
            break

def test_bernoulli_GFE_derivatives():
    num_units = 500

    layer_1 = layers.BernoulliLayer(num_units)
    layer_2 = layers.BernoulliLayer(num_units)
    layer_3 = layers.BernoulliLayer(num_units)

    rbm = model.Model([layer_1, layer_2, layer_3])
    for i in range(len(rbm.weights)):
        rbm.weights[i].params.matrix[:] = \
        0.01 * be.randn(rbm.weights[i].shape)

    for lay in rbm.layers:
        lay.params.loc[:] = be.rand_like(lay.params.loc)

    state = rbm.compute_StateTAP(init_lr=0.1, tol=1e-7, max_iters=50)
    GFE = rbm.gibbs_free_energy(state)

    lr = 0.1
    gogogo = True
    grad = rbm.grad_TAP_free_energy(0.1, 1e-7, 50)
    while gogogo:
        cop = deepcopy(rbm)
        lr_mul = partial(be.tmul, -lr)

        delta = gu.grad_apply(lr_mul, grad)
        cop.parameter_update(delta)

        cop_state = cop.compute_StateTAP(init_lr=0.1, tol=1e-7, max_iters=50)
        cop_GFE = cop.gibbs_free_energy(cop_state)

        regress = cop_GFE - GFE < 0.0
        print(lr, cop_GFE, GFE, cop_GFE - GFE, regress)
        if regress:
            if lr < 1e-6:
                assert False, \
                "TAP FE gradient is not working properly for Bernoulli models"
                break
            else:
                lr *= 0.5
        else:
            break


def test_gaussian_conditional_params():
    num_visible_units = 100
    num_hidden_units = 50
    batch_size = 25

    # set a seed for the random number generator
    be.set_seed()

    # set up some layer and model objects
    vis_layer = layers.GaussianLayer(num_visible_units)
    hid_layer = layers.GaussianLayer(num_hidden_units)
    rbm = model.Model([vis_layer, hid_layer])

    # randomly set the intrinsic model parameters
    a = be.randn((num_visible_units,))
    b = be.randn((num_hidden_units,))
    log_var_a = 0.1 * be.randn((num_visible_units,))
    log_var_b = 0.1 * be.randn((num_hidden_units,))
    W = be.randn((num_visible_units, num_hidden_units))

    rbm.layers[0].params.loc[:] = a
    rbm.layers[1].params.loc[:] = b
    rbm.layers[0].params.log_var[:] = log_var_a
    rbm.layers[1].params.log_var[:] = log_var_b
    rbm.weights[0].params.matrix[:] = W

    # generate a random batch of data
    vdata = rbm.layers[0].random((batch_size, num_visible_units))
    hdata = rbm.layers[1].random((batch_size, num_hidden_units))

    # compute the variance
    visible_var = be.exp(log_var_a)
    hidden_var = be.exp(log_var_b)

    # rescale the data
    vdata_scaled = vdata / visible_var
    hdata_scaled = hdata / hidden_var

    # test rescale
    assert be.allclose(vdata_scaled, rbm.layers[0].rescale(vdata)),\
    "visible rescale wrong in gaussian-gaussian rbm"

    assert be.allclose(hdata_scaled, rbm.layers[1].rescale(hdata)),\
    "hidden rescale wrong in gaussian-gaussian rbm"

    # compute the mean
    hidden_mean = be.dot(vdata_scaled, W) # (batch_size, num_hidden_units)
    hidden_mean += b

    visible_mean = be.dot(hdata_scaled, be.transpose(W)) # (batch_size, num_hidden_units)
    visible_mean += a

    # update the conditional parameters using the layer functions
    vis_mean_func, vis_var_func = rbm.layers[0]._conditional_params(
        [hdata_scaled], [rbm.weights[0].W_T()])
    hid_mean_func, hid_var_func = rbm.layers[1]._conditional_params(
        [vdata_scaled], [rbm.weights[0].W()])

    assert be.allclose(visible_var, vis_var_func),\
    "visible variance wrong in gaussian-gaussian rbm"

    assert be.allclose(hidden_var, hid_var_func),\
    "hidden variance wrong in gaussian-gaussian rbm"

    assert be.allclose(visible_mean, vis_mean_func),\
    "visible mean wrong in gaussian-gaussian rbm"

    assert be.allclose(hidden_mean, hid_mean_func),\
    "hidden mean wrong in gaussian-gaussian rbm"


def test_gaussian_derivatives():
    num_visible_units = 100
    num_hidden_units = 50
    batch_size = 25

    # set a seed for the random number generator
    be.set_seed()

    # set up some layer and model objects
    vis_layer = layers.GaussianLayer(num_visible_units)
    hid_layer = layers.GaussianLayer(num_hidden_units)
    rbm = model.Model([vis_layer, hid_layer])

    # randomly set the intrinsic model parameters
    a = be.randn((num_visible_units,))
    b = be.randn((num_hidden_units,))
    log_var_a = 0.1 * be.randn((num_visible_units,))
    log_var_b = 0.1 * be.randn((num_hidden_units,))
    W = be.randn((num_visible_units, num_hidden_units))

    rbm.layers[0].params.loc[:] = a
    rbm.layers[1].params.loc[:] = b
    rbm.layers[0].params.log_var[:] = log_var_a
    rbm.layers[1].params.log_var[:] = log_var_b
    rbm.weights[0].params.matrix[:] = W

    # generate a random batch of data
    vdata = rbm.layers[0].random((batch_size, num_visible_units))
    visible_var = be.exp(log_var_a)
    vdata_scaled = vdata / visible_var

    # compute the mean of the hidden layer
    hid_mean = rbm.layers[1].conditional_mean(
        [vdata_scaled], [rbm.weights[0].W()])
    hidden_var = be.exp(log_var_b)
    hid_mean_scaled = rbm.layers[1].rescale(hid_mean)

    # compute the derivatives
    d_vis_loc = -be.mean(vdata_scaled, axis=0)
    d_vis_logvar = -0.5 * be.mean(be.square(be.subtract(a, vdata)), axis=0)
    d_vis_logvar += be.batch_dot(hid_mean_scaled, be.transpose(W), vdata,
                                 axis=0) / len(vdata)
    d_vis_logvar /= visible_var

    d_hid_loc = -be.mean(hid_mean_scaled, axis=0)

    d_hid_logvar = -0.5 * be.mean(be.square(hid_mean - b), axis=0)
    d_hid_logvar += be.batch_dot(vdata_scaled, W, hid_mean,
                                 axis=0) / len(hid_mean)
    d_hid_logvar /= hidden_var

    d_W = -be.batch_outer(vdata_scaled, hid_mean_scaled) / len(vdata_scaled)

    # compute the derivatives using the layer functions
    vis_derivs = rbm.layers[0].derivatives(vdata, [hid_mean_scaled],
                                            [rbm.weights[0].W_T()])

    hid_derivs = rbm.layers[1].derivatives(hid_mean, [vdata_scaled],
                                           [rbm.weights[0].W()])

    weight_derivs = rbm.weights[0].derivatives(vdata_scaled, hid_mean_scaled)

    assert be.allclose(d_vis_loc, vis_derivs[0].loc), \
    "derivative of visible loc wrong in gaussian-gaussian rbm"

    assert be.allclose(d_hid_loc, hid_derivs[0].loc), \
    "derivative of hidden loc wrong in gaussian-gaussian rbm"

    assert be.allclose(d_vis_logvar, vis_derivs[0].log_var, rtol=1e-05, atol=1e-01), \
    "derivative of visible log_var wrong in gaussian-gaussian rbm"

    assert be.allclose(d_hid_logvar, hid_derivs[0].log_var, rtol=1e-05, atol=1e-01), \
    "derivative of hidden log_var wrong in gaussian-gaussian rbm"

    assert be.allclose(d_W, weight_derivs[0].matrix), \
    "derivative of weights wrong in gaussian-gaussian rbm"

if __name__ == "__main__":
    pytest.main([__file__])
