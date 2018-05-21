from paysage import backends as be
from paysage import layers
from paysage.models import BoltzmannMachine
from paysage.models import gradient_util as gu
from paysage.models.state import StateTAP
import pytest
from copy import deepcopy
from cytoolz import partial
import math

# ----- Functional Programs with Gradients ----- #

def test_zero_grad():
    num_visible_units = 100
    num_hidden_units = 50

    # set a seed for the random number generator
    be.set_seed()

    # set up some layer and model objects
    vis_layer = layers.BernoulliLayer(num_visible_units)
    hid_layer = layers.BernoulliLayer(num_hidden_units)
    rbm = BoltzmannMachine([vis_layer, hid_layer])

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
    rbm = BoltzmannMachine([vis_layer, hid_layer])

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
    rbm = BoltzmannMachine([vis_layer, hid_layer])

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
    rbm = BoltzmannMachine([vis_layer, hid_layer])

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
    rbm = BoltzmannMachine([vis_layer, hid_layer])

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
    rbm = BoltzmannMachine([vis_layer, hid_layer])

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
    rbm = BoltzmannMachine([vis_layer, hid_layer])

    # create a gradient object filled with random numbers
    grad_1 = gu.random_grad(rbm)
    grad_2 = gu.random_grad(rbm)
    gu.grad_mapzip_(be.add_, grad_1, grad_2)

def test_grad_rms():
    num_visible_units = 100
    num_hidden_units = 50

    # set a seed for the random number generator
    be.set_seed()

    # set up some layer and model objects
    vis_layer = layers.BernoulliLayer(num_visible_units)
    hid_layer = layers.BernoulliLayer(num_hidden_units)
    rbm = BoltzmannMachine([vis_layer, hid_layer])

    # create a gradient object filled with random numbers
    grad = gu.zero_grad(rbm)
    mag = gu.grad_rms(grad)
    assert mag == 0

def test_grad_norm():
    num_visible_units = 1000
    num_hidden_units = 1000

    # set a seed for the random number generator
    be.set_seed()

    # set up some layer and model objects
    vis_layer = layers.BernoulliLayer(num_visible_units)
    hid_layer = layers.BernoulliLayer(num_hidden_units)
    rbm = BoltzmannMachine([vis_layer, hid_layer])

    # create a gradient object filled with random numbers
    grad = gu.random_grad(rbm)
    nrm = gu.grad_norm(grad)
    assert nrm > math.sqrt(be.float_scalar(num_hidden_units + \
        num_visible_units + num_visible_units*num_hidden_units)/3) - 1
    assert nrm < math.sqrt(be.float_scalar(num_hidden_units + \
        num_visible_units + num_visible_units*num_hidden_units)/3) + 1

def test_grad_normalize_():
    num_visible_units = 10
    num_hidden_units = 10

    # set a seed for the random number generator
    be.set_seed()

    # set up some layer and model objects
    vis_layer = layers.BernoulliLayer(num_visible_units)
    hid_layer = layers.BernoulliLayer(num_hidden_units)
    rbm = BoltzmannMachine([vis_layer, hid_layer])

    # create a gradient object filled with random numbers
    grad = gu.random_grad(rbm)
    gu.grad_normalize_(grad)
    nrm = gu.grad_norm(grad)
    assert nrm > 1-1e-6
    assert nrm < 1+1e-6


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
    hdata = rbm.layers[1].random((batch_size, num_hidden_units))

    # compute conditional parameters
    hidden_field = be.dot(vdata, W) # (batch_size, num_hidden_units)
    hidden_field += b

    visible_field = be.dot(hdata, be.transpose(W)) # (batch_size, num_visible_units)
    visible_field += a

    # compute conditional parameters with layer funcitons
    hidden_field_layer = rbm.layers[1].conditional_params(
        [vdata], [rbm.connections[0].W()])
    visible_field_layer = rbm.layers[0].conditional_params(
        [hdata], [rbm.connections[0].W(trans=True)])

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
    vdata_scaled = rbm.layers[0].rescale(vdata)

    # compute the conditional mean of the hidden layer
    hid_mean = rbm.layers[1].conditional_mean([vdata], [rbm.connections[0].W()])
    hid_mean_scaled = rbm.layers[1].rescale(hid_mean)

    # compute the derivatives
    d_visible_loc = -be.mean(vdata, axis=0)
    d_hidden_loc = -be.mean(hid_mean_scaled, axis=0)
    d_W = -be.batch_outer(vdata, hid_mean_scaled) / len(vdata)

    # compute the derivatives using the layer functions
    vis_derivs = rbm.layers[0].derivatives(vdata, [hid_mean_scaled],
                                            [rbm.connections[0].W(trans=True)])

    hid_derivs = rbm.layers[1].derivatives(hid_mean, [vdata_scaled],
                                          [rbm.connections[0].W()])

    weight_derivs = rbm.connections[0].weights.derivatives(vdata, hid_mean_scaled)

    # compute simple weighted derivatives using the layer functions
    scale = 2
    scale_func = partial(be.multiply, be.float_scalar(scale))
    vis_derivs_scaled = rbm.layers[0].derivatives(vdata, [hid_mean_scaled],
                        [rbm.connections[0].W(trans=True)], weighting_function=scale_func)

    hid_derivs_scaled = rbm.layers[1].derivatives(hid_mean, [vdata_scaled],
                        [rbm.connections[0].W()], weighting_function=scale_func)

    weight_derivs_scaled = rbm.connections[0].weights.derivatives(vdata, hid_mean_scaled,
                                                weighting_function=scale_func)

    assert be.allclose(d_visible_loc, vis_derivs[0].loc), \
    "derivative of visible loc wrong in bernoulli-bernoulli rbm"

    assert be.allclose(d_hidden_loc, hid_derivs[0].loc), \
    "derivative of hidden loc wrong in bernoulli-bernoulli rbm"

    assert be.allclose(d_W, weight_derivs[0].matrix), \
    "derivative of weights wrong in bernoulli-bernoulli rbm"

    assert be.allclose(scale * d_visible_loc, vis_derivs_scaled[0].loc), \
    "weighted derivative of visible loc wrong in bernoulli-bernoulli rbm"

    assert be.allclose(scale * d_hidden_loc, hid_derivs_scaled[0].loc), \
    "weighted derivative of hidden loc wrong in bernoulli-bernoulli rbm"

    assert be.allclose(scale * d_W, weight_derivs_scaled[0].matrix), \
    "weighted derivative of weights wrong in bernoulli-bernoulli rbm"


def test_bernoulli_GFE_entropy_gradient():
    num_units = 5

    lay = layers.BernoulliLayer(num_units)
    lay.params.loc[:] = be.rand_like(lay.params.loc)

    from cytoolz import compose
    sum_square = compose(be.tsum, be.square)

    for itr in range(10):
        lr = 0.1
        gogogo = True
        mag = lay.get_random_magnetization()
        lms = lay.lagrange_multipliers_analytic(mag)
        entropy = lay.TAP_entropy(mag)

        grad = lay.TAP_magnetization_grad(mag, [], [], [])
        grad_mag = math.sqrt(be.float_scalar(be.accumulate(sum_square, grad)))
        normit = partial(be.tmul_, be.float_scalar(1.0/grad_mag))
        be.apply_(normit, grad)
        rand_grad = lay.get_random_magnetization()
        grad_mag = math.sqrt(be.float_scalar(be.accumulate(sum_square, rand_grad)))
        normit = partial(be.tmul_, be.float_scalar(1.0/grad_mag))
        be.apply_(normit, rand_grad)
        while gogogo:
            cop1_mag = deepcopy(mag)
            cop1_lms = deepcopy(lms)
            cop2_mag = deepcopy(mag)
            cop2_lms = deepcopy(lms)

            cop1_mag.mean[:] = mag.mean + lr * grad.mean
            cop2_mag.mean[:] = mag.mean + lr * rand_grad.mean
            lay.clip_magnetization_(cop1_mag)
            lay.clip_magnetization_(cop2_mag)
            cop1_lms = lay.lagrange_multipliers_analytic(cop1_mag)
            cop2_lms = lay.lagrange_multipliers_analytic(cop2_mag)

            entropy_1 = lay.TAP_entropy(cop1_mag)
            entropy_2 = lay.TAP_entropy(cop2_mag)

            regress = entropy_1 - entropy_2 < 0.0
            #print(itr, "[",lr, "] ", entropy, entropy_1, entropy_2, regress)
            if regress:
                #print(grad, rand_grad)
                if lr < 1e-6:
                    assert False,\
                    "Bernoulli GFE magnetization gradient is wrong"
                    break
                else:
                    lr *= 0.5
            else:
                break

def test_gaussian_GFE_entropy_gradient():
    num_units = 5
    lay = layers.GaussianLayer(num_units)

    lay.params.loc[:] = be.rand_like(lay.params.loc)
    lay.params.log_var[:] = be.randn(be.shape(lay.params.loc))

    from cytoolz import compose
    sum_square = compose(be.tsum, be.square)

    for itr in range(10):
        mag = lay.get_random_magnetization()
        lms = lay.lagrange_multipliers_analytic(mag)
        entropy = lay.TAP_entropy(mag)
        lr = 0.001
        gogogo = True
        grad = lay.TAP_magnetization_grad(mag, [], [], [])
        grad_mag = math.sqrt(be.float_scalar(be.accumulate(sum_square, grad)))
        normit = partial(be.tmul_, be.float_scalar(1.0/grad_mag))
        be.apply_(normit, grad)
        rand_grad = lay.get_random_magnetization()
        grad_mag = math.sqrt(be.float_scalar(be.accumulate(sum_square, rand_grad)))
        normit = partial(be.tmul_, be.float_scalar(1.0/grad_mag))
        be.apply_(normit, rand_grad)
        while gogogo:
            cop1_mag = deepcopy(mag)
            cop1_lms = deepcopy(lms)
            cop2_mag = deepcopy(mag)
            cop2_lms = deepcopy(lms)

            cop1_mag.mean[:] = mag.mean + lr * grad.mean
            cop2_mag.mean[:] = mag.mean + lr * rand_grad.mean
            cop1_mag.variance[:] = mag.variance + lr * grad.variance
            cop2_mag.variance[:] = mag.variance + lr * rand_grad.variance
            lay.clip_magnetization_(cop1_mag)
            lay.clip_magnetization_(cop2_mag)
            cop1_lms = lay.lagrange_multipliers_analytic(cop1_mag)
            cop2_lms = lay.lagrange_multipliers_analytic(cop2_mag)

            entropy_1 = lay.TAP_entropy(cop1_mag)
            entropy_2 = lay.TAP_entropy(cop2_mag)

            regress = entropy_1 - entropy_2 < 0.0
            #print(itr, "[",lr, "] ", entropy, entropy_1, entropy_2, regress)
            if regress:
                #print(grad, rand_grad)
                if lr < 1e-6:
                    assert False,\
                    "Gaussian GFE magnetization gradient is wrong"
                    break
                else:
                    lr *= 0.5
            else:
                break

def test_bernoulli_GFE_derivatives():
    # Tests that the GFE derivative update increases GFE versus 100
    # random update vectors
    num_units = 5

    layer_1 = layers.BernoulliLayer(num_units)
    layer_2 = layers.BernoulliLayer(num_units)
    layer_3 = layers.BernoulliLayer(num_units)
    rbm = BoltzmannMachine([layer_1, layer_2, layer_3])

    for i in range(len(rbm.connections)):
        rbm.connections[i].weights.params.matrix[:] = \
        0.01 * be.randn(rbm.connections[i].shape)

    for lay in rbm.layers:
        lay.params.loc[:] = be.rand_like(lay.params.loc)

    state, cop1_GFE = rbm.compute_StateTAP(init_lr=0.1, tol=1e-7, max_iters=50)
    grad = rbm._grad_gibbs_free_energy(state)
    gu.grad_normalize_(grad)

    for i in range(100):
        lr = 1.0
        gogogo = True
        random_grad = gu.random_grad(rbm)
        gu.grad_normalize_(random_grad)
        while gogogo:
            cop1 = deepcopy(rbm)
            lr_mul = partial(be.tmul, lr)

            cop1.parameter_update(gu.grad_apply(lr_mul, grad))
            cop1_state, cop1_GFE = cop1.compute_StateTAP(init_lr=0.1, tol=1e-7, max_iters=50)

            cop2 = deepcopy(rbm)
            cop2.parameter_update(gu.grad_apply(lr_mul, random_grad))
            cop2_state, cop2_GFE = cop2.compute_StateTAP(init_lr=0.1, tol=1e-7, max_iters=50)

            regress = cop2_GFE - cop1_GFE < 0.0

            if regress:
                if lr < 1e-6:
                    assert False, \
                    "TAP FE gradient is not working properly for Bernoulli models"
                    break
                else:
                    lr *= 0.5
            else:
                break

def test_gaussian_GFE_derivatives_gradient_descent():
    num_units = 5

    layer_1 = layers.GaussianLayer(num_units)
    layer_2 = layers.BernoulliLayer(num_units)

    rbm = BoltzmannMachine([layer_1, layer_2])

    for i in range(len(rbm.connections)):
        rbm.connections[i].weights.params.matrix[:] = \
        0.01 * be.randn(rbm.connections[i].shape)

    for lay in rbm.layers:
        lay.params.loc[:] = be.rand_like(lay.params.loc)

    state, GFE = rbm.compute_StateTAP(use_GD=False, tol=1e-7, max_iters=50)
    grad = rbm._grad_gibbs_free_energy(state)
    gu.grad_normalize_(grad)

    for i in range(100):
        lr = 0.001
        gogogo = True
        random_grad = gu.random_grad(rbm)
        gu.grad_normalize_(random_grad)
        while gogogo:
            cop1 = deepcopy(rbm)
            lr_mul = partial(be.tmul, lr)

            cop1.parameter_update(gu.grad_apply(lr_mul, grad))
            cop1_state, cop1_GFE = cop1.compute_StateTAP(use_GD=False, tol=1e-7, max_iters=50)

            cop2 = deepcopy(rbm)
            cop2.parameter_update(gu.grad_apply(lr_mul, random_grad))
            cop2_state, cop2_GFE = cop2.compute_StateTAP(use_GD=False, tol=1e-7, max_iters=50)

            regress = cop2_GFE - cop1_GFE < 0

            if regress:
                if lr < 1e-6:
                    assert False, \
                    "TAP FE gradient is not working properly for Gaussian models"
                    break
                else:
                    lr *= 0.5
            else:
                break

def test_gaussian_GFE_derivatives_self_consistent():
    num_units = 5

    layer_1 = layers.GaussianLayer(num_units)
    layer_2 = layers.BernoulliLayer(num_units)

    rbm = BoltzmannMachine([layer_1, layer_2])
    for i in range(len(rbm.connections)):
        rbm.connections[i].weights.params.matrix[:] = \
        0.01 * be.randn(rbm.connections[i].shape)

    for lay in rbm.layers:
        lay.params.loc[:] = be.rand_like(lay.params.loc)

    state, cop1_GFE = rbm._compute_StateTAP_self_consistent()
    grad = rbm._grad_gibbs_free_energy(state)
    gu.grad_normalize_(grad)

    for i in range(100):
        lr = 0.1
        gogogo = True
        random_grad = gu.random_grad(rbm)
        gu.grad_normalize_(random_grad)
        while gogogo:
            cop1 = deepcopy(rbm)
            lr_mul = partial(be.tmul, lr)

            cop1.parameter_update(gu.grad_apply(lr_mul, grad))
            cop1_state, cop1_GFE = cop1._compute_StateTAP_self_consistent()

            cop2 = deepcopy(rbm)
            cop2.parameter_update(gu.grad_apply(lr_mul, random_grad))
            cop2_state, cop2_GFE = cop2._compute_StateTAP_self_consistent()

            regress = cop2_GFE - cop1_GFE < 0.0

            if regress:
                if lr < 1e-6:
                    assert False, \
                    "TAP FE gradient is not working properly for Gaussian models"
                    break
                else:
                    lr *= 0.5
            else:
                break

def test_gaussian_Compute_StateTAP_self_consistent():
    num_units = 10

    layer_1 = layers.GaussianLayer(num_units)
    layer_2 = layers.BernoulliLayer(num_units)

    rbm = BoltzmannMachine([layer_1, layer_2])
    for i in range(len(rbm.connections)):
        rbm.connections[i].weights.params.matrix[:] = \
        0.01 * be.randn(rbm.connections[i].shape)

    for lay in rbm.layers:
        lay.params.loc[:] = be.rand_like(lay.params.loc)

    for i in range(100):
        random_state = StateTAP.from_model_rand(rbm)
        GFE = rbm.gibbs_free_energy(random_state.cumulants)
        _,min_GFE = rbm._compute_StateTAP_self_consistent(seed=random_state)

        if GFE - min_GFE < 0.0:
            assert False, \
                "compute_StateTAP_self_consistent is not reducing the GFE"

def test_gaussian_Compute_StateTAP_GD():
    num_units = 10

    layer_1 = layers.GaussianLayer(num_units)
    layer_2 = layers.BernoulliLayer(num_units)

    rbm = BoltzmannMachine([layer_1, layer_2])
    for i in range(len(rbm.connections)):
        rbm.connections[i].weights.params.matrix[:] = \
        0.01 * be.randn(rbm.connections[i].shape)

    for lay in rbm.layers:
        lay.params.loc[:] = be.rand_like(lay.params.loc)

    for i in range(100):
        random_state = StateTAP.from_model_rand(rbm)
        GFE = rbm.gibbs_free_energy(random_state.cumulants)
        _,min_GFE = rbm._compute_StateTAP_GD(seed=random_state)

        if GFE - min_GFE < 0.0:
            assert False, \
                "compute_StateTAP_self_consistent is not reducing the GFE"


def test_gaussian_conditional_params():
    num_visible_units = 100
    num_hidden_units = 50
    batch_size = 25

    # set a seed for the random number generator
    be.set_seed()

    # set up some layer and model objects
    vis_layer = layers.GaussianLayer(num_visible_units)
    hid_layer = layers.GaussianLayer(num_hidden_units)
    rbm = BoltzmannMachine([vis_layer, hid_layer])

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
    rbm.connections[0].weights.params.matrix[:] = W

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
    vis_mean_func, vis_var_func = rbm.layers[0].conditional_params(
        [hdata_scaled], [rbm.connections[0].W(trans=True)])
    hid_mean_func, hid_var_func = rbm.layers[1].conditional_params(
        [vdata_scaled], [rbm.connections[0].W()])

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
    rbm = BoltzmannMachine([vis_layer, hid_layer])

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
    rbm.connections[0].weights.params.matrix[:] = W

    # generate a random batch of data
    vdata = rbm.layers[0].random((batch_size, num_visible_units))
    visible_var = be.exp(log_var_a)
    vdata_scaled = vdata / visible_var

    # compute the mean of the hidden layer
    hid_mean = rbm.layers[1].conditional_mean(
        [vdata_scaled], [rbm.connections[0].W()])
    hidden_var = be.exp(log_var_b)
    hid_mean_scaled = rbm.layers[1].rescale(hid_mean)

    # compute the derivatives
    d_vis_loc = be.mean((a-vdata)/visible_var, axis=0)
    d_vis_logvar = -0.5 * be.mean(be.square(be.subtract(a, vdata)), axis=0)
    d_vis_logvar += be.batch_quadratic(hid_mean_scaled, be.transpose(W), vdata,
                                 axis=0) / len(vdata)
    d_vis_logvar /= visible_var

    d_hid_loc = be.mean((b-hid_mean)/hidden_var, axis=0)

    d_hid_logvar = -0.5 * be.mean(be.square(hid_mean - b), axis=0)
    d_hid_logvar += be.batch_quadratic(vdata_scaled, W, hid_mean,
                                 axis=0) / len(hid_mean)
    d_hid_logvar /= hidden_var

    d_W = -be.batch_outer(vdata_scaled, hid_mean_scaled) / len(vdata_scaled)

    # compute the derivatives using the layer functions
    vis_derivs = rbm.layers[0].derivatives(vdata, [hid_mean_scaled],
                                            [rbm.connections[0].W(trans=True)])

    hid_derivs = rbm.layers[1].derivatives(hid_mean, [vdata_scaled],
                                           [rbm.connections[0].W()])

    weight_derivs = rbm.connections[0].weights.derivatives(vdata_scaled, hid_mean_scaled)

    # compute simple weighted derivatives using the layer functions
    scale = 2
    scale_func = partial(be.multiply, be.float_scalar(scale))
    vis_derivs_scaled = rbm.layers[0].derivatives(vdata, [hid_mean_scaled],
                        [rbm.connections[0].W(trans=True)], weighting_function=scale_func)

    hid_derivs_scaled = rbm.layers[1].derivatives(hid_mean, [vdata_scaled],
                        [rbm.connections[0].W()], weighting_function=scale_func)

    weight_derivs_scaled = rbm.connections[0].weights.derivatives(vdata_scaled, hid_mean_scaled,
                                                weighting_function=scale_func)

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

    assert be.allclose(scale * d_vis_loc, vis_derivs_scaled[0].loc), \
    "weighted derivative of visible loc wrong in gaussian-gaussian rbm"

    assert be.allclose(scale * d_hid_loc, hid_derivs_scaled[0].loc), \
    "weighted derivative of hidden loc wrong in gaussian-gaussian rbm"

    assert be.allclose(scale * d_vis_logvar, vis_derivs_scaled[0].log_var, rtol=1e-05, atol=1e-01), \
    "weighted derivative of visible log_var wrong in gaussian-gaussian rbm"

    assert be.allclose(scale * d_hid_logvar, hid_derivs_scaled[0].log_var, rtol=1e-05, atol=1e-01), \
    "weighted derivative of hidden log_var wrong in gaussian-gaussian rbm"

    assert be.allclose(scale * d_W, weight_derivs_scaled[0].matrix), \
    "weighted derivative of weights wrong in gaussian-gaussian rbm"


def test_onehot_conditional_params():
    num_visible_units = 100
    num_hidden_units = 50
    batch_size = 25

    # set a seed for the random number generator
    be.set_seed()

    # set up some layer and model objects
    vis_layer = layers.OneHotLayer(num_visible_units)
    hid_layer = layers.OneHotLayer(num_hidden_units)
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
    hdata = rbm.layers[1].random((batch_size, num_hidden_units))

    # compute conditional parameters
    hidden_field = be.dot(vdata, W) # (batch_size, num_hidden_units)
    hidden_field += b

    visible_field = be.dot(hdata, be.transpose(W)) # (batch_size, num_visible_units)
    visible_field += a

    # compute conditional parameters with layer funcitons
    hidden_field_layer = rbm.layers[1].conditional_params(
        [vdata], [rbm.connections[0].W()])
    visible_field_layer = rbm.layers[0].conditional_params(
        [hdata], [rbm.connections[0].W(trans=True)])

    assert be.allclose(hidden_field, hidden_field_layer), \
    "hidden field wrong in onehot-onehot rbm"

    assert be.allclose(visible_field, visible_field_layer), \
    "visible field wrong in onehot-onehot rbm"


def test_onehot_derivatives():
    num_visible_units = 100
    num_hidden_units = 50
    batch_size = 25

    # set a seed for the random number generator
    be.set_seed()

    # set up some layer and model objects
    vis_layer = layers.OneHotLayer(num_visible_units)
    hid_layer = layers.OneHotLayer(num_hidden_units)
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
    vdata_scaled = rbm.layers[0].rescale(vdata)

    # compute the conditional mean of the hidden layer
    hid_mean = rbm.layers[1].conditional_mean([vdata], [rbm.connections[0].W()])
    hid_mean_scaled = rbm.layers[1].rescale(hid_mean)

    # compute the derivatives
    d_visible_loc = -be.mean(vdata, axis=0)
    d_hidden_loc = -be.mean(hid_mean_scaled, axis=0)
    d_W = -be.batch_outer(vdata, hid_mean_scaled) / len(vdata)

    # compute the derivatives using the layer functions
    vis_derivs = rbm.layers[0].derivatives(vdata, [hid_mean_scaled],
                                            [rbm.connections[0].W(trans=True)])

    hid_derivs = rbm.layers[1].derivatives(hid_mean, [vdata_scaled],
                                          [rbm.connections[0].W()])

    weight_derivs = rbm.connections[0].weights.derivatives(vdata, hid_mean_scaled)

    # compute simple weighted derivatives using the layer functions
    scale = 2
    scale_func = partial(be.multiply, be.float_scalar(scale))
    vis_derivs_scaled = rbm.layers[0].derivatives(vdata, [hid_mean_scaled],
                        [rbm.connections[0].W(trans=True)], weighting_function=scale_func)

    hid_derivs_scaled = rbm.layers[1].derivatives(hid_mean, [vdata_scaled],
                          [rbm.connections[0].W()], weighting_function=scale_func)

    weight_derivs_scaled = rbm.connections[0].weights.derivatives(vdata, hid_mean_scaled,
                                                weighting_function=scale_func)

    assert be.allclose(d_visible_loc, vis_derivs[0].loc), \
    "derivative of visible loc wrong in onehot-onehot rbm"

    assert be.allclose(d_hidden_loc, hid_derivs[0].loc), \
    "derivative of hidden loc wrong in onehot-onehot rbm"

    assert be.allclose(d_W, weight_derivs[0].matrix), \
    "derivative of weights wrong in onehot-onehot rbm"

    assert be.allclose(scale * d_visible_loc, vis_derivs_scaled[0].loc), \
    "weighted derivative of visible loc wrong in onehot-onehot rbm"

    assert be.allclose(scale * d_hidden_loc, hid_derivs_scaled[0].loc), \
    "weighted derivative of hidden loc wrong in onehot-onehot rbm"

    assert be.allclose(scale * d_W, weight_derivs_scaled[0].matrix), \
    "weighted derivative of weights wrong in onehot-onehot rbm"

if __name__ == "__main__":
    pytest.main([__file__])
