from paysage import layers
from paysage import constraints
from paysage import penalties
from paysage import backends as be

import pytest

num_vis = 8
num_hid = 5
num_samples = 10

# ----- CONSTRUCTORS ----- #

def test_Layer_creation():
    layers.Layer()

def test_Weights_creation():
    layers.Weights((num_vis, num_hid))

def test_Gaussian_creation():
    layers.GaussianLayer(num_vis)

def test_Ising_creation():
    layers.IsingLayer(num_vis)

def test_Bernoulli_creation():
    layers.BernoulliLayer(num_vis)

def test_Exponential_creation():
    layers.ExponentialLayer(num_vis)


# ----- BASE METHODS ----- #

def test_add_constraint():
    ly = layers.Weights((num_vis, num_hid))
    ly.add_constraint({'matrix': constraints.non_negative})

def test_enforce_constraints():
    ly = layers.Weights((num_vis, num_hid))
    ly.add_constraint({'matrix': constraints.non_negative})
    ly.enforce_constraints()

def test_add_penalty():
    ly = layers.Weights((num_vis, num_hid))
    p = penalties.l2_penalty(0.37)
    ly.add_penalty({'matrix': p})

def test_get_penalties():
    ly = layers.Weights((num_vis, num_hid))
    p = penalties.l2_penalty(0.37)
    ly.add_penalty({'matrix': p})
    ly.get_penalties()

def test_get_penalty_grad():
    ly = layers.Weights((num_vis, num_hid))
    p = penalties.l2_penalty(0.37)
    ly.add_penalty({'matrix': p})
    ly.get_penalty_grad(ly.W(), 'matrix')

def test_parameter_step():
    ly = layers.Weights((num_vis, num_hid))
    deltas = layers.IntrinsicParamsWeights(be.randn(ly.shape))
    ly.parameter_step(deltas)

def test_get_base_config():
    ly = layers.Weights((num_vis, num_hid))
    ly.add_constraint({'matrix': constraints.non_negative})
    p = penalties.l2_penalty(0.37)
    ly.add_penalty({'matrix': p})
    ly.get_base_config()


# ----- Weights LAYER ----- #

def test_weights_build_from_config():
    ly = layers.Weights((num_vis, num_hid))
    ly.add_constraint({'matrix': constraints.non_negative})
    p = penalties.l2_penalty(0.37)
    ly.add_penalty({'matrix': p})
    ly_new = layers.Layer.from_config(ly.get_config())
    assert ly_new.get_config() == ly.get_config()

def test_weights_derivative():
    ly = layers.Weights((num_vis, num_hid))
    p = penalties.l2_penalty(0.37)
    ly.add_penalty({'matrix': p})
    vis = be.randn((num_samples, num_vis))
    hid = be.randn((num_samples, num_hid))
    derivs = ly.derivatives(vis, hid)

def test_weights_energy():
    ly = layers.Weights((num_vis, num_hid))
    vis = be.randn((num_samples, num_vis))
    hid = be.randn((num_samples, num_hid))
    ly.energy(vis, hid)


# ----- Gaussian LAYER ----- #

def test_gaussian_build_from_config():
    ly = layers.GaussianLayer(num_vis)
    ly.add_constraint({'loc': constraints.non_negative})
    p = penalties.l2_penalty(0.37)
    ly.add_penalty({'log_var': p})
    ly_new = layers.Layer.from_config(ly.get_config())
    assert ly_new.get_config() == ly.get_config()

def test_gaussian_energy():
    ly = layers.GaussianLayer(num_vis)
    vis = ly.random((num_samples, num_vis))
    ly.energy(vis)

def test_gaussian_log_partition_function():
    ly = layers.GaussianLayer(num_vis)
    vis = ly.random((num_samples, num_vis))
    ly.log_partition_function(vis)

def test_gaussian_online_param_update():
    ly = layers.GaussianLayer(num_vis)
    vis = ly.random((num_samples, num_vis))
    ly.online_param_update(vis)

def test_gaussian_shrink_parameters():
    ly = layers.GaussianLayer(num_vis)
    ly.shrink_parameters(0.1)

def test_gaussian_update():
    ly = layers.GaussianLayer(num_vis)
    w = layers.Weights((num_vis, num_hid))
    scaled_units = [be.randn((num_samples, num_hid))]
    weights = [w.W_T()]
    beta = be.rand((num_samples, 1))
    ly.update(scaled_units, weights, beta)

def test_gaussian_derivatives():
    ly = layers.GaussianLayer(num_vis)
    w = layers.Weights((num_vis, num_hid))
    vis = ly.random((num_samples, num_vis))
    hid = [be.randn((num_samples, num_hid))]
    weights = [w.W_T()]
    beta = be.rand((num_samples, 1))
    ly.derivatives(vis, hid, weights, beta)


# ----- Ising LAYER ----- #

def test_ising_build_from_config():
    ly = layers.IsingLayer(num_vis)
    ly.add_constraint({'loc': constraints.non_negative})
    p = penalties.l2_penalty(0.37)
    ly.add_penalty({'log_var': p})
    ly_new = layers.Layer.from_config(ly.get_config())
    assert ly_new.get_config() == ly.get_config()

def test_ising_energy():
    ly = layers.IsingLayer(num_vis)
    vis = ly.random((num_samples, num_vis))
    ly.energy(vis)

def test_ising_log_partition_function():
    ly = layers.IsingLayer(num_vis)
    vis = ly.random((num_samples, num_vis))
    ly.log_partition_function(vis)

def test_ising_online_param_update():
    ly = layers.IsingLayer(num_vis)
    vis = ly.random((num_samples, num_vis))
    ly.online_param_update(vis)

def test_ising_update():
    ly = layers.IsingLayer(num_vis)
    w = layers.Weights((num_vis, num_hid))
    scaled_units = [be.randn((num_samples, num_hid))]
    weights = [w.W_T()]
    beta = be.rand((num_samples, 1))
    ly.update(scaled_units, weights, beta)

def test_ising_derivatives():
    ly = layers.IsingLayer(num_vis)
    w = layers.Weights((num_vis, num_hid))
    vis = ly.random((num_samples, num_vis))
    hid = [be.randn((num_samples, num_hid))]
    weights = [w.W()]
    beta = be.rand((num_samples, 1))
    ly.derivatives(vis, hid, weights, beta)


# ----- Bernoulli LAYER ----- #

def test_bernoulli_build_from_config():
    ly = layers.BernoulliLayer(num_vis)
    ly.add_constraint({'loc': constraints.non_negative})
    p = penalties.l2_penalty(0.37)
    ly.add_penalty({'log_var': p})
    ly_new = layers.Layer.from_config(ly.get_config())
    assert ly_new.get_config() == ly.get_config()

def test_bernoulli_energy():
    ly = layers.BernoulliLayer(num_vis)
    vis = ly.random((num_samples, num_vis))
    ly.energy(vis)

def test_bernoulli_log_partition_function():
    ly = layers.BernoulliLayer(num_vis)
    vis = ly.random((num_samples, num_vis))
    ly.log_partition_function(vis)

def test_bernoulli_online_param_update():
    ly = layers.BernoulliLayer(num_vis)
    vis = ly.random((num_samples, num_vis))
    ly.online_param_update(vis)

def test_bernoulli_update():
    ly = layers.BernoulliLayer(num_vis)
    w = layers.Weights((num_vis, num_hid))
    scaled_units = [be.randn((num_samples, num_hid))]
    weights = [w.W_T()]
    beta = be.rand((num_samples, 1))
    ly.update(scaled_units, weights, beta)

def test_bernoulli_derivatives():
    ly = layers.BernoulliLayer(num_vis)
    w = layers.Weights((num_vis, num_hid))
    vis = ly.random((num_samples, num_vis))
    hid = [be.randn((num_samples, num_hid))]
    weights = [w.W()]
    beta = be.rand((num_samples, 1))
    ly.derivatives(vis, hid, weights, beta)


# ----- Exponential LAYER ----- #

def test_exponential_build_from_config():
    ly = layers.ExponentialLayer(num_vis)
    ly.add_constraint({'loc': constraints.non_negative})
    p = penalties.l2_penalty(0.37)
    ly.add_penalty({'log_var': p})
    ly_new = layers.Layer.from_config(ly.get_config())
    assert ly_new.get_config() == ly.get_config()

def test_exponential_energy():
    ly = layers.ExponentialLayer(num_vis)
    vis = ly.random((num_samples, num_vis))
    ly.energy(vis)

def test_exponential_log_partition_function():
    ly = layers.ExponentialLayer(num_vis)
    vis = ly.random((num_samples, num_vis))
    ly.log_partition_function(vis)

def test_exponential_online_param_update():
    ly = layers.ExponentialLayer(num_vis)
    vis = ly.random((num_samples, num_vis))
    ly.online_param_update(vis)

def test_exponential_update():
    ly = layers.ExponentialLayer(num_vis)
    w = layers.Weights((num_vis, num_hid))
    scaled_units = [be.randn((num_samples, num_hid))]
    weights = [w.W_T()]
    beta = be.rand((num_samples, 1))
    ly.update(scaled_units, weights, beta)

def test_exponential_derivatives():
    ly = layers.ExponentialLayer(num_vis)
    w = layers.Weights((num_vis, num_hid))
    vis = ly.random((num_samples, num_vis))
    hid = [be.randn((num_samples, num_hid))]
    weights = [w.W()]
    beta = be.rand((num_samples, 1))
    ly.derivatives(vis, hid, weights, beta)


if __name__ == "__main__":
    pytest.main([__file__])
