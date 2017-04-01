from paysage import layers
from paysage import constraints
from paysage import penalties
from paysage import backends as be


# ----- CONSTRUCTORS ----- #

def test_Layer_creation():
    layers.Layer()

def test_Weights_creation():
    layers.Weights((8,5))

def test_Gaussian_creation():
    layers.GaussianLayer(8)

def test_Ising_creation():
    layers.IsingLayer(8)

def test_Bernoulli_creation():
    layers.BernoulliLayer(8)

def test_Exponential_creation():
    layers.ExponentialLayer(8)


# ----- BASE METHODS ----- #

def test_add_constraint():
    ly = layers.Weights((5,3))
    ly.add_constraint({'matrix': constraints.non_negative})

def test_enforce_constraints():
    ly = layers.Weights((5,3))
    ly.add_constraint({'matrix': constraints.non_negative})
    ly.enforce_constraints()

def test_add_penalty():
    ly = layers.Weights((5,3))
    p = penalties.l2_penalty(0.37)
    ly.add_penalty({'matrix': p})

def test_get_penalties():
    ly = layers.Weights((5,3))
    p = penalties.l2_penalty(0.37)
    ly.add_penalty({'matrix': p})
    ly.get_penalties()

def test_get_penalty_gradients():
    ly = layers.Weights((5,3))
    p = penalties.l2_penalty(0.37)
    ly.add_penalty({'matrix': p})
    ly.get_penalty_gradients()

def test_parameter_step():
    ly = layers.Weights((5,3))
    deltas = {'matrix': be.randn(ly.shape)}
    ly.parameter_step(deltas)

def test_get_base_config():
    ly = layers.Weights((5,3))
    ly.add_constraint({'matrix': constraints.non_negative})
    p = penalties.l2_penalty(0.37)
    ly.add_penalty({'matrix': p})
    ly.get_base_config()


# ----- Weights LAYER ----- #

def test_weights_build_from_config():
    ly = layers.Weights((5,3))
    ly.add_constraint({'matrix': constraints.non_negative})
    p = penalties.l2_penalty(0.37)
    ly.add_penalty({'matrix': p})
    ly_new = layers.Layer.from_config(ly.get_config())

def test_weights_derivative():
    ly = layers.Weights((5,3))
    p = penalties.l2_penalty(0.37)
    ly.add_penalty({'matrix': p})
    vis = be.randn((10,ly.shape[0]))
    hid = be.randn((10,ly.shape[1]))
    derivs = ly.derivatives(vis, hid)

def test_weights_energy():
    ly = layers.Weights((5,3))
    vis = be.randn((10,ly.shape[0]))
    hid = be.randn((10,ly.shape[1]))
    ly.energy(vis, hid)


# ----- Gaussian LAYER ----- #

def test_gaussian_build_from_config():
    ly = layers.GaussianLayer(8)
    ly.add_constraint({'loc': constraints.non_negative})
    p = penalties.l2_penalty(0.37)
    ly.add_penalty({'log_var': p})
    ly_new = layers.Layer.from_config(ly.get_config())

def test_gaussian_energy():
    ly = layers.GaussianLayer(8)
    vis = be.randn((10, ly.len))
    ly.energy(vis)

def test_gaussian_log_partition_function():
    ly = layers.GaussianLayer(8)
    vis = be.randn((10, ly.len))
    ly.log_partition_function(vis)

def test_gaussian_online_param_update():
    ly = layers.GaussianLayer(8)
    vis = be.randn((10, ly.len))
    ly.log_partition_function(vis)

def test_gaussian_shrink_parameters():
    ly = layers.GaussianLayer(8)
    ly.shrink_parameters(0.1)

def test_gaussian_update():
    ly = layers.GaussianLayer(8)
    w = layers.Weights((5, ly.len))
    num_samples = 10
    scaled_units = [be.ones((num_samples, w.shape[0]))]
    weights = [w.W()]
    beta = be.rand((10, 1))
    ly.update(scaled_units, weights, beta)

def test_gaussian_derivatives():
    ly = layers.GaussianLayer(8)
    w = layers.Weights((5, ly.len))
    num_samples = 10
    vis = be.randn((10, w.shape[1]))
    hid = [be.randn((10, w.shape[0]))]
    weights = [w.W_T()]
    beta = be.rand((10, 1))
    ly.derivatives(vis, hid, weights, beta)
