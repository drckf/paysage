import os
import numpy

import plotting
from paysage import batch
from paysage.models.model_utils import State
from paysage import backends as be
from paysage import schedules

# ----- DEFAULT PATHS ----- #

def default_paysage_path():
    try:
        # base on script location (eg from import or command line script)
        paysage_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    except NameError:
        # Called from outside of script or import (eg within ipython, idle, or py2exe)
        paysage_path = os.path.dirname(os.getcwd())
    return(paysage_path)

def default_filepath(paysage_path = None):
    if not paysage_path:
        paysage_path = default_paysage_path()
    return os.path.join(paysage_path, 'mnist', 'mnist.h5')

def default_shuffled_filepath(paysage_path, filepath):
    shuffled_filepath = os.path.join(paysage_path, 'mnist', 'shuffled_mnist.h5')
    if not os.path.exists(shuffled_filepath):
        print("Shuffled file does not exist, creating a shuffled dataset.")
        shuffler = batch.DataShuffler(filepath, shuffled_filepath, complevel=0)
        shuffler.shuffle()
    return shuffled_filepath

def default_paths(paysage_path = None):
    if not paysage_path:
        paysage_path = default_paysage_path()
    filepath = default_filepath(paysage_path)
    if not os.path.exists(filepath):
        raise IOError("{} does not exist. run mnist/download_mnist.py to fetch from the web"
                      .format(filepath))
    shuffled_path = default_shuffled_filepath(paysage_path, filepath)
    return (paysage_path, filepath, shuffled_path)

# ----- CHECK MODEL ----- #

def example_plot(grid, show_plot, dim=28):
    if show_plot:
        plotting.plot_image_grid(grid, (dim,dim), vmin=grid.min(), vmax=grid.max())

def show_metrics(rbm, performance):
    print('Final performance metrics:')
    performance.check_progress(rbm, show=True)

def compute_reconstructions(rbm, v_data, fit, n_recon=10, vertical=False):
    sampler = fit.DrivenSequentialMC(rbm)
    data = v_data[0] if isinstance(v_data, tuple) else v_data
    data_state = State.from_visible(data, rbm)
    sampler.set_positive_state(data_state)
    sampler.update_positive_state(1)
    v_model = rbm.deterministic_iteration(1, sampler.pos_state).units[0]

    idx = numpy.random.choice(range(len(v_model)), n_recon, replace=False)
    if isinstance(v_data, tuple):
        t_model = rbm.deterministic_iteration(1, sampler.pos_state).units[-1]
        print("Class predictions: {}".format(be.to_numpy_array(t_model)[idx]))

    grid = numpy.array([[be.to_numpy_array(data[i]),
                         be.to_numpy_array(v_model[i])] for i in idx])
    if vertical:
        return grid
    else:
        return grid.swapaxes(0,1)

def show_reconstructions(rbm, v_data, fit, show_plot, dim=28, n_recon=10, vertical=False):
    print("\nPlot a random sample of reconstructions")
    grid = compute_reconstructions(rbm, v_data, fit, n_recon, vertical)
    example_plot(grid, show_plot, dim=dim)

def compute_fantasy_particles(rbm, v_data, fit, n_fantasy=25):
    from math import sqrt
    grid_size = int(sqrt(n_fantasy))
    assert grid_size == sqrt(n_fantasy), "n_fantasy must be the square of an integer"
    
    random_samples = rbm.random(v_data)
    model_state = State.from_visible(random_samples, rbm)

    schedule = schedules.power_law_decay(initial=1.0, coefficient=0.5)
    sampler = fit.DrivenSequentialMC(rbm, schedule=schedule)

    sampler.set_negative_state(model_state)
    sampler.update_negative_state(1000)

    v_model = rbm.deterministic_iteration(1, sampler.neg_state).units[0]
    idx = numpy.random.choice(range(len(v_model)), n_fantasy, replace=False)

    grid = numpy.array([be.to_numpy_array(v_model[i]) for i in idx])
    return grid.reshape(grid_size, grid_size, -1)

def show_fantasy_particles(rbm, v_data, fit, show_plot, dim=28, n_fantasy=25):
    print("\nPlot a random sample of fantasy particles")
    grid = compute_fantasy_particles(rbm, v_data, fit, n_fantasy)
    example_plot(grid, show_plot, dim=dim)

def compute_weights(rbm, n_weights=25):
    from math import sqrt
    grid_size = int(sqrt(n_weights))
    assert grid_size == sqrt(n_weights), "n_weights must be the square of an integer"
    
    idx = numpy.random.choice(range(rbm.weights[0].shape[1]),
                              n_weights, replace=False)
    grid = numpy.array([be.to_numpy_array(rbm.weights[0].W()[:, i])
                        for i in idx])
    return grid.reshape(grid_size, grid_size, -1)

def show_weights(rbm, show_plot, dim=28, n_weights=25):
    print("\nPlot a random sample of the weights")
    grid = compute_weights(rbm, n_weights)
    example_plot(grid, show_plot, dim=dim)
