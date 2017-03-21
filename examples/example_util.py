import os

import numpy

import plotting
from paysage import batch
from paysage import backends as be

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

def example_plot(grid, show_plot):
    if show_plot:
        plotting.plot_image_grid(grid, (28,28), vmin=grid.min(), vmax=grid.max())

def show_metrics(rbm, performance):
    print('Final performance metrics:')
    performance.check_progress(rbm, 0, show=True)

def compute_reconstructions(rbm, v_data, fit):
    sampler = fit.DrivenSequentialMC(rbm)
    sampler.set_state(v_data)
    sampler.update_state(1)
    v_model = rbm.deterministic_step(sampler.state)

    idx = numpy.random.choice(range(len(v_model)), 5, replace=False)
    return numpy.array([[be.to_numpy_array(v_data[i]),
                         be.to_numpy_array(v_model[i])] for i in idx])

def show_reconstructions(rbm, v_data, fit, show_plot):
    print("\nPlot a random sample of reconstructions")
    grid = compute_reconstructions(rbm, v_data, fit)
    example_plot(grid, show_plot)

def compute_fantasy_particles(rbm, v_data, fit):
    random_samples = rbm.random(v_data)
    sampler = fit.DrivenSequentialMC(rbm)
    sampler.set_state(random_samples)
    sampler.update_state(1000)
    v_model = rbm.deterministic_step(sampler.state)

    idx = numpy.random.choice(range(len(v_model)), 5, replace=False)
    return numpy.array([[be.to_numpy_array(v_model[i])] for i in idx])

def show_fantasy_particles(rbm, v_data, fit, show_plot):
    print("\nPlot a random sample of fantasy particles")
    grid = compute_fantasy_particles(rbm, v_data, fit)
    example_plot(grid, show_plot)

def compute_weights(rbm):
    idx = numpy.random.choice(
          range(rbm.weights[0].shape[1]),
          5, replace=False)
    return numpy.array([[be.to_numpy_array(rbm.weights[0].W()[:, i])] for i in idx])

def show_weights(rbm, show_plot):
    print("\nPlot a random sample of the weights")
    grid = compute_weights(rbm)
    example_plot(grid, show_plot)
