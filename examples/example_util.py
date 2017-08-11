import os
import numpy
from math import sqrt

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

def example_plot(grid, show_plot, dim=28, vmin=0, vmax=1, cmap=plotting.cm.gray_r):
    numpy_grid = be.to_numpy_array(grid)
    if show_plot:
        plotting.plot_image_grid(numpy_grid, (dim,dim), vmin, vmax, cmap=cmap)

def show_metrics(rbm, performance):
    print('Final performance metrics:')
    performance.check_progress(rbm, show=True)

def compute_reconstructions(rbm, v_data, fit, n_recon=10, vertical=False, num_to_avg=1):

    v_model = be.zeros_like(v_data)

    # Average over n reconstruction attempts
    for k in range(num_to_avg):
        data_state = State.from_visible(v_data, rbm)
        visible = data_state.units[0]
        reconstructions = fit.DrivenSequentialMC(rbm)
        reconstructions.set_state(data_state)
        dropout_scale = State.dropout_rescale(rbm)
        reconstructions.update_state(1, dropout_scale)
        v_model += rbm.deterministic_iteration(1, reconstructions.state, dropout_scale).units[0]

    v_model /= num_to_avg

    idx = numpy.random.choice(range(len(v_model)), n_recon, replace=False)
    grid = numpy.array([[be.to_numpy_array(visible[i]),
                         be.to_numpy_array(v_model[i])] for i in idx])
    if vertical:
        return grid
    else:
        return grid.swapaxes(0,1)

def compute_one_hot_reconstructions(rbm, fit, level, n_recon, num_to_avg=1):
    n = rbm.layers[level].len
    grid_size = int(sqrt(n_recon))
    one_hotz = rbm.layers[level].onehot(n)

    v_model = be.zeros((n, rbm.layers[0].len))
    for k in range(num_to_avg):
        # set up the initial state
        state = State.from_model(n, rbm)
        state.units[level] = one_hotz
        dropout_scale = State.dropout_rescale(rbm)

        # set up a sampler and update the state
        reconstructions = fit.SequentialMC(rbm, clamped=[level], updater='mean_field_iteration')
        reconstructions.set_state(state)
        reconstructions.update_state(10, dropout_scale)
        v_model += reconstructions.state.units[0]

    v_model /= num_to_avg
    # plot the resulting visible unit activations
    idx = numpy.random.choice(range(len(v_model)), n_recon, replace=False)
    recons = numpy.array([be.to_numpy_array(v_model[i]) for i in idx])
    recons = recons.reshape(grid_size, grid_size, -1)
    return recons

def show_one_hot_reconstructions(rbm, fit, dim=28, n_recon=25, num_to_avg=1):
    print("\nPlot a random sample of one-hot reconstructions")
    for level in range(1, len(rbm.layers)):
        print("\nLayer ", level)
        grid = compute_one_hot_reconstructions(rbm, fit, level, n_recon, num_to_avg)
        grid = grid*0.5 + 0.5
        example_plot(grid, True, dim=dim)

def show_reconstructions(rbm, v_data, fit, show_plot, dim=28, n_recon=10, vertical=False, num_to_avg=1):
    print("\nPlot a random sample of reconstructions")
    grid = compute_reconstructions(rbm, v_data, fit, n_recon, vertical, num_to_avg)
    example_plot(grid, show_plot, dim=dim)

def compute_fantasy_particles(rbm, v_data, fit, n_fantasy=25):
    grid_size = int(sqrt(n_fantasy))
    assert grid_size == sqrt(n_fantasy), "n_fantasy must be the square of an integer"

    random_samples = rbm.random(v_data)
    model_state = State.from_visible(random_samples, rbm)

    schedule = schedules.PowerLawDecay(initial=1.0, coefficient=0.5)
    fantasy = fit.DrivenSequentialMC(rbm, schedule=schedule)
    dropout_scale = State.dropout_rescale(rbm)
    fantasy.set_state(model_state)
    fantasy.update_state(1000, dropout_scale)

    v_model = rbm.deterministic_iteration(1, fantasy.state, dropout_scale).units[0]
    idx = numpy.random.choice(range(len(v_model)), n_fantasy, replace=False)

    grid = numpy.array([be.to_numpy_array(v_model[i]) for i in idx])
    return grid.reshape(grid_size, grid_size, -1)

def show_fantasy_particles(rbm, v_data, fit, show_plot, dim=28, n_fantasy=25):
    print("\nPlot a random sample of fantasy particles")
    grid = compute_fantasy_particles(rbm, v_data, fit, n_fantasy)
    example_plot(grid, show_plot, dim=dim)

def compute_weights(rbm, n_weights=25, l=0):
    grid_size = int(sqrt(n_weights))
    assert grid_size == sqrt(n_weights), "n_weights must be the square of an integer"

    idx = numpy.random.choice(range(rbm.weights[l].shape[1]),
                              n_weights, replace=False)
    grid = numpy.array([be.to_numpy_array(rbm.weights[l].W()[:, i])
                        for i in idx])
    return grid.reshape(grid_size, grid_size, -1)

def show_weights(rbm, show_plot, dim=[28, 10], n_weights=25):
    print("\nPlot a random sample of the weights")
    for l in range(rbm.num_weights):
        grid = compute_weights(rbm, n_weights, l=l)

        # normalize the grid between -1 and +1
        maxval = numpy.max(numpy.abs(grid))
        grid /= maxval

        example_plot(grid, show_plot, dim=dim[l], vmin=-1, vmax=+1,
                     cmap=plotting.cm.bwr)

def weight_norm_histogram(rbm, show_plot=False, filename=None):
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, ax = plt.subplots()
    for l in range(rbm.num_weights):
        num_inputs = rbm.weights[l].shape[0]
        norm = be.to_numpy_array(be.norm(rbm.weights[l].W(), axis=0) / sqrt(num_inputs))
        sns.distplot(norm, ax=ax, label=str(l))
    ax.legend()

    if show_plot:
        plt.show(fig)
    if filename is not None:
        fig.savefig(filename)
    plt.close(fig)
