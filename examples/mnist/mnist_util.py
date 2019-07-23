import os
import numpy as np
import pandas
from math import sqrt
from paysage import samplers, schedules, batch
from paysage import backends as be

# import the plotting module using the absolute path
from importlib import util
filename = os.path.join(os.path.dirname(
           os.path.dirname(os.path.abspath(__file__))), "plotting.py")
spec = util.spec_from_file_location("plotting", location=filename)
plotting = util.module_from_spec(spec)
spec.loader.exec_module(plotting)


# ----- DEFAULT PATHS ----- #
def default_paths(file = "shuffled"):
    files = {"shuffled": {"input": "mnist.h5", "output": "shuffled_mnist.h5"},
            }
    file_path = os.path.abspath(__file__)
    mnist_path = os.path.join(os.path.dirname(file_path), files[file]["input"])
    if not os.path.exists(mnist_path):
        raise IOError("{} does not exist. run download_mnist.py to fetch from the web"
                      .format(mnist_path))
    shuffled_path = os.path.join(os.path.dirname(file_path), files[file]["output"])
    if not os.path.exists(shuffled_path):
        print("Shuffled file does not exist, creating a shuffled dataset.")
        shuffler = batch.DataShuffler(mnist_path, shuffled_path, complevel=0)
        shuffler.shuffle()
    return shuffled_path

# ----- DATA MANIPULATION ----- #

def create_batch(batch_size, train_fraction=0.95, transform=be.do_nothing):
    """
    Create a Batch reader.

    Args:
        transform (callable): the transform function.
        train_fraction (float): the training data fraction.

    Returns:
        data (Batch): a batcher.

    """
    samples = be.float_tensor(pandas.read_hdf(
                default_paths(), key='train/images').values)
    return batch.in_memory_batch(samples, batch_size, train_fraction, transform)

# ----- CHECK MODEL ----- #

def example_plot(grid, show_plot, dim=28, vmin=0, vmax=1, cmap=plotting.cm.gray):
    numpy_grid = be.to_numpy_array(grid)
    if show_plot:
        plotting.plot_image_grid(numpy_grid, (dim,dim), vmin, vmax, cmap=cmap)

def show_metrics(rbm, performance, show_plot=True):
    performance.plot_metrics(show=show_plot)

def compute_reconstructions(rbm, v_data, n_recon=10, vertical=False, num_to_avg=1):
    v_model = be.zeros_like(v_data)
    # Average over n reconstruction attempts
    for k in range(num_to_avg):
        reconstructions = rbm.compute_reconstructions(v_data)
        v_model += reconstructions.get_visible() / num_to_avg

    idx = np.random.choice(range(len(v_model)), n_recon, replace=False)
    grid = np.array([[be.to_numpy_array(v_data[i]),
                         be.to_numpy_array(v_model[i])] for i in idx])
    if vertical:
        return grid
    else:
        return grid.swapaxes(0,1)

def show_reconstructions(rbm, v_data, show_plot, dim=28, n_recon=10,
                         vertical=False, num_to_avg=1):
    print("\nPlot a random sample of reconstructions")
    grid = compute_reconstructions(rbm, v_data, n_recon, vertical, num_to_avg)
    example_plot(grid, show_plot, dim=dim)

def compute_fantasy_particles(rbm, n_fantasy=5, fantasy_steps=100, beta_std=0.6,
                              run_mean_field=True):
    schedule = schedules.Linear(initial=1.0, delta = 1 / (fantasy_steps-1))
    fantasy = samplers.SequentialMC.generate_fantasy_state(rbm,
                                                           n_fantasy*n_fantasy,
                                                           fantasy_steps,
                                                           schedule=schedule,
                                                           beta_std=beta_std,
                                                           beta_momentum=0.0)

    if run_mean_field:
        fantasy = rbm.mean_field_iteration(1, fantasy)

    v_model = fantasy[0]
    grid = np.array([be.to_numpy_array(v) for v in v_model])
    return grid.reshape(n_fantasy, n_fantasy, -1)

def show_fantasy_particles(rbm, v_data, show_plot, dim=28, n_fantasy=5,
                           fantasy_steps=100, beta_std=0.6, run_mean_field=True):
    print("\nPlot a random sample of fantasy particles")
    grid = compute_fantasy_particles(rbm, n_fantasy, fantasy_steps,
                                     beta_std=beta_std,
                                     run_mean_field=run_mean_field)
    example_plot(grid, show_plot, dim=dim)

def compute_weights(rbm, n_weights=25, l=0, random=True):
    # can't sample more than what we've got
    n_weights = min(n_weights, rbm.connections[l].shape[1])
    # floor to the nearest square below
    grid_size = int(sqrt(n_weights))
    n_weights = grid_size**2
    if random:
        idx = np.random.choice(range(rbm.connections[l].shape[1]),
                              n_weights, replace=False)
    else:
        idx = np.arange(n_weights)

    wprod = rbm.connections[0].weights.W()
    for i in range(1,l+1):
        wprod = be.dot(wprod, rbm.connections[i].weights.W())
    grid = np.array([be.to_numpy_array(wprod[:, i])
                        for i in idx])
    return grid.reshape(grid_size, grid_size, -1)

def show_weights(rbm, show_plot, dim=28, n_weights=25, random=True):
    print("\nPlot a random sample of the weights")
    for l in range(rbm.num_connections):
        grid = compute_weights(rbm, n_weights, l=l, random=random)

        # normalize the grid between -1 and +1
        maxval = np.max(np.abs(grid))
        grid /= maxval

        example_plot(grid, show_plot, dim=dim, vmin=-1, vmax=+1,
                     cmap=plotting.cm.bwr)

def weight_norm_histogram(rbm, show_plot=False, filename=None):
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, ax = plt.subplots()
    for l in range(rbm.num_connections):
        num_inputs = rbm.connections[l].shape[0]
        norm = be.to_numpy_array(be.norm(rbm.connections[l].weights.W(), axis=0) / sqrt(num_inputs))
        sns.distplot(norm, ax=ax, label=str(l))
    ax.legend()

    if show_plot:
        plt.show(fig)
    if filename is not None:
        fig.savefig(filename)
    plt.close(fig)
