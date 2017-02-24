import os
import plotting
from paysage import batch

def default_paysage_path():
    try:
        # base on script location (eg from import or command line script)
        paysage_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    except NameError:
        # Called from outside of script or import (eg within ipython, idle, or py2exe)
        paysage_path = os.path.dirname(os.getcwd())
    return(paysage_path)

def default_mnist_path(paysage_path = None):
    if not paysage_path:
        paysage_path = default_paysage_path()
    return os.path.join(paysage_path, 'mnist', 'mnist.h5')

def default_shuffled_filepath(paysage_path):
    shuffled_filepath = os.path.join(paysage_path, 'mnist', 'shuffled_mnist.h5')
    if not os.path.exists(shuffled_filepath):
        print("Shuffled file does not exist, creating a shuffled dataset.")
        shuffler = batch.DataShuffler(filepath, shuffled_filepath, complevel=0)
        shuffler.shuffle()
    return shuffled_filepath

def default_paths(paysage_path = None):
    if not paysage_path:
        paysage_path = default_paysage_path()
    mnist_path = default_mnist_path(paysage_path)
    if not os.path.exists(mnist_path):
        raise IOError("{} does not exist. run mnist/download_mnist.py to fetch from the web".format(filepath))
    shuffled_path = default_shuffled_filepath(paysage_path)
    return (paysage_path, mnist_path, shuffled_path)

def example_plot(grid, show_plot):
    if show_plot:
        plotting.plot_image_grid(grid, (28,28), vmin=grid.min(), vmax=grid.max())
