from paysage import batch
from paysage.models import visible
from paysage import fit

import example_util as util
import plotting

def example_mnist_ising(paysage_path=None, show_plot = False):

    batch_size = 50

    (paysage_path, filepath, shuffled_filepath) = \
        util.default_paths(paysage_path)

    # set up the reader to get minibatches
    data = batch.Batch(shuffled_filepath,
                       'train/images',
                       batch_size,
                       transform=batch.color_to_ising,
                       train_fraction=0.99)

    # set up the model
    ising = visible.IsingModel(data.ncols)

    # use the thouless-anderson-palmer (tap) approximation to estimate params
    ising.initialize(data, method='tap')

    # evaluate the model
    metrics = ['EnergyDistance','EnergyGap', 'EnergyZscore']
    performance = fit.ProgressMonitor(0, data, metrics=metrics)

    # check model
    util.show_metrics(performance, ising)
    util.show_fantasy_particles(ising, data.get('validate'), fit, show_plot)

    # close the HDF5 store
    data.close()
    print("Done")

if __name__ == "__main__":
    example_mnist_ising(show_plot = False)
