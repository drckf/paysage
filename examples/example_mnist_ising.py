import os, sys, numpy, pandas

from paysage import batch
from paysage.models import visible
from paysage import fit

try:
    import plotting
except ImportError:
    from . import plotting

if __name__ == "__main__":

    batch_size = 50

    paysage_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    filepath = os.path.join(paysage_path, 'mnist', 'mnist.h5')

    if not os.path.exists(filepath):
        raise IOError("{} does not exist. run mnist/download_mnist.py to fetch from the web".format(filepath))

    shuffled_filepath = os.path.join(paysage_path, 'mnist', 'shuffled_mnist.h5')

    # shuffle the data
    if not os.path.exists(shuffled_filepath):
        shuffler = batch.DataShuffler(filepath, shuffled_filepath, complevel=0)
        shuffler.shuffle()

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
    performance = fit.ProgressMonitor(0,
                                      data,
                                      metrics=['EnergyDistance',
                                               'EnergyGap',
                                               'EnergyZscore'])
    print('Final performance metrics:')
    performance.check_progress(ising, 0, show=True)

    print("\nPlot a random sample of fantasy particles")
    v_data = data.get('validate')
    random_samples = ising.random(v_data)
    sampler = fit.DrivenSequentialMC(ising)
    sampler.initialize(random_samples)
    sampler.update_state(1000)
    v_model = ising.deterministic_step(sampler.state)

    idx = numpy.random.choice(range(len(v_model)), 5, replace=False)
    grid = numpy.array([[v_model[i]] for i in idx])
    plotting.plot_image_grid(grid, (28,28), vmin=grid.min(), vmax=grid.max())

    # close the HDF5 store
    data.close()
