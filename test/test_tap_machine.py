import os

from paysage import backends as be
from paysage import batch
from paysage import preprocess as pre
from paysage import layers
from paysage.models import BoltzmannMachine
from paysage.metrics import ReconstructionError, TAPLogLikelihood, TAPFreeEnergy
from paysage import fit
from paysage.metrics import ProgressMonitor
from paysage import optimizers
from paysage import schedules

import pytest
import pandas

def test_tap_machine(paysage_path=None):
    num_hidden_units = 10
    batch_size = 100
    num_epochs = 5
    learning_rate = schedules.PowerLawDecay(initial=0.1, coefficient=1.0)

    if not paysage_path:
        paysage_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    filepath = os.path.join(paysage_path, 'examples', 'mnist', 'mnist.h5')

    if not os.path.exists(filepath):
        raise IOError("{} does not exist. run mnist/download_mnist.py to fetch from the web".format(filepath))

    shuffled_filepath = os.path.join(paysage_path, 'examples', 'mnist', 'shuffled_mnist.h5')

    # shuffle the data
    if not os.path.exists(shuffled_filepath):
        shuffler = batch.DataShuffler(filepath, shuffled_filepath, complevel=0)
        shuffler.shuffle()

    # set a seed for the random number generator
    be.set_seed()

    # set up the reader to get minibatches
    samples = pre.binarize_color(be.float_tensor(pandas.read_hdf(
                shuffled_filepath, key='train/images').as_matrix()[:10000]))
    samples_train, samples_validate = batch.split_tensor(samples, 0.95)
    data = batch.Batch({'train': batch.InMemoryTable(samples_train, batch_size),
                        'validate': batch.InMemoryTable(samples_validate, batch_size)})

    # set up the model and initialize the parameters
    vis_layer = layers.BernoulliLayer(data.ncols)
    hid_layer = layers.BernoulliLayer(num_hidden_units)

    rbm = BoltzmannMachine([vis_layer, hid_layer])
    rbm.initialize(data)

    # obtain initial estimate of the reconstruction error
    perf = ProgressMonitor(generator_metrics = \
            [ReconstructionError(), TAPLogLikelihood(10), TAPFreeEnergy(10)])
    untrained_performance = perf.epoch_update(data, rbm, store=True, show=False)

    # set up the optimizer and the fit method
    opt = optimizers.Gradient(stepsize=learning_rate, tolerance=1e-5)
    tap = fit.TAP(True, 0.1, 0.01, 25, True, 0.5, 0.001, 0.0)
    solver = fit.SGD(rbm, data)
    solver.monitor.generator_metrics.append(TAPLogLikelihood(10))
    solver.monitor.generator_metrics.append(TAPFreeEnergy(10))

    # fit the model
    print('training with stochastic gradient ascent')
    solver.train(opt, num_epochs, method=tap.tap_update)

    # obtain an estimate of the reconstruction error after 1 epoch
    trained_performance = solver.monitor.memory[-1]

    assert (trained_performance['TAPLogLikelihood'] >
            untrained_performance['TAPLogLikelihood']), \
    "TAP log-likelihood did not increase"
    assert (trained_performance['ReconstructionError'] <
            untrained_performance['ReconstructionError']), \
    "Reconstruction error did not decrease"

    # close the HDF5 store
    data.close()

if __name__ == "__main__":
    pytest.main([__file__])
