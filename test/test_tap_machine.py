import os

from paysage import backends as be
from paysage import batch
from paysage import preprocess as pre
from paysage import layers
from paysage.models import model
from paysage import fit
from paysage import optimizers
from paysage import schedules

import pytest
import pandas

def test_tap_machine(paysage_path=None):
    num_hidden_units = 10
    batch_size = 10
    num_epochs = 1
    learning_rate = schedules.PowerLawDecay(initial=0.1, coefficient=0.1)

    if not paysage_path:
        paysage_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    filepath = os.path.join(paysage_path, 'mnist', 'mnist.h5')

    if not os.path.exists(filepath):
        raise IOError("{} does not exist. run mnist/download_mnist.py to fetch from the web".format(filepath))

    shuffled_filepath = os.path.join(paysage_path, 'mnist', 'shuffled_mnist.h5')

    # shuffle the data
    if not os.path.exists(shuffled_filepath):
        shuffler = batch.DataShuffler(filepath, shuffled_filepath, complevel=0)
        shuffler.shuffle()

    # set a seed for the random number generator
    be.set_seed()

    # set up the reader to get minibatches
    small_dataset = pandas.read_hdf(shuffled_filepath, key='train/images').as_matrix()[:1000,]
    print(small_dataset.shape)
    data = batch.InMemoryBatch(
           pre.binarize_color(be.float_tensor(small_dataset)),
           batch_size,
           train_fraction=0.95
           )

    # set up the model and initialize the parameters
    vis_layer = layers.BernoulliLayer(data.ncols)
    hid_layer = layers.BernoulliLayer(num_hidden_units)

    rbm = model.Model([vis_layer, hid_layer])
    rbm.initialize(data)

    # obtain initial estimate of the reconstruction error
    perf  = fit.ProgressMonitor(data,
                                metrics=[
                                'ReconstructionError'])
    untrained_performance = perf.check_progress(rbm)


    # set up the optimizer and the fit method
    opt = optimizers.Gradient(stepsize=learning_rate,
                              tolerance=1e-3,
                              ascent=True)

    sampler = fit.SequentialMC(rbm)

    solver = fit.SGD(rbm, data, opt, num_epochs, sampler, method=fit.tap,
                     monitor=perf)

    # fit the model
    print('training with stochastic gradient ascent')
    solver.train()

    # obtain an estimate of the reconstruction error after 1 epoch
    trained_performance = perf.check_progress(rbm)

    assert (trained_performance['ReconstructionError'] <
            untrained_performance['ReconstructionError']), \
    "Reconstruction error did not decrease"

    # close the HDF5 store
    data.close()

if __name__ == "__main__":
    pytest.main([__file__])
