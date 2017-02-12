import os, sys, numpy, pandas, time

from paysage import batch
from paysage.models import hidden
from paysage import fit
from paysage import optimizers

import pytest

def test_rbm(paysage_path=None):
    """TODO : this is just a placeholder, need to clean up & simplifiy setup. Also
    need to figure how to deal with consistent random seeding throughout the
    codebase to obtain deterministic checkable results.
    """
    num_hidden_units = 500
    batch_size = 50
    num_epochs = 1
    learning_rate = 0.01
    mc_steps = 1

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

    # set up the reader to get minibatches
    data = batch.Batch(shuffled_filepath,
                       'train/images',
                       batch_size,
                       transform=batch.binarize_color,
                       train_fraction=0.99)

    # set up the model and initialize the parameters
    rbm = hidden.RestrictedBoltzmannMachine(data.ncols,
                                            num_hidden_units,
                                            vis_type='bernoulli',
                                            hid_type='bernoulli')
    rbm.initialize(data, method='hinton')

    # set up the optimizer and the fit method
    opt = optimizers.ADAM(rbm,
                          stepsize=learning_rate,
                          scheduler=optimizers.PowerLawDecay(0.1))

    sampler = fit.DrivenSequentialMC.from_batch(rbm, data,
                                                method='stochastic')

    cd = fit.PCD(rbm,
                 data,
                 opt,
                 sampler,
                 num_epochs,
                 mcsteps=mc_steps,
                 skip=200,
                 metrics=['ReconstructionError',
                          'EnergyDistance',
                          'EnergyGap',
                          'EnergyZscore'])


    # fit the model
    print('training with contrastive divergence')
    cd.train()

    # evaluate the model
    # this will be the same as the final epoch results
    # it is repeated here to be consistent with the sklearn rbm example
    performance = fit.ProgressMonitor(0,
                                      data,
                                      metrics=['ReconstructionError',
                                               'EnergyDistance',
                                               'EnergyGap',
                                               'EnergyZscore'])
    print('Final performance metrics:')
    performance.check_progress(rbm, 0, show=True)

    metdict = {m.name: m.value() for m in performance.metrics}

    assert metdict['ReconstructionError'] < 9, \
        "Reconstruction error too high after 1 epoch"

    assert metdict['EnergyDistance'] < 4, \
        "Energy distance too high after 1 epoch"

    # close the HDF5 store
    data.close()

if __name__ == "__main__":
    test_rbm()
