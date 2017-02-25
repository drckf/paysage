import os, sys, numpy, pandas, time

from functools import partial

from paysage import batch
from paysage.models import hidden
from paysage import fit
from paysage import optimizers
from paysage import backends as be

import example_util as util
import plotting

transform = partial(batch.scale, denominator=255)

def example_mnist_grbm(paysage_path = None, show_plot = True):

    num_hidden_units = 500
    batch_size = 50
    num_epochs = 10
    learning_rate = 0.001
    mc_steps = 1

    (paysage_path, filepath, shuffled_filepath) = \
        util.default_paths(paysage_path)

    # set up the reader to get minibatches
    data = batch.Batch(shuffled_filepath,
                       'train/images',
                       batch_size,
                       transform=transform,
                       train_fraction=0.99)

    # set up the model and initialize the parameters
    rbm = hidden.GRBM(data.ncols,
                      num_hidden_units,
                      hid_type = 'bernoulli')
    rbm.initialize(data, method='hinton')

    # set up the optimizer, sampler, and fit method
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
    metrics = ['ReconstructionError', 'EnergyDistance',
               'EnergyGap', 'EnergyZscore']
    performance = fit.ProgressMonitor(0, data, metrics=metrics)

    util.show_metrics(rbm, performance)
    util.show_reconstructions(rbm, data.get('validate'), fit, show_plot)
    util.show_fantasy_particles(rbm, data.get('validate'), fit, show_plot)
    util.show_weights()

    # close the HDF5 store
    data.close()
    print("Done")

if __name__ == "__main__":
    example_mnist_grbm(show_plot = False)
