import os, sys, numpy, pandas, time

from paysage import batch
from paysage import layers
from paysage.models import hidden
from paysage import fit
from paysage import optimizers

import example_util as util

def example_mnist_rbm(paysage_path=None, show_plot = False):
    num_hidden_units = 500
    batch_size = 50
    num_epochs = 10
    learning_rate = 0.01
    mc_steps = 1

    (paysage_path, filepath, shuffled_filepath) = \
            util.default_paths(paysage_path)

    # set up the reader to get minibatches
    data = batch.Batch(shuffled_filepath,
                       'train/images',
                       batch_size,
                       transform=batch.binarize_color,
                       train_fraction=0.99)

    # set up the model and initialize the parameters
    vis_layer = layers.BernoulliLayer(data.ncols)
    hid_layer = layers.BernoulliLayer(num_hidden_units)

    rbm = hidden.Model(vis_layer, hid_layer)
    rbm.initialize(data)

    # set up the optimizer and the fit method
    opt = optimizers.RMSProp(rbm,
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
                          'EnergyDistance'])


    # fit the model
    print('training with contrastive divergence')
    cd.train()

    # evaluate the model
    # this will be the same as the final epoch results
    # it is repeated here to be consistent with the sklearn rbm example
    metrics = ['ReconstructionError', 'EnergyDistance']
    performance = fit.ProgressMonitor(0, data, metrics=metrics)

    util.show_metrics(rbm, performance)
    util.show_reconstructions(rbm, data.get('validate'), fit, show_plot)
    util.show_fantasy_particles(rbm, data.get('validate'), fit, show_plot)
    util.show_weights(rbm, show_plot)

    # close the HDF5 store

    data.close()
    print("Done")

if __name__ == "__main__":
    example_mnist_rbm(show_plot = False)
