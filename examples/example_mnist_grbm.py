from functools import partial

from paysage import batch
from paysage import layers
from paysage.models import model
from paysage import fit
from paysage import optimizers
from paysage import backends as be

be.set_seed(137) # for determinism

import example_util as util

transform = partial(batch.scale, denominator=255)

def example_mnist_grbm(paysage_path=None, num_epochs=10, show_plot=False):

    num_hidden_units = 500
    batch_size = 50
    learning_rate = 0.001 # gaussian rbm usually requires smaller learnign rate
    mc_steps = 1

    (_, _, shuffled_filepath) = \
            util.default_paths(paysage_path)

    # set up the reader to get minibatches
    data = batch.Batch(shuffled_filepath,
                       'train/images',
                       batch_size,
                       transform=transform,
                       train_fraction=0.99)

    # set up the model and initialize the parameters
    vis_layer = layers.GaussianLayer(data.ncols)
    hid_layer = layers.BernoulliLayer(num_hidden_units)

    rbm = model.Model([vis_layer, hid_layer])
    rbm.initialize(data)

    # set up the optimizer, sampler, and fit method
    opt = optimizers.ADAM(stepsize=learning_rate,
                          scheduler=optimizers.PowerLawDecay(0.1))

    sampler = fit.DrivenSequentialMC.from_batch(rbm, data,
                                                method='stochastic')

    cd = fit.PCD(rbm, data, opt, sampler, num_epochs, mcsteps=mc_steps, skip=200,
                 metrics=['ReconstructionError',
                          'EnergyDistance',
                          'EnergyGap',
                          'EnergyZscore'])

    # fit the model
    print('training with contrastive divergence')
    cd.train()

    # evaluate the model
    # this will be the same as the final epoch results
    metrics = ['ReconstructionError', 'EnergyDistance', 'EnergyGap', 'EnergyZscore']
    performance = fit.ProgressMonitor(0, data, metrics=metrics)

    util.show_metrics(rbm, performance)
    util.show_reconstructions(rbm, data.get('validate'), fit, show_plot)
    util.show_fantasy_particles(rbm, data.get('validate'), fit, show_plot)
    util.show_weights(rbm, show_plot)

    # close the HDF5 store
    data.close()
    print("Done")

if __name__ == "__main__":
    example_mnist_grbm(show_plot = False)
