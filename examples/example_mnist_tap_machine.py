from paysage import batch
from paysage import layers
from paysage.models import tap_machine
from paysage import fit
from paysage import optimizers
from paysage import backends as be

be.set_seed(137) # for determinism

import example_util as util

def example_mnist_tap_machine(paysage_path=None, num_epochs = 10, show_plot=True):

    num_hidden_units = 500
    batch_size = 100
    learning_rate = 0.1

    (_, _, shuffled_filepath) = \
            util.default_paths(paysage_path)

    # set up the reader to get minibatches
    data = batch.Batch(shuffled_filepath,
                       'train/images',
                       batch_size,
                       transform=batch.binarize_color,
                       train_fraction=0.95)

    # set up the model and initialize the parameters
    vis_layer = layers.BernoulliLayer(data.ncols)
    hid_layer = layers.BernoulliLayer(num_hidden_units)

    rbm = tap_machine.TAP_rbm([vis_layer, hid_layer],
                              tolerance_EMF=1e-1, max_iters_EMF=100)
    rbm.initialize(data, 'hinton')

    # set up the optimizer and the fit method
    #opt = optimizers.ADAM(rbm,
    #                      stepsize=learning_rate,
    #                      scheduler=optimizers.PowerLawDecay(0.1))
    opt = optimizers.Gradient(rbm,
                              stepsize=learning_rate,
                              scheduler=optimizers.PowerLawDecay(0.1),
                              tolerance=1e-3,
                              ascent=True)


    sgd = fit.SGD(rbm,
                  data,
                  opt,
                  num_epochs)

    # fit the model
    print('training with stochastic gradient ascent ')
    sgd.train()

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
    example_mnist_tap_machine(show_plot = False)
