from paysage import batch
from paysage import layers
from paysage.models import tap_machine
from paysage import fit
from paysage import optimizers
from paysage import backends as be

be.set_seed(137) # for determinism

import example_util as util

def example_mnist_tap_machine(paysage_path=None, num_epochs = 10, show_plot=True):

    num_hidden_units = 256
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

    rbm = tap_machine.TAP_rbm([vis_layer, hid_layer], num_persistent_samples=0,
                              tolerance_EMF=1e-4, max_iters_EMF=25, terms=2)
    rbm.initialize(data, 'glorot_normal')

    perf  = fit.ProgressMonitor(data,
                                metrics=['ReconstructionError',
                                         'EnergyDistance',
                                         'HeatCapacity'])

    opt = optimizers.Gradient(stepsize=learning_rate,
                              scheduler=optimizers.PowerLawDecay(0.1),
                              tolerance=1e-4,
                              ascent=True)

    sgd = fit.SGD(rbm, data, opt, num_epochs, method=fit.tap, monitor=perf)

    # fit the model
    print('training with stochastic gradient ascent ')
    sgd.train()

    util.show_metrics(rbm, perf)
    util.show_reconstructions(rbm, data.get('validate'), fit, show_plot)
    util.show_fantasy_particles(rbm, data.get('validate'), fit, show_plot)
    util.show_weights(rbm, show_plot)

    # close the HDF5 store
    data.close()
    print("Done")

if __name__ == "__main__":
    example_mnist_tap_machine(show_plot = True)
