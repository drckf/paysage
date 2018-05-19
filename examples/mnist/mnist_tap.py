from paysage import preprocess as pre
from paysage import layers
from paysage.models import BoltzmannMachine
from paysage.metrics import TAPLogLikelihood, TAPFreeEnergy
from paysage import fit
from paysage import optimizers
from paysage import backends as be
from paysage import schedules
from paysage import penalties as pen

be.set_seed(137) # for determinism

import mnist_util as util

transform = pre.Transformation(pre.binarize_color)

def run(num_epochs=5, show_plot=False):

    num_hidden_units = 256
    batch_size = 100
    learning_rate = schedules.PowerLawDecay(initial=0.1, coefficient=3.0)
    mc_steps = 1

    # set up the reader to get minibatches
    data = util.create_batch(batch_size, train_fraction=0.95, transform=transform)

    # set up the model and initialize the parameters
    vis_layer = layers.BernoulliLayer(data.ncols)
    hid_layer = layers.BernoulliLayer(num_hidden_units)

    rbm = BoltzmannMachine([vis_layer, hid_layer])
    rbm.connections[0].weights.add_penalty(
            {'matrix': pen.l1_adaptive_decay_penalty_2(0.00001)})
    rbm.initialize(data, 'glorot_normal')

    opt = optimizers.Gradient(stepsize=learning_rate, tolerance=1e-4)

    tap = fit.TAP(True, 0.1, 0.01, 25, True, 0.5, 0.001, 0.0)
    sgd = fit.SGD(rbm, data)
    sgd.monitor.generator_metrics.append(TAPLogLikelihood())
    sgd.monitor.generator_metrics.append(TAPFreeEnergy())

    # fit the model
    print('Training with stochastic gradient ascent using TAP expansion')
    sgd.train(opt, num_epochs, method=tap.tap_update, mcsteps=mc_steps)

    util.show_metrics(rbm, sgd.monitor)
    valid = data.get('validate')
    util.show_reconstructions(rbm, valid, show_plot,
                              n_recon=10, vertical=False, num_to_avg=10)
    util.show_fantasy_particles(rbm, valid, show_plot, n_fantasy=5)
    util.show_weights(rbm, show_plot, n_weights=25)
    # close the HDF5 store
    data.close()
    print("Done")

if __name__ == "__main__":
    run(show_plot = True)
