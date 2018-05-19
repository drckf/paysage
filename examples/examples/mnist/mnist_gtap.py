from paysage import preprocess as pre
from paysage import layers
from paysage.models import BoltzmannMachine
from paysage.metrics import TAPLogLikelihood, TAPFreeEnergy
from paysage import fit
from paysage import optimizers
from paysage import backends as be
from paysage import schedules

be.set_seed(137) # for determinism

import mnist_util as util

transform = pre.Transformation(pre.scale, kwargs={'denominator': 255})

def run(num_epochs=10, show_plot=False):

    num_hidden_units = 256
    batch_size = 100
    learning_rate = schedules.PowerLawDecay(initial=0.001, coefficient=0.1)
    mc_steps = 1

    # set up the reader to get minibatches
    data = util.create_batch(batch_size, train_fraction=0.95, transform=transform)

    # set up the model and initialize the parameters
    vis_layer = layers.GaussianLayer(data.ncols)
    hid_layer = layers.BernoulliLayer(num_hidden_units)

    rbm = BoltzmannMachine([vis_layer, hid_layer])
    rbm.initialize(data, 'stddev')
    rbm.layers[0].params.log_var[:] = \
      be.log(0.05*be.ones_like(rbm.layers[0].params.log_var))

    opt = optimizers.ADAM(stepsize=learning_rate)

    # This example parameter set for TAP uses gradient descent to optimize the
    # Gibbs free energy:
    tap = fit.TAP(True, 1.0, 0.01, 100, False, 0.9, 0.001, 0.5)

    # This example parameter set for TAP uses self-consistent iteration to
    # optimize the Gibbs free energy:
    #tap = fit.TAP(False, tolerance=0.001, max_iters=100)
    sgd = fit.SGD(rbm, data)
    sgd.monitor.generator_metrics.append(TAPFreeEnergy())
    sgd.monitor.generator_metrics.append(TAPLogLikelihood())

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
