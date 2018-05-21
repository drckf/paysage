from paysage import preprocess as pre
from paysage import layers
from paysage.models import BoltzmannMachine
from paysage import fit
from paysage import optimizers
from paysage import backends as be
from paysage import schedules

be.set_seed(137) # for determinism

import mnist_util as util

transform = pre.Transformation(pre.binarize_color)

def run(num_epochs=10, show_plot=False):

    num_hidden_units = 256
    batch_size = 100
    learning_rate = schedules.PowerLawDecay(initial=0.001, coefficient=0.1)
    mc_steps = 1

    # set up the reader to get minibatches
    data = util.create_batch(batch_size, train_fraction=0.95, transform=transform)

    # set up the model and initialize the parameters
    vis_layer = layers.BernoulliLayer(data.ncols)
    hid_layer = layers.GaussianLayer(num_hidden_units)

    rbm = BoltzmannMachine([vis_layer, hid_layer])
    rbm.initialize(data)

    # set up the optimizer and the fit method
    opt = optimizers.ADAM(stepsize=learning_rate)
    cd = fit.SGD(rbm, data)

    # fit the model
    print('training with contrastive divergence')
    cd.train(opt, num_epochs, method=fit.pcd, mcsteps=mc_steps)

    # evaluate the model
    util.show_metrics(rbm, cd.monitor)
    valid = data.get('validate')
    util.show_reconstructions(rbm, valid, show_plot, n_recon=10, vertical=False,
                              num_to_avg=10)
    util.show_fantasy_particles(rbm, valid, show_plot, n_fantasy=5)
    util.show_weights(rbm, show_plot, n_weights=25)

    # close the HDF5 store
    data.close()
    print("Done")

if __name__ == "__main__":
    run(show_plot = True)
