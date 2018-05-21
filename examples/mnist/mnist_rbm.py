from paysage import preprocess as pre
from paysage import layers
from paysage.models import BoltzmannMachine
from paysage import fit
from paysage import optimizers
from paysage import backends as be
from paysage import schedules
from paysage import penalties as pen

be.set_seed(137) # for determinism

import mnist_util as util

transform = pre.Transformation(pre.binarize_color)

def run(num_epochs=10, show_plot=False):
    num_hidden_units = 100
    batch_size = 100
    mc_steps = 10
    beta_std = 0.6

    # set up the reader to get minibatches
    with util.create_batch(batch_size, train_fraction=0.95, transform=transform) as data:

        # set up the model and initialize the parameters
        vis_layer = layers.BernoulliLayer(data.ncols)
        hid_layer = layers.BernoulliLayer(num_hidden_units, center=False)

        rbm = BoltzmannMachine([vis_layer, hid_layer])
        rbm.connections[0].weights.add_penalty({'matrix': pen.l2_penalty(0.001)})
        rbm.initialize(data, method='pca')

        print('training with persistent contrastive divergence')
        cd = fit.SGD(rbm, data)

        learning_rate = schedules.PowerLawDecay(initial=0.01, coefficient=0.1)
        opt = optimizers.ADAM(stepsize=learning_rate)

        cd.train(opt, num_epochs, mcsteps=mc_steps, method=fit.pcd)
        util.show_metrics(rbm, cd.monitor)

        # evaluate the model
        valid = data.get('validate')
        util.show_reconstructions(rbm, valid, show_plot, n_recon=10, vertical=False,
                                  num_to_avg=10)
        util.show_fantasy_particles(rbm, valid, show_plot, n_fantasy=5,
                                    beta_std=beta_std, fantasy_steps=100)

        util.show_weights(rbm, show_plot, n_weights=100)
        print("Done")

    return rbm

if __name__ == "__main__":
    rbm = run(show_plot = True)
