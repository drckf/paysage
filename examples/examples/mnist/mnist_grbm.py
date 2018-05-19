from paysage import preprocess as pre
from paysage import layers
from paysage.models import BoltzmannMachine
from paysage import fit
from paysage import optimizers
from paysage import backends as be
from paysage import schedules
from paysage import metrics as M

be.set_seed(137) # for determinism

import mnist_util as util

transform = pre.Transformation(pre.scale, kwargs={'denominator': 255})

def run(num_epochs=20, show_plot=False):
    num_hidden_units = 200
    batch_size = 100
    mc_steps = 10
    beta_std = 0.95

    # set up the reader to get minibatches
    data = util.create_batch(batch_size, train_fraction=0.95, transform=transform)

    # set up the model and initialize the parameters
    vis_layer = layers.GaussianLayer(data.ncols, center=False)
    hid_layer = layers.BernoulliLayer(num_hidden_units, center=True)
    hid_layer.set_fixed_params(hid_layer.get_param_names())

    rbm = BoltzmannMachine([vis_layer, hid_layer])
    rbm.initialize(data, 'pca', epochs = 500, verbose=True)

    print('training with persistent contrastive divergence')
    cd = fit.SGD(rbm, data, fantasy_steps=10)
    cd.monitor.generator_metrics.append(M.JensenShannonDivergence())

    learning_rate = schedules.PowerLawDecay(initial=1e-3, coefficient=5)
    opt = optimizers.ADAM(stepsize=learning_rate)
    cd.train(opt, num_epochs, method=fit.pcd, mcsteps=mc_steps,
             beta_std=beta_std, burn_in=1)

    # evaluate the model
    util.show_metrics(rbm, cd.monitor)
    valid = data.get('validate')
    util.show_reconstructions(rbm, valid, show_plot, n_recon=10, vertical=False)
    util.show_fantasy_particles(rbm, valid, show_plot, n_fantasy=5)
    util.show_weights(rbm, show_plot, n_weights=100)

    # close the HDF5 store
    data.close()
    print("Done")

    return rbm

if __name__ == "__main__":
    rbm = run(show_plot = True)
    import seaborn
    import matplotlib.pyplot as plt
    for conn in rbm.connections:
        c = be.corr(conn.weights.W(), conn.weights.W())
        fig, ax = plt.subplots()
        seaborn.heatmap(be.to_numpy_array(c), vmin=-1, vmax=1, ax=ax)
        plt.show(fig)

        n = be.norm(conn.weights.W(), axis=0)
        fig, ax = plt.subplots()
        seaborn.distplot(be.to_numpy_array(n), ax=ax)
        plt.show(fig)
