from paysage import preprocess as pre
from paysage import layers
from paysage.models import BoltzmannMachine
from paysage import fit
from paysage import optimizers
from paysage import backends as be
from paysage import schedules
from paysage import penalties as pen
from paysage import metrics as M

be.set_seed(137) # for determinism

import mnist_util as util

transform = pre.Transformation(pre.scale, kwargs={'denominator': 255})

def run(pretrain_epochs=5, finetune_epochs=5, fit_method=fit.LayerwisePretrain,
        show_plot=False):
    num_hidden_units = [20**2, 15**2, 10**2]
    batch_size = 100
    mc_steps = 5
    beta_std = 0.6

    # set up the reader to get minibatches
    data = util.create_batch(batch_size, train_fraction=0.95, transform=transform)

    # set up the model and initialize the parameters
    vis_layer = layers.BernoulliLayer(data.ncols)
    hid_layer = [layers.BernoulliLayer(n) for n in num_hidden_units]
    rbm = BoltzmannMachine([vis_layer] + hid_layer)

    # add some penalties
    for c in rbm.connections:
        c.weights.add_penalty({"matrix": pen.l1_adaptive_decay_penalty_2(1e-4)})

    print("Norms of the weights before training")
    util.weight_norm_histogram(rbm, show_plot=show_plot)

    print('pre-training with persistent contrastive divergence')
    cd = fit_method(rbm, data)
    learning_rate = schedules.PowerLawDecay(initial=5e-3, coefficient=1)
    opt = optimizers.ADAM(stepsize=learning_rate)
    cd.train(opt, pretrain_epochs, method=fit.pcd, mcsteps=mc_steps,
             init_method="glorot_normal")

    util.show_weights(rbm, show_plot, n_weights=16)

    print('fine tuning')
    cd = fit.StochasticGradientDescent(rbm, data)
    cd.monitor.generator_metrics.append(M.JensenShannonDivergence())

    learning_rate = schedules.PowerLawDecay(initial=1e-3, coefficient=1)
    opt = optimizers.ADAM(stepsize=learning_rate)
    cd.train(opt, finetune_epochs, mcsteps=mc_steps, beta_std=beta_std)
    util.show_metrics(rbm, cd.monitor)

    # evaluate the model
    valid = data.get('validate')
    util.show_reconstructions(rbm, valid, show_plot, num_to_avg=10)
    util.show_fantasy_particles(rbm, valid, show_plot, n_fantasy=10,
                                beta_std=beta_std, fantasy_steps=100)

    util.show_weights(rbm, show_plot, n_weights=16)

    print("Norms of the weights after training")

    util.weight_norm_histogram(rbm, show_plot=show_plot)

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
