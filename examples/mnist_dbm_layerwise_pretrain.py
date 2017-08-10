from paysage import batch
from paysage import preprocess as pre
from paysage import layers
from paysage.models import model
from paysage import fit
from paysage import optimizers
from paysage import backends as be
from paysage import schedules
from paysage import penalties as pen

be.set_seed(137) # for determinism

import example_util as util

def run(paysage_path=None, num_epochs=10, show_plot=False):
    num_hidden_units = 100
    batch_size = 100
    learning_rate = schedules.PowerLawDecay(initial=0.012, coefficient=0.1)
    mc_steps = 1

    (_, _, shuffled_filepath) = \
            util.default_paths(paysage_path)

    # set up the reader to get minibatches
    data = batch.HDFBatch(shuffled_filepath,
                          'train/images',
                          batch_size,
                          transform=pre.binarize_color,
                          train_fraction=0.95)

    # set up the model and initialize the parameters
    vis_layer = layers.BernoulliLayer(data.ncols)
    hid_1_layer = layers.BernoulliLayer(num_hidden_units)
    hid_2_layer = layers.BernoulliLayer(num_hidden_units)
    hid_3_layer = layers.BernoulliLayer(num_hidden_units)

    rbm = model.Model([vis_layer, hid_1_layer, hid_2_layer, hid_3_layer])
    rbm.initialize(data, method='glorot_normal')

    print("Norms of the weights before training")
    util.weight_norm_histogram(rbm, show_plot=show_plot)

    # small penalties prevent the weights from consolidating
    rbm.weights[1].add_penalty({'matrix': pen.logdet_penalty(0.001)})
    rbm.weights[2].add_penalty({'matrix': pen.logdet_penalty(0.001)})

    metrics = ['ReconstructionError', 'EnergyDistance', 'EnergyGap',
               'EnergyZscore', 'HeatCapacity', 'WeightSparsity', 'WeightSquare']
    perf = fit.ProgressMonitor(data, metrics=metrics)

    # set up the optimizer and the fit method
    opt = optimizers.ADAM(stepsize=learning_rate)
    cd = fit.LayerwisePretrain(rbm, data, opt, num_epochs, method=fit.pcd,
                 mcsteps=mc_steps, metrics=metrics)

    # fit the model
    print('training with persistent contrastive divergence')
    cd.train()

    # evaluate the model
    util.show_metrics(rbm, perf)
    valid = data.get('validate')
    util.show_reconstructions(rbm, valid, fit, show_plot, num_to_avg=10)
    util.show_fantasy_particles(rbm, valid, fit, show_plot)
    from math import sqrt
    dim = tuple([28] + [int(sqrt(num_hidden_units)) for _ in range(rbm.num_weights)])
    util.show_weights(rbm, show_plot, dim=dim, n_weights=16)
    util.show_one_hot_reconstructions(rbm, fit, dim=28, n_recon=16, num_to_avg=1)

    print("Norms of the weights after training")
    util.weight_norm_histogram(rbm, show_plot=show_plot)

    # close the HDF5 store
    data.close()
    print("Done")

if __name__ == "__main__":
    run(show_plot = True)
