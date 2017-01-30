import os, sys, numpy, pandas, time

from paysage import batch
from paysage.models import hidden
from paysage import fit
from paysage import optimizers

try:
    import plotting
except ImportError:
    from . import plotting

if __name__ == "__main__":

    num_hidden_units = 500
    batch_size = 50
    num_epochs = 20
    learning_rate = 0.001
    mc_steps = 1

    paysage_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    filepath = os.path.join(paysage_path, 'mnist', 'mnist.h5')
    shuffled_filepath = os.path.join(paysage_path, 'mnist', 'shuffled_mnist.h5')

    # shuffle the data
    if not os.path.exists(shuffled_filepath):
        shuffler = batch.DataShuffler(filepath, shuffled_filepath, complevel=0)
        shuffler.shuffle()

    # set up the reader to get minibatches
    data = batch.Batch(shuffled_filepath,
                       'train/images',
                       batch_size,
                       transform=batch.binarize_color,
                       train_fraction=0.99)

    # set up the model and initialize the parameters
    rbm = hidden.RestrictedBoltzmannMachine(data.ncols,
                                            num_hidden_units,
                                            vis_type='bernoulli',
                                            hid_type='bernoulli')
    rbm.initialize(data, method='hinton')

    # set up the optimizer and the fit method
    opt = optimizers.ADAM(rbm,
                          stepsize=learning_rate,
                          lr_decay=0.993333)
    cd = fit.PCD(rbm,
                 data,
                 opt,
                 num_epochs,
                 mc_steps,
                 skip=200,
                 update_method='stochastic',
                 metrics=['ReconstructionError',
                          'EnergyDistance',
                          'EnergyGap',
                          'EnergyZscore'])


    # fit the model
    print('training with contrastive divergence')
    cd.train()

    # evaluate the model
    # this will be the same as the final epoch results
    # it is repeated here to be consistent with the sklearn rbm example
    performance = fit.ProgressMonitor(0,
                                      data,
                                      metrics=['ReconstructionError',
                                               'EnergyDistance',
                                               'EnergyGap',
                                               'EnergyZscore'])
    print('Final performance metrics:')
    performance.check_progress(rbm, 0, show=True)

    print("\nPlot a random sample of reconstructions")
    v_data = data.get('validate')
    sampler = fit.SequentialMC(rbm, v_data)
    sampler.update_state(1)
    v_model = sampler.state

    idx = numpy.random.choice(range(len(v_model)), 5, replace=False)
    grid = numpy.array([[v_data[i], v_model[i]] for i in idx])
    plotting.plot_image_grid(grid, (28,28), vmin=grid.min(), vmax=grid.max())

    print("\nPlot a random sample of fantasy particles")
    random_samples = rbm.random(v_data)
    sampler = fit.SequentialMC(rbm, random_samples)
    sampler.update_state(1000)
    v_model = sampler.state

    idx = numpy.random.choice(range(len(v_model)), 5, replace=False)
    grid = numpy.array([[v_model[i]] for i in idx])
    plotting.plot_image_grid(grid, (28,28), vmin=grid.min(), vmax=grid.max())

    print("\nPlot a random sample of the weights")
    idx = numpy.random.choice(range(rbm.params['weights'].shape[1]), 5, replace=False)
    grid = numpy.array([[rbm.params['weights'][:, i]] for i in idx])
    plotting.plot_image_grid(grid, (28,28), vmin=grid.min(), vmax=grid.max())

    # close the HDF5 store
    data.close()
