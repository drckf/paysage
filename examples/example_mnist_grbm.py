import os, sys, numpy, pandas, time

#from paysage.backends import numba_engine as en
from paysage import batch
from paysage.models import hidden
from paysage import fit
from paysage import optimizers
from paysage import backends as B

import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import seaborn as sns

import plotting

def plot_image(image_vector, shape):
    f, ax = plt.subplots(figsize=(4,4))
    hm = sns.heatmap(numpy.reshape(image_vector, shape), ax=ax, cmap="gray_r", cbar=False)
    hm.set(yticks=[])
    hm.set(xticks=[])
    plt.show(f)
    plt.close(f)

if __name__ == "__main__":
    start = time.time()

    num_hidden_units = 500
    batch_size = 50
    num_epochs = 10
    learning_rate = 0.0001
    mc_steps = 1

    def transform(x):
        return numpy.float32(x) / 255

    filepath = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'mnist', 'mnist.h5')
    shuffled_filepath = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'mnist', 'shuffled_mnist.h5')

    # shuffle the data
    shuffler = batch.DataShuffler(filepath, shuffled_filepath, complevel=0)
    shuffler.shuffle()

    # set up the reader to get minibatches
    data = batch.Batch(shuffled_filepath,
                       'train/images',
                       batch_size,
                       transform=transform,
                       train_fraction=0.99)

    # set up the model and initialize the parameters
    rbm = hidden.GRBM(data.ncols,
                      num_hidden_units,
                      hid_type = 'bernoulli')
    rbm.initialize(data, method='hinton')

    # set up the optimizer and the fit method
    opt = optimizers.RMSProp(rbm, stepsize=learning_rate)
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

    # plot some reconstructions
    v_data = data.get('validate')
    sampler = fit.SequentialMC(rbm, v_data)
    sampler.update_state(1, resample=False, temperature=1.0)
    v_model = sampler.state

    idx = numpy.random.choice(range(len(v_model)), 5, replace=False)
    grid = numpy.array([[v_data[i], v_model[i]] for i in idx])
    plot_image(v_data[0], (28,28))
    plot_image(v_model[0], (28,28))

    # plot some fantasy particles
    random_samples = rbm.random(v_data)
    sampler = fit.SequentialMC(rbm, random_samples)
    sampler.update_state(1000, resample=False, temperature=1.0)
    v_model = sampler.state

    idx = numpy.random.choice(range(len(v_model)), 5, replace=False)
    plot_image(v_model[0], (28,28))

    # close the HDF5 store
    data.close()

    end = time.time()
    print('Total time: {0:.2f} seconds'.format(end - start))
