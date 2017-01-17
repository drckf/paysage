import os, sys, numpy, pandas, time

from paysage import batch
from paysage.models import hidden
from paysage import fit
from paysage import optimizers
from paysage import backends as B

import matplotlib.pyplot as plt
import seaborn as sns

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
    num_epochs = 20
    learning_rate = 0.001

    # set up the batch object to read the data
    filepath = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'mnist', 'mnist.h5')
    data = batch.Batch(filepath,
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
    opt = optimizers.RMSProp(rbm, stepsize=learning_rate)
    cd = fit.PCD(rbm, data, opt, num_epochs, 1, skip=200, update_method='stochastic')


    # fit the model
    print('training with contrastive divergence')
    cd.train()

    # plot some reconstructions
    v_data = data.get('validate')
    sampler = fit.SequentialMC(rbm, v_data)
    sampler.update_state(1, resample=False, temperature=1.0)
    v_model = sampler.state

    recon = numpy.sqrt(numpy.sum((v_data - v_model)**2) / len(v_data))

    plot_image(v_data[0], (28,28))
    plot_image(v_model[0], (28,28))

    # plot some fantasy particles
    sampler.update_state(1000, resample=False, temperature=1.0)
    v_model = sampler.state

    plot_image(v_data[0], (28,28))
    plot_image(v_model[0], (28,28))

    edist = B.fast_energy_distance(v_data.astype(numpy.float32), v_model.astype(numpy.float32), downsample=100)

    print('Reconstruction error:  {0:.2f}'.format(recon))
    print('Energy distance:  {0:.2f}'.format(edist))

    # close the HDF5 store
    data.close()

    end = time.time()
    print('Total time: {0:.2f} seconds'.format(end - start))
