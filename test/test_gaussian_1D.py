import numpy

from paysage import backends as be
from paysage import batch
from paysage import layers
from paysage.models import BoltzmannMachine
from paysage import fit
from paysage import samplers
from paysage import optimizers
from paysage import schedules

import pytest

def test_gaussian_1D_1mode_train():
    # create some example data
    num = 10000
    mu = 3
    sigma = 1
    samples = be.randn((num, 1)) * sigma + mu

    # set up the reader to get minibatches
    batch_size = 100
    samples_train, samples_validate = batch.split_tensor(samples, 0.9)
    data = batch.Batch({'train': batch.InMemoryTable(samples_train, batch_size),
                        'validate': batch.InMemoryTable(samples_validate, batch_size)})

    # parameters
    learning_rate = schedules.PowerLawDecay(initial=0.1, coefficient=0.1)
    mc_steps = 1
    num_epochs = 10
    num_sample_steps = 100

    # set up the model and initialize the parameters
    vis_layer = layers.GaussianLayer(1)
    hid_layer = layers.OneHotLayer(1)

    rbm = BoltzmannMachine([vis_layer, hid_layer])
    rbm.initialize(data, method='hinton')

    # modify the parameters to shift the initialized model from the data
    # this forces it to train
    rbm.layers[0].params = layers.ParamsGaussian(rbm.layers[0].params.loc - 3,
                                                 rbm.layers[0].params.log_var - 1)

    # set up the optimizer and the fit method
    opt = optimizers.ADAM(stepsize=learning_rate)
    cd = fit.SGD(rbm, data)

    # fit the model
    print('training with persistent contrastive divergence')
    cd.train(opt, num_epochs, method=fit.pcd, mcsteps=mc_steps)

    # sample data from the trained model
    model_state = \
        samplers.SequentialMC.generate_fantasy_state(rbm, num, num_sample_steps)
    pts_trained = model_state[0]

    percent_error = 10
    mu_trained = be.mean(pts_trained)
    assert numpy.abs(mu_trained / mu - 1) < (percent_error/100)

    sigma_trained = numpy.sqrt(be.var(pts_trained))
    assert numpy.abs(sigma_trained / sigma - 1) < (percent_error/100)

def test_gaussian_1D_1mode_stationary():
    # create some example data
    num = 10000
    mu = 3
    sigma = 1
    samples = be.randn((num, 1)) * sigma + mu

    # set up the reader to get minibatches
    batch_size = 100
    samples_train, samples_validate = batch.split_tensor(samples, 0.9)
    data = batch.Batch({'train': batch.InMemoryTable(samples_train, batch_size),
                        'validate': batch.InMemoryTable(samples_validate, batch_size)})

    # parameters
    learning_rate = schedules.PowerLawDecay(initial=0.1, coefficient=0.1)
    mc_steps = 1
    num_epochs = 10
    num_sample_steps = 100

    # set up the model and initialize the parameters
    vis_layer = layers.GaussianLayer(1)
    hid_layer = layers.OneHotLayer(1)

    rbm = BoltzmannMachine([vis_layer, hid_layer])
    # keep the parameters as initialized.
    # the model should not change much in training.
    rbm.initialize(data, method='hinton')

    # set up the optimizer and the fit method
    opt = optimizers.ADAM(stepsize=learning_rate)
    cd = fit.SGD(rbm, data)

    # fit the model
    print('training with persistent contrastive divergence')
    cd.train(opt, num_epochs, method=fit.pcd, mcsteps=mc_steps)

    # sample data from the trained model
    model_state = samplers.SequentialMC.generate_fantasy_state(rbm, num,
                                                               num_sample_steps)
    pts_trained = model_state[0]

    percent_error = 10
    mu_trained = be.mean(pts_trained)
    assert numpy.abs(mu_trained / mu - 1) < (percent_error/100)

    sigma_trained = numpy.sqrt(be.var(pts_trained))
    assert numpy.abs(sigma_trained / sigma - 1) < (percent_error/100)

if __name__ == "__main__":
    pytest.main([__file__])
