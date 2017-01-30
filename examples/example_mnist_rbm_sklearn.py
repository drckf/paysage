import os, sys, numpy, pandas, time

from paysage import backends as B
from paysage import batch
from paysage import metrics as M
from paysage.layers import BernoulliLayer
from sklearn.neural_network import BernoulliRBM

try:
    import plotting
except ImportError:
    from . import plotting

if __name__ == "__main__":

    num_hidden_units = 500
    batch_size = 50
    num_epochs = 10
    # the step size has been hand-tuned for the sklearn implementation
    learning_rate = 0.01

    paysage_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    filepath = os.path.join(paysage_path, 'mnist', 'mnist.h5')
    shuffled_filepath = os.path.join(paysage_path, 'mnist', 'shuffled_mnist.h5')

    # shuffle the data
    if not os.path.exists(shuffled_filepath):
        shuffler = batch.DataShuffler(filepath, shuffled_filepath, complevel=0)
        shuffler.shuffle()

    # set up the reader to get the whole training set
    data = batch.Batch(shuffled_filepath,
                       'train/images',
                       60000,
                       transform=batch.binarize_color,
                       train_fraction=0.99)

    X = data.get('train')
    data.close()

    rbm = BernoulliRBM(n_components=num_hidden_units,
                       learning_rate=learning_rate,
                       batch_size=batch_size,
                       n_iter=num_epochs,
                       verbose=1)
    rbm.fit(X)

    # set up the reader to read the first batch of the validation set
    data = batch.Batch(shuffled_filepath,
                   'train/images',
                   batch_size,
                   transform=batch.binarize_color,
                   train_fraction=0.99)

    # compute some metrics to evaluate the model
    metrics=['ReconstructionError',
             'EnergyDistance'
             ]

    metrics = [M.__getattribute__(m)() for m in
                                    ['ReconstructionError',
                                     'EnergyDistance'
                                     ]]
    while True:
        try:
            v_data = data.get(mode='validate')
        except StopIteration:
            break

        reconstructions = rbm.gibbs(v_data)

        random_samples = BernoulliLayer().random(v_data).astype(numpy.float32)
        fantasy_particles = random_samples.astype(numpy.float32)
        for t in range(10):
            fantasy_particles = rbm.gibbs(fantasy_particles)

        argdict = {
        'minibatch': v_data.astype(numpy.float32),
        'reconstructions': reconstructions.astype(numpy.float32),
        'random_samples': random_samples.astype(numpy.float32),
        'samples': fantasy_particles.astype(numpy.float32),
        'amodel': rbm
        }

        for m in metrics:
            m.update(**argdict)

    print('\nFinal performance metrics:')
    metdict = {m.name: m.value() for m in metrics}
    for m in metdict:
        print("-{0}: {1:.6f}".format(m, metdict[m]))

    print("\nPlot a random sample of reconstructions")
    v_data = data.get('validate')
    v_model = rbm.gibbs(v_data)

    idx = numpy.random.choice(range(len(v_model)), 5, replace=False)
    grid = numpy.array([[v_data[i], v_model[i]] for i in idx])
    plotting.plot_image_grid(grid, (28,28), vmin=grid.min(), vmax=grid.max())

    print("\nPlot a random sample of fantasy particles")
    random_samples = BernoulliLayer().random(v_data).astype(numpy.float32)
    v_model = random_samples.astype(numpy.float32)
    for t in range(1000):
            v_model = rbm.gibbs(v_model)

    idx = numpy.random.choice(range(len(v_model)), 5, replace=False)
    grid = numpy.array([[v_model[i]] for i in idx])
    plotting.plot_image_grid(grid, (28,28), vmin=grid.min(), vmax=grid.max())

    print("\nPlot a random sample of the weights")
    W = rbm.components_.T
    idx = numpy.random.choice(range(W.shape[1]), 5, replace=False)
    grid = numpy.array([[W[:, i]] for i in idx])
    plotting.plot_image_grid(grid, (28,28), vmin=grid.min(), vmax=grid.max())

    # close the HDF5 store
    data.close()
