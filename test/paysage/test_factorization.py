import numpy as np
import pandas as pd
import tempfile

from paysage import factorization
from paysage import backends as be
from paysage import batch

import pytest


def test_pca():
    # create some random data
    num_samples = 10000
    dim = 10
    batch_size = 100
    num_components = 3

    # generate some data
    mean = np.random.random(dim)
    cov_factor = np.random.random((dim, dim))
    cov = np.dot(cov_factor, cov_factor.T)
    samples = be.float_tensor(np.random.multivariate_normal(mean, cov, size=num_samples))

    samples_train, samples_validate = batch.split_tensor(samples, 0.9)
    data = batch.Batch({'train': batch.InMemoryTable(samples_train, batch_size),
                        'validate': batch.InMemoryTable(samples_validate, batch_size)})

    # find the principal directions
    pca = factorization.PCA.from_batch(data, num_components, epochs=10,
                                       grad_steps_per_minibatch=1, stepsize=0.01)

    assert be.shape(pca.W) == (dim, num_components)
    assert be.shape(pca.var) == (num_components,)


def test_pca_svd():
    # create some random data
    num_samples = 10000
    dim = 10
    num_components = 3

    # generate some data
    mean = np.random.random(dim)
    cov_factor = np.random.random((dim, dim))
    cov = np.dot(cov_factor, cov_factor.T)
    samples = be.float_tensor(np.random.multivariate_normal(mean, cov, size=num_samples))

    # find the principal directions
    pca = factorization.PCA.from_svd(samples, num_components)

    assert be.shape(pca.W) == (dim, num_components)
    assert be.shape(pca.var) == (num_components,)


def test_pca_compare_var():
    # create some random data
    num_samples = 10000
    dim = 10
    batch_size = 100
    num_components = 3

    # generate some data
    mean = np.random.random(dim)
    cov_factor = np.random.random((dim, dim))
    cov = np.dot(cov_factor, cov_factor.T)
    samples = be.float_tensor(np.random.multivariate_normal(mean, cov, size=num_samples))

    samples_train, samples_validate = batch.split_tensor(samples, 0.9)
    data = batch.Batch({'train': batch.InMemoryTable(samples_train, batch_size),
                        'validate': batch.InMemoryTable(samples_validate, batch_size)})

    # find the principal directions
    pca_sgd = factorization.PCA.from_batch(data, num_components, epochs=10,
                                           grad_steps_per_minibatch=1, stepsize=0.01)
    pca_svd = factorization.PCA.from_svd(samples_train, num_components)

    assert be.norm(pca_sgd.var - pca_svd.var) / be.norm(pca_sgd.var) < 1e-1


def test_pca_save_read():
    # create some random data
    num_samples = 10000
    dim = 10
    batch_size = 100
    num_components = 3

    # generate some data
    mean = np.random.random(dim)
    cov_factor = np.random.random((dim, dim))
    cov = np.dot(cov_factor, cov_factor.T)
    samples = be.float_tensor(np.random.multivariate_normal(mean, cov, size=num_samples))

    samples_train, samples_validate = batch.split_tensor(samples, 0.9)
    data = batch.Batch({'train': batch.InMemoryTable(samples_train, batch_size),
                        'validate': batch.InMemoryTable(samples_validate, batch_size)})

    # find the principal directions
    pca = factorization.PCA.from_batch(data, num_components, epochs=10,
                                       grad_steps_per_minibatch=1, stepsize=0.01)

    # save it
    pca_file = tempfile.NamedTemporaryFile()
    store = pd.HDFStore(pca_file.name, mode="w")
    pca.save(store)

    # read it
    pca_read = factorization.PCA.from_saved(store)
    store.close()

    # check it
    assert be.allclose(pca.W, pca_read.W)
    assert be.allclose(pca.var, pca_read.var)
    assert pca.stepsize == pca_read.stepsize
    assert pca.num_components == pca_read.num_components


def test_pca_svd_save_read():
    # create some random data
    num_samples = 10000
    dim = 10
    num_components = 3

    # generate some data
    mean = np.random.random(dim)
    cov_factor = np.random.random((dim, dim))
    cov = np.dot(cov_factor, cov_factor.T)
    samples = be.float_tensor(np.random.multivariate_normal(mean, cov, size=num_samples))

    # find the principal directions
    pca = factorization.PCA.from_svd(samples, num_components)

    # save it
    pca_file = tempfile.NamedTemporaryFile()
    store = pd.HDFStore(pca_file.name, mode="w")
    pca.save(store)

    # read it
    pca_read = factorization.PCA.from_saved(store)
    store.close()

    # check it
    assert be.allclose(pca.W, pca_read.W)
    assert be.allclose(pca.var, pca_read.var)
    assert pca.stepsize == pca_read.stepsize
    assert pca.num_components == pca_read.num_components


def test_pca_save_read_num_components():
    # create some random data
    num_samples = 10000
    dim = 10
    batch_size = 100
    num_components = 3
    num_components_save = 2

    # generate some data
    mean = np.random.random(dim)
    cov_factor = np.random.random((dim, dim))
    cov = np.dot(cov_factor, cov_factor.T)
    samples = be.float_tensor(np.random.multivariate_normal(mean, cov, size=num_samples))

    samples_train, samples_validate = batch.split_tensor(samples, 0.9)
    data = batch.Batch({'train': batch.InMemoryTable(samples_train, batch_size),
                        'validate': batch.InMemoryTable(samples_validate, batch_size)})

    # find the principal directions
    pca = factorization.PCA.from_batch(data, num_components, epochs=10,
                                       grad_steps_per_minibatch=1, stepsize=0.01)

    # save it
    pca_file = tempfile.NamedTemporaryFile()
    store = pd.HDFStore(pca_file.name, mode="w")
    pca.save(store, num_components_save=num_components_save)

    # read it
    pca_read = factorization.PCA.from_saved(store)
    store.close()

    # check it
    assert be.allclose(pca.W[:,:num_components_save], pca_read.W)
    assert be.allclose(pca.var[:num_components_save], pca_read.var)
    assert pca.stepsize == pca_read.stepsize
    assert pca_read.num_components == num_components_save


if __name__ == "__main__":
    pytest.main([__file__])
