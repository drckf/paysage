import tempfile
import numpy as np
import pandas as pd

from paysage import batch

import pytest

def test_shuffle():
    # create temporary files
    file_original = tempfile.NamedTemporaryFile()
    file_shuffle = tempfile.NamedTemporaryFile()

    # create data
    num_rows = 10000
    num_cols_A = 100
    num_cols_B = 1
    df_A = pd.DataFrame(np.arange(num_rows*num_cols_A).reshape(num_rows, num_cols_A),
                        columns=['col_{}'.format(i) for i in np.arange(num_cols_A)],
                        index=['ix_{}'.format(i) for i in np.arange(num_rows)])
    df_B = pd.DataFrame(np.arange(num_rows*num_cols_B).reshape(num_rows, num_cols_B),
                        columns=['col_{}'.format(i) for i in np.arange(num_cols_B)],
                        index=['ix_{}'.format(i) for i in np.arange(num_rows)])

    # save it
    store = pd.HDFStore(file_original.name, mode='w')
    store.append("A", df_A)
    store.append("B", df_B)
    store.close()

    # shuffle it, with an artificially low memory limit
    shuffler = batch.DataShuffler(file_original.name, file_shuffle.name,
                                  allowed_mem=0.001)
    shuffler.shuffle()

    # read the shuffled data
    df_As = pd.read_hdf(file_shuffle.name, "A")
    df_Bs = pd.read_hdf(file_shuffle.name, "B")

    # check the two shuffles are consistent
    assert (df_As.index == df_Bs.index).all()
    assert (df_As['col_0'] // num_cols_A == df_Bs['col_0'] // num_cols_B).all()

    # check that the shuffles preserve the index
    ix_A_orig = sorted(list(df_A.index))
    ix_A_shuffled = sorted(list(df_As.index))
    assert ix_A_orig == ix_A_shuffled

    # check a couple of statistics
    vals_B = df_B['col_0'].values
    vals_Bs = df_Bs['col_0'].values
    # the number of fixed points tends to a Poisson distribution with e.v. = 1
    assert (vals_B == vals_Bs).sum() < 5
    # the difference between values (using the natural numbers as values)
    # is a triangular distribution centered at 0. Can check the variance.
    diff_dist_std = (vals_B - vals_Bs).std()
    assert np.abs(diff_dist_std / (num_rows / np.sqrt(6)) - 1) < 0.05


if __name__ == "__main__":
    pytest.main([__file__])
