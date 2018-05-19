import tempfile
import numpy as np
import pandas as pd

from paysage import batch
from paysage import backends as be

import pytest

def test_hdf_table_batch():
    # the temporary storage file
    store_file = tempfile.NamedTemporaryFile()

    # create data
    num_rows = 10000
    num_cols = 10
    df_A = pd.DataFrame(np.arange(num_rows*num_cols).reshape(num_rows, num_cols))

    # save it
    with pd.HDFStore(store_file.name, mode="w", format="table") as store:
        store.append("train", df_A)

    # read it back with the HDFtable
    batch_size = 1000
    num_train_batches = num_rows // batch_size
    data = batch.HDFtable(store_file.name, "train", batch_size)

    # loop through thrice, checking the data
    for i_loop in range(3):
        i_batch = 0
        while True:
            # get the data
            try:
                batch_data = data.get()
            except StopIteration:
                assert i_batch == num_train_batches
                i_batch = 0
                break

            # check it
            assert np.all(be.to_numpy_array(batch_data) == \
                df_A.values[i_batch * batch_size: (i_batch + 1) * batch_size])

            i_batch += 1


def test_hdf_batch():
    # the temporary storage file
    store_file = tempfile.NamedTemporaryFile()

    # create data
    num_rows = 10000
    num_cols = 10
    df_A = pd.DataFrame(np.arange(num_rows*num_cols).reshape(num_rows, num_cols))
    df_B = df_A + num_rows*num_cols

    # save it
    with pd.HDFStore(store_file.name, mode="w", format="table") as store:
        store.append("train", df_A)
        store.append("validate", df_B)

    # read it back with the HDFtable
    batch_size = 1000
    num_train_batches = num_rows // batch_size
    data = batch.Batch(
            {"train": batch.HDFtable(store_file.name, "train", batch_size),
             "validate": batch.HDFtable(store_file.name, "validate", batch_size)})

    # loop through thrice, checking the data
    for i_loop in range(3):
        i_batch = 0
        while True:
            # get the data
            try:
                batch_data_train = data.get("train")
                batch_data_validate = data.get("validate")
            except StopIteration:
                assert i_batch == num_train_batches
                i_batch = 0
                data.reset_generator("all")
                break

            # check it
            assert np.all(be.to_numpy_array(batch_data_train) == \
                df_A.values[i_batch * batch_size: (i_batch + 1) * batch_size])
            assert np.all(be.to_numpy_array(batch_data_validate) == \
                df_B.values[i_batch * batch_size: (i_batch + 1) * batch_size])

            i_batch += 1


if __name__ == "__main__":
    pytest.main([__file__])
