from paysage import batch
from paysage import backends as be

import pytest

def test_in_memory_table_batch():
    # create data
    num_rows = 10000
    num_cols = 10
    tensor = be.rand((num_rows, num_cols))

    # batch it with InMemoryTable
    batch_size = 1000
    num_train_batches = num_rows // batch_size
    data = batch.InMemoryTable(tensor, batch_size)

    # loop through, checking the data
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
        assert be.allclose(batch_data,
            tensor[i_batch * batch_size: (i_batch + 1) * batch_size])

        i_batch += 1


def test_in_memory_batch():
    # create data
    num_rows = 10000
    num_cols = 10
    tensor = be.rand((num_rows, num_cols))

    # read it back with Batch
    batch_size = 1000
    num_train_batches = num_rows // batch_size
    with batch.Batch({'train': batch.InMemoryTable(tensor, batch_size),
                      'validate': batch.InMemoryTable(tensor, batch_size)}) as data:

        # loop through thrice, checking the data
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
            assert be.allclose(batch_data_train,
                tensor[i_batch * batch_size: (i_batch + 1) * batch_size])
            assert be.allclose(batch_data_validate,
                tensor[i_batch * batch_size: (i_batch + 1) * batch_size])

            i_batch += 1


if __name__ == "__main__":
    pytest.main([__file__])
