import numpy
from .backends import numba_engine as en

# ----- CLASSES ----- #

class Initializer(object):
    
    def __init__(self, batch):
        self.batch = batch

    def initial_bias(self, inverse_mean):
        p = numpy.ones(self.batch.ncols, dtype=numpy.float32) / self.batch.batch_size
        nbatches = 1
        while True:
            try:
                v_data = self.batch.get(mode='train')
            except StopIteration:
                break
            p += numpy.mean(v_data, axis=0).astype(numpy.float32)
            nbatches += 1
        print(p)
        return inverse_mean(p/nbatches)
