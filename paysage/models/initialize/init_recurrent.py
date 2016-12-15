import numpy
from ...backends import numba_engine as en

# ----- CLASSES ----- #

class Initializer(object):
    
    def __init__(self, batch):
        self.batch = batch

    def initial_bias(self, inverse_mean):
        p = numpy.zeros(self.batch.ncols, dtype=numpy.float32)
        nbatches = 0
        while True:
            try:
                v_data = self.batch.get(mode='train')
            except StopIteration:
                break
            p += numpy.mean(v_data, axis=0).astype(numpy.float32)
            nbatches += 1
        return inverse_mean(p/nbatches)

    
