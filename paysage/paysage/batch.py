import os, sys, numpy, pandas

# ---- FUNCTIONS ----- #

# binarize_color: (np.ndarray) -> np.ndarray
def binarize_color(anarray):
    return (anarray / 255).astype(numpy.int8)

# binary_to_ising: (np.ndarray) -> np.ndarray
def binary_to_ising(anarray):
    return 2 * anarray -1
    
# color_to_ising: (np.ndarray) -> np.ndarray
def color_to_ising(anarray):
    return binary_to_ising(binarize_color(anarray))
    
    
# ---- CLASSES ----- #

class IndexBatch(object):
    
    def __init__(self, nrows, batch_size):
        self.start = 0
        self.end = nrows
        self.batch_size = batch_size
        
    @classmethod
    def from_store(cls, store, key, batch_size):
        return cls(store.get_storer(key).nrows, batch_size)
        
    def reset(self):
        self.start = 0
        
    def get(self):
        if self.start >= self.end:
            self.reset()
            raise StopIteration
        else:
            new_end = min(self.start + self.batch_size, self.end)
            indices = (self.start, new_end)
            self.start += self.batch_size
            return indices
        

class Batch(object):
    
    def __init__(self, filename, key, batch_size, transform=None):
        if transform:
            assert callable(transform)
        self.store = pandas.HDFStore(filename, mode='r')
        self.key = key
        self.index = IndexBatch.from_store(self.store, key, batch_size) 
        self.cols = self.store.get_storer(key).ncols       
        self.transform = transform
        
    def reset(self):
        self.index.reset()
            
    def get(self):
        start, stop = self.index.get()
        tmp = self.store.select(self.key, start=start, stop=stop).as_matrix()
        if not self.transform:
            return tmp
        else:
            return self.transform(tmp)
        