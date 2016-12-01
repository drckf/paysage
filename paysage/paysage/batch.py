import os, sys, numpy, pandas

# ---- FUNCTIONS ----- #

# binarize_color: (np.ndarray) -> np.ndarray
def binarize_color(anarray):
    return (anarray/255).astype(numpy.int8)

# binary_to_ising: (np.ndarray) -> np.ndarray
def binary_to_ising(anarray):
    return 2 * anarray - 1
    
# color_to_ising: (np.ndarray) -> np.ndarray
def color_to_ising(anarray):
    return binary_to_ising(binarize_color(anarray)).astype(numpy.float32)
    
    
# ---- CLASSES ----- #

class IndexBatch(object):
    
    def __init__(self, nrows, batch_size, train_fraction=0.9):
        self.start = {}
        self.end = {}
        self.start['train'] = 0
        self.end['train'] = int(numpy.ceil(0.9*nrows))
        self.start['validate'] = self.end['train']
        self.end['validate'] = nrows
        self.batch_size = batch_size
        
    @classmethod
    def from_store(cls, store, key, batch_size, train_fraction=0.9):
        return cls(store.get_storer(key).nrows, batch_size, train_fraction=train_fraction)
        
    def reset(self, mode='train'):
        self.start[mode] = 0
        
    def get(self, mode='train'):
        if self.start[mode] >= self.end[mode]:
            self.reset(mode=mode)
            raise StopIteration
        else:
            new_end = min(self.start[mode] + self.batch_size, self.end[mode])
            indices = (self.start[mode], new_end)
            self.start[mode] += self.batch_size
            return indices
        

class Batch(object):
    
    def __init__(self, filename, key, batch_size, train_fraction=0.9,
                 transform=None, flatten=False, dtype=numpy.float32):
        if transform:
            assert callable(transform)
        self.store = pandas.HDFStore(filename, mode='r')
        self.key = key
        self.index = IndexBatch.from_store(self.store, key, batch_size, train_fraction=train_fraction) 
        self.cols = self.store.get_storer(key).ncols       
        self.transform = transform
        self.dtype = dtype
        self.flatten = flatten
        
    def reset(self, mode='train'):
        self.index.reset()
            
    def get(self, mode='train'):
        start, stop = self.index.get(mode=mode)
        tmp = self.store.select(self.key, start=start, stop=stop).as_matrix().astype(self.dtype)
        if self.flatten:
            tmp = tmp.reshape((len(tmp),-1))
        if not self.transform:
            return tmp
        else:
            return self.transform(tmp)
        