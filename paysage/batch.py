import numpy, pandas    
    
# ----- CLASSES ----- #
    

class IndexBatch(object):
    """IndexBatch
       Dishes out batches of start/stop positions for reading minibatches of data from an HDFStore. 
       The validation set is taken as the last (1 - train_fraction) samples in the store. 
       
    """
    def __init__(self, nrows, batch_size, train_fraction=0.9):
        self.nrows = nrows
        self.end = int(numpy.ceil(train_fraction * nrows))
        self.batch_size = batch_size
        self.create_iterators()
        
    @classmethod
    def from_store(cls, store, key, batch_size, train_fraction=0.9):
        return cls(store.get_storer(key).nrows, batch_size, train_fraction=train_fraction)

    def create_iterators(self):
        self.iterators = {}
        self.iterators['train'] = inclusive_slice(0, self.end, self.batch_size)
        self.iterators['validate'] = inclusive_slice(self.end, self.nrows, self.batch_size)
        
    def reset(self, mode='train'):
        if mode == 'train':
            self.iterators['train'] = inclusive_slice(0, self.end, self.batch_size)
        else:
            self.iterators['validate'] = inclusive_slice(self.end, self.nrows, self.batch_size)
        
    def get(self, mode='train'):
        try:
            next_iter = next(self.iterators[mode])
        except StopIteration:
            self.reset(mode)
            raise StopIteration
        return next_iter
        

class Batch(object):
    """Batch
       Serves up minibatches from an HDFStore. 
       The validation set is taken as the last (1 - train_fraction) samples in the store. 
       The data should probably be randomly shuffled if being used to train a non-recurrent model.
    
    """    
    def __init__(self, filename, key, batch_size, train_fraction=0.9, memory = 1,
                 transform=None, flatten=False, dtype=numpy.float32):
        if transform:
            assert callable(transform)
        self.store = pandas.HDFStore(filename, mode='r')
        self.key = key
        self.ncols = self.store.get_storer(key).ncols       
        self.nrows = self.store.get_storer(key).nrows
        self.batch_size = batch_size
        self.memory = int(memory * 10 ** 9)
        self.chunk_size = chunksize(self.ncols, dtype(1).itemsize, chunk=self.memory)
        self.index = IndexBatch.from_store(self.store, self.key, self.chunk_size, train_fraction=train_fraction) 
        self.dtype = dtype
        self.transform = transform
        self.flatten = flatten
        self.chunk = {'train': numpy.array([], dtype=self.dtype), 'validate': numpy.array([], dtype=self.dtype)}
        self.batch_iterator = {'train': None, 'validate': None}
        self.read_chunk('train')
        self.read_chunk('validate')
        
    def read_chunk(self, mode='train'):
        start, stop = self.index.get(mode)
        tmp = self.store.select(self.key, start=start, stop=stop).as_matrix().astype(self.dtype)
        if self.flatten:
            tmp = tmp.reshape((len(tmp),-1))
        if not self.transform:
            self.chunk[mode] = tmp
        else:
            self.chunk[mode] = self.transform(tmp)
        self.batch_iterator[mode] = inclusive_slice(0, len(self.chunk[mode]), self.batch_size)
        
    def reset(self, mode='train'):
        if mode == 'all':
            self.index.reset('train')
            self.index.reset('validate')
        else:
            self.index.reset(mode)
            
    def get(self, mode='train'):
        try:
            start, stop = next(self.batch_iterator[mode])
        except StopIteration:
            self.read_chunk(mode)
            start, stop = next(self.batch_iterator[mode])
        return self.chunk[mode][start:stop]
            
    def close(self):
        self.store.close()
            
            
#TODO: DataShuffler         
class DataShuffler(object):
    
    def __init__(self, filename, iterations=1, batch_size=1000):
        self.filename = filename
        self.batch_size = batch_size
        self.iterations = iterations
            
            
# ----- FUNCTIONS ----- #
            
def inclusive_slice(start, stop, step):
    current = start
    while current < stop:
        next_iter = min(stop, current + step)
        result = (current, next_iter)
        current = next_iter
        yield result
        
def chunksize(ncols, nbytes, chunk=10**7):
    bytes_per_row = ncols * nbytes
    return chunk // bytes_per_row

# vectorize('int8(int8)')
def binarize_color(anarray):
    return numpy.round(anarray/255).astype(numpy.float32)

# vectorize('float32(int8)')
def binary_to_ising(anarray):
    return 2.0 * anarray.astype(numpy.float32) - 1.0
    
# vectorize('float32(int8)')
def color_to_ising(anarray):
    return binary_to_ising(binarize_color(anarray))
        