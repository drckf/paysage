import numpy, pandas    
    
# ---- CLASSES ----- #

class IndexBatch(object):
    """IndexBatch
       Dishes out batches of start/stop positions for reading minibatches of data from an HDFStore. 
       The validation set is taken as the last (1 - train_fraction) samples in the store. 
       
    """
    def __init__(self, nrows, batch_size, train_fraction=0.9):
        self.start = {}
        self.end = {}
        self.start['train'] = 0
        self.end['train'] = int(numpy.ceil(train_fraction * nrows))
        self.start['validate'] = self.end['train']
        self.end['validate'] = nrows
        self.batch_size = batch_size
        
    @classmethod
    def from_store(cls, store, key, batch_size, train_fraction=0.9):
        return cls(store.get_storer(key).nrows, batch_size, train_fraction=train_fraction)
        
    def reset(self, mode='train'):
        if mode == 'train':
            self.start['train'] = 0
        else:
            self.start['validate'] = self.end['train']
        
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
    """Batch
       Serves up minibatches from an HDFStore. 
       The validation set is taken as the last (1 - train_fraction) samples in the store. 
       The data should probably be randomly shuffled if being used to train a non-recurrent model.
    
    """    
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
        if mode == 'all':
            self.index.reset('train')
            self.index.reset('validate')
        else:
            self.index.reset(mode)
            
    def get(self, mode='train'):
        start, stop = self.index.get(mode=mode)
        tmp = self.store.select(self.key, start=start, stop=stop).as_matrix().astype(self.dtype)
        if self.flatten:
            tmp = tmp.reshape((len(tmp),-1))
        if not self.transform:
            return tmp
        else:
            return self.transform(tmp)
            
    def close(self):
        self.store.close()
            
            
#TODO: DataShuffler         
class DataShuffler(object):
    
    def __init__(self, filename, iterations=1, batch_size=1000):
        self.filename = filename
        self.batch_size = batch_size
        self.iterations = iterations
            
            
# ---- FUNCTIONS ----- #
            
def shuffled_index(length, batch_size, train_end):
    index = numpy.random.permutation(numpy.arange(length))
    train = index[:train_end]
    validate = index[train_end:]
    train = [numpy.sort(train[b * batch_size : (b + 1) * batch_size]) for b in range(int(numpy.ceil(len(train)/batch_size)))]
    validate = [numpy.sort(validate[b * batch_size : (b + 1) * batch_size]) for b in range(int(numpy.ceil(len(validate)/batch_size)))]
    return train, validate

# vectorize('int8(int8)')
def binarize_color(anarray):
    return (anarray/255).astype(numpy.int8)

# vectorize('float32(int8)')
def binary_to_ising(anarray):
    return 2.0 * anarray.astype(numpy.float32) - 1.0
    
# vectorize('float32(int8)')
def color_to_ising(anarray):
    return binary_to_ising(binarize_color(anarray))
        