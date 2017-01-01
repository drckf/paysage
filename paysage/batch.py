import numpy, pandas
    
# ----- CLASSES ----- #
    
class Batch(object):
    
    def __init__(self, filename, key, batch_size, train_fraction=0.9,
                 transform=None, dtype=numpy.float32):
        if transform:
            assert callable(transform)
        self.transform = transform
        
        self.store = pandas.HDFStore(filename, mode='r')
        self.key = key
        self.dtype = dtype
        self.batch_size = batch_size
        self.ncols = self.store.get_storer(key).ncols       
        self.nrows = self.store.get_storer(key).nrows
        self.split = int(numpy.ceil(train_fraction * self.nrows))
        
        self.iterators = {}
        self.iterators['train'] = self.store.select(key, stop=self.split, iterator=True, chunksize=self.batch_size)
        self.iterators['validate'] = self.store.select(key, start=self.split, iterator=True, chunksize=self.batch_size)
        
        self.generators = {mode: self.iterators[mode].__iter__() for mode in self.iterators}
        
    def num_validation_samples(self):
        return self.nrows - self.split
        
    def close(self):
        self.store.close()
        
    def reset_generator(self, mode):
        if mode == 'train':
            self.generators['train'] = self.iterators['train'].__iter__()
        elif mode == 'validate':
            self.generators['validate'] = self.iterators['validate'].__iter__()
        else:
            self.generators = {mode: self.iterators[mode].__iter__() for mode in self.iterators}
            
    def get(self, mode):
        try:
            vals = next(self.generators[mode]).as_matrix()
        except StopIteration:
            self.reset_generator(mode)
            raise StopIteration
        if self.transform:
            return self.transform(vals).astype(self.dtype)
        else:
            return vals.astype(self.dtype)
            
            
#TODO: DataShuffler         
class DataShuffler(object):
    
    def __init__(self, filename, iterations=1, batch_size=1000):
        self.filename = filename
        self.batch_size = batch_size
        self.iterations = iterations
            
           
# ----- FUNCTIONS ----- #

# vectorize('int8(int8)')
def binarize_color(anarray):
    return numpy.round(anarray/255).astype(numpy.float32)

# vectorize('float32(int8)')
def binary_to_ising(anarray):
    return 2.0 * anarray.astype(numpy.float32) - 1.0
    
# vectorize('float32(int8)')
def color_to_ising(anarray):
    return binary_to_ising(binarize_color(anarray))
        