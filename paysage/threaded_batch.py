import numpy, pandas, threading, queue
    
# ----- CLASSES ----- #
    
class AsynchronousReader(threading.Thread):
    
    def __init__(self, filename, key, batch_size, batch_queue, 
                 transform=None, dtype=numpy.float32, beginning=None, end=None):
        super().__init__()
        self.store = pandas.HDFStore(filename, mode='r')
        self.key = key
        self.ncols = self.store.get_storer(key).ncols       
        self.nrows = self.store.get_storer(key).nrows
        self.dtype = dtype
        self.batch_size = batch_size
        self.queue = batch_queue
        self.transform = transform
        self.is_on = True
        
        self.beginning = beginning if beginning else 0
        self.end = end if end else self.nrows
        self.current = self.beginning
        
    def reset(self):
        self.current = self.beginning
        
    def close(self):
        self.is_on = False
        self.store.close()
        self.join()
        
    def get_batch(self):
        stop = min(self.current + self.batch_size, self.end)
        if self.current < stop:
            chunk = self.store.select(self.key, start=self.current, stop=stop).as_matrix().astype(self.dtype)
            self.current += self.batch_size
            if self.transform:
                return self.transform(chunk)
            else:
                return chunk
        else:
            self.reset()
            return None

    def run(self):
        while self.is_on:
            if self.queue.empty():
                print('get batch')
                self.queue.put(self.get_batch())                  
       

class Batch(object):
   
    def __init__(self, filename, key, batch_size, train_fraction=0.9,
                 transform=None, flatten=False, dtype=numpy.float32):
        if transform:
            assert callable(transform)
            
        self.nrows = pandas.HDFStore(filename, mode='r').get_storer(key).nrows
        self.ncols = pandas.HDFStore(filename, mode='r').get_storer(key).ncols
            
        self.end = int(numpy.ceil(train_fraction * self.nrows))
        self.queue = {'train':queue.Queue(), 'validate':queue.Queue()}
        self.readers = {}
        self.readers['train'] = AsynchronousReader(filename, key, batch_size, self.queue['train'],
            dtype=dtype, transform=transform, end=self.end)
        self.readers['validate'] = AsynchronousReader(filename, key, batch_size, self.queue['validate'],
            dtype=dtype, transform=transform, beginning=self.end)
            
        for mode in self.readers:
            self.readers[mode].start()
            
    def get(self, mode='train'):
        tmp = self.queue[mode].get()
        if isinstance(tmp, type(None)):
            raise StopIteration
        else:
            return tmp
            
    def reset(self, mode='train'):
        if mode == 'all':
            self.readers['train'].reset()
            self.readers['validate'].reset()
        else:
            self.readers[mode].reset()
            
    def close(self):
        for mode in self.readers:
            self.readers[mode].close()
            
            
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
        