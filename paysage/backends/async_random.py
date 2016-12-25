import numpy, threading, queue

threading.stack_size(10*524288)

class AsynchronousFunction(threading.Thread):
    
    def __init__(self, func, queue, *args, **kwargs):
        super().__init__()
        self.daemon = False
        self.func = func       
        self.args = args
        self.kwargs = kwargs
        self.queue = queue

    def run(self):
        self.queue.put(self.func(*self.args, **self.kwargs))
        self.is_alive = False


class ThreadRandom(object):
   
    def __init__(self, distribution, seed = 137, dtype = numpy.float32):    
        self.dtype = dtype
        # set up the random number generator object
        self.rng = numpy.random.RandomState(seed)
        self.generator = getattr(self.rng, distribution)        
        # set up a queue
        self.queue = queue.Queue()
        
    def spawn(self, *args, num_threads = 1, **kwargs):
        thread = [None for i in range(num_threads)]
        for i in range(num_threads):
            thread[i] = AsynchronousFunction(self.generator, self.queue, *args, **kwargs)
            thread[i].start()
            
    def get(self, *args, **kwargs):
        if self.queue.empty():
            # if the queue is empty, spawn two thread to put some things into the queue
            self.spawn(*args, num_threads=2, **kwargs)
        else:
            # if the queue is not empty, spawn a single thread to add something to it
            self.spawn(*args, num_threads=1, **kwargs)
        # pop off the end of the queue and return it
        val = self.queue.get()
        return self.dtype(val)   
      
      
if __name__ == "__main__":
    import time
    r = ThreadRandom('normal')
    
    for t in range(10):
        r.get(size=(100,100))
        lst = threading.enumerate()
        for l in lst:
            print(l)
        time.sleep(1)
        print()
 