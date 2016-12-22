import numpy
import threading
import queue, time

class AsynchronousFunction(threading.Thread):
    
    def __init__(self, func, queue, *args, **kwargs):
        super().__init__()
        self.func = func       
        self.args = args
        self.kwargs = kwargs
        self.queue = queue

    def run(self):
        self.queue.put(self.func(*self.args, **self.kwargs))


class ThreadRandom(object):
   
    def __init__(self, distribution, seed = 137, dtype = numpy.float32):    
        self.dtype = dtype
        # set up the random number generator object
        self.rng = numpy.random.RandomState(seed)
        self.generator = getattr(self.rng, distribution)        
        # set up a queue
        self.queue = queue.Queue()
        
    def spawn(self, num_threads = 1, *args, **kwargs):
        self.thread = [None for i in range(num_threads)]
        for i in range(num_threads):
            self.thread[i] = AsynchronousFunction(self.generator, self.queue, *args, **kwargs)
            self.thread[i].start()
            
    def get(self, *args, **kwargs):
        #TODO: check that the currently queued value has the right shape
        if self.queue.empty():
            # if the queue is empty, spawn two threads that will put two things into the queue
            self.spawn(num_threads=2, *args, **kwargs)
        else:
            # if the queue is not empty, spawn a single thread to add something to it
            self.spawn(num_threads=1, *args, **kwargs)
        # pop off the end of the queue and return it
        val = self.queue.get()
        return val
            
            
foo = ThreadRandom('rand')

start = time.time()

for t in range(10):
    v = foo.get()
    time.sleep(1)

end = time.time()

print(end - start)
         
 