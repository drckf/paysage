from . import models
from . import batch
import numpy
from math import exp

# ----- FUNCTIONS ----- #

# normalize: (numpy.ndarray) -> numpy.ndarray
def normalize(anarray):
    return anarray / numpy.sum(numpy.abs(anarray))


# ----- CLASSES ----- #

class SequentialMC(object):
    
    def __init__(self, amodel, adataframe):
        self.model = amodel
        self.state = numpy.array(adataframe)
        
    @classmethod
    def from_batch(cls, amodel, abatch):
        tmp = cls(amodel, abatch.get())
        abatch.reset()
        return tmp
        
    def update_state(self, steps):
        for i in range(len(self.state)):
            self.state[i,:] = numpy.ravel(self.model.gibbs_chain(self.state[i], steps))
        
    def resample_state(self, temperature=1.0):
        weights = normalize(numpy.array([exp(-self.model.energy(x) / temperature) for x in self.state]))
        indices = numpy.random.choice(numpy.arange(len(self.state)), size=len(self.state), replace=True, p=weights)
        new_state = numpy.empty_like(self.state)
        for i in range(len(self.state)):
            new_state[i,:] = self.state[indices[i]]
        self.state = new_state
        
    
class ContrastiveDivergence(object):
    
    def __init__(self, amodel, abatch, steps):
        pass
    

class PersistentContrastiveDivergence(object):
    
    def __init__(self, amodel, abatch, steps):
        pass
    
class HopfieldContrastiveDivergence(object):
    
    def __init__(self, amodel, abatch, steps):
        pass


"""
    
# grad_a: (array, array) -> vec
def grad_a(v_data, v_free):
    return numpy.mean(v_data - v_free, axis=0)
    
# grad_b: (array, array) -> vec
def grad_b(h_data, h_free):
    return numpy.mean(h_data - h_free, axis=0)
    
# grad_W: (array, array, array, array) -> array
def grad_W(h_data, v_data, h_free, v_free):
    return (numpy.dot(v_data.T,h_data) - numpy.dot(v_free.T,h_free))/len(v_data)
    
# descent: (array, int, tuple, int, int, float, int, OPTIONAL) -> tuple
def descent(reader, n_hidden, n, momentum, epochs, method = "RMSprop", verbose = True):
    astep, bstep, Wstep = 0.001, 0.001, 0.001
    a,b,W = random_params(reader.rows * reader.cols, n_hidden, (-0.5, -0.2, 0.0), (0.05, 0.05, 0.5))
    Da, Db, DW = numpy.zeros_like(a), numpy.zeros_like(b), numpy.zeros_like(W) 
    MSa, MSb, MSW = numpy.zeros_like(a), numpy.zeros_like(b), numpy.zeros_like(W) 
    # an array to store the reconstruction error values during the descent
    mem = []
        
    for epoch in range(epochs):
        #learning rate decays slowly duing descent
        lr = 0.8**epoch
        final_batch = False
        
        t = 0
        while not final_batch:
            v_data, labels, final_batch = reader.yield_next_minibatch()
            
            if epoch == 0:
                h_data, v_free, h_free = gibbs(v_data, a, b, W, n)
            else:
                h_data, v_free, h_free = gibbs(v_free, a, b, W, n)
            if t % 100 == 0:
                err = rmse(v_data, v_free)
                if verbose:
                    print("{0}: {1}: {2:.4f}".format(epoch,t,err))
                mem.append(err)      
            t += 1
            
            # compute the gradients
            da, db, dW = grad_a(v_data, v_free), grad_b(h_data, h_free), grad_W(h_data, v_data, h_free, v_free)

            # compute the updates using RMSprop (slide 29 lecture 6 of Geoff Hinton's coursera course) or momentum
            if method == "RMSprop":
                MSa = 0.9*MSa + (1-0.9)*da**2
                MSb = 0.9*MSb + (1-0.9)*db**2
                MSW = 0.9*MSW + (1-0.9)*dW**2
                
                Da = lr*astep*da / (0.00001 + numpy.sqrt(MSa))   
                Db = lr*bstep*db / (0.00001 + numpy.sqrt(MSb))  
                DW = lr*Wstep*dW / (0.00001 + numpy.sqrt(MSW))
            elif method == "momentum":
                Da = lr*astep*da + momentum*Da   
                Db = lr*bstep*db + momentum*Db     
                DW = lr*Wstep*dW + momentum*DW    
            else:
                raise ValueError("method must be one of: RMSprop, momentum")
                                
            # update the paramters
            #a, W = a + Da, W + DW
            a, b, W = a + Da, b + Db, W + DW
        
    
    return (a, b, W, mem)

if __name__ == "__main__":
    pass
"""
