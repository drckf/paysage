from . import models
from . import batch
import numpy
from math import exp

# ----- FUNCTIONS ----- #

# normalize: (numpy.ndarray) -> numpy.ndarray
def normalize(anarray):
    return anarray / numpy.sum(numpy.abs(anarray))
    
# gradient: (LatentModel, numpy.ndarray, numpy.ndarray) -> numpy.ndarray
def gradient(model, key, minibatch, samples):    
    positive_phase = numpy.empty_like(model.params[key])
    for row in minibatch:
        positive_phase[:] += model.derivatives(row, key)
    positive_phase = positive_phase / len(minibatch)

    negative_phase = numpy.empty_like(model.params[key])
    for row in samples:
        negative_phase[:] += model.derivatives(row, key)
    negative_phase = negative_phase / len(samples)  
        
    gradient = positive_phase - negative_phase
    return gradient
    
# total_energy: (LatentModel, numpy.ndarray) -> float
def total_energy(model, minibatch):
    return sum(model.energy(v) for v in minibatch)        
    

# ----- SAMPLER CLASS ----- #


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
        
        
"""
Steps in training algorithm:

sampler = fit.SequentialMC.from_batch(model, batch)

for e in range(epochs):

    with stop_on(StopIteration):
    
        while True:
        
            for key in model.params:
        
                minibatch = batch.get()
                
                if method == 'contrastive_divergence':
                    sampler = fit.SequentialMC(model, minibatch)                
                sampler.update_state(steps)
                if resample:
                    sampler.resample_state(temperature)
                    
                current_energy = total_energy(model, minibatch)
                current_gradient[key] = gradient(model, key, minibatch, sampler.state)
                
                delta[key]  = current_gradient[key] + momentum * previous_gradient[key]
                model.params[key][:] = model.params[key] + stepsize[key] * delta[key]
                    
                new_energy = total_energy(model, minibatch)
                
                if new_energy < current_energy:
                    # accept the step
                    previous_gradient[key][:] = current_gradient[key]
                    stepsize = 1.1 * stepsize
                else:
                    # reject the step and reset the model
                    model.params[key][:] = model.params[key] - stepsize[key] * delta[key]
                    stepsize = 0.5 * stepsize

"""        
        
        
# ---- LEARNING ALGORITHM CLASSES ----- #
        
class ContrastiveDivergence(object):
    
    def __init__(self, amodel, abatch, steps, resample=False):
        pass
            
            
#TODO:
class PersistentContrastiveDivergence(object):
    
    def __init__(self, amodel, abatch, steps):
        pass
    
    
#TODO:
class HopfieldContrastiveDivergence(object):
    
    def __init__(self, amodel, abatch, steps):
        pass


# ---- OPTIMIZER CLASSES ----- #

class StochasticGradientDescent(object):
    
    def __init__(self, method):
        assert method.lower() in ['sgd', 'momentum', 'rmsprop']
        self.method = method.lower()
        
"""
    
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
