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
        

def basic_train(model, batch, epochs, method="RMSprop", verbose=True):
    momentum = 0.0
    sampler = fit.SequentialMC.from_batch(model, batch)
    steps = {key: 0.001 for key in model.params}
    grad = {key: numpy.zeros_like(model.params[key]) for key in model.params}
    delta = {key: numpy.zeros_like(model.params[key]) for key in model.params}
    mean_square = {key: numpy.zeros_like(model.params[key]) for key in model.params}

    # an array to store the reconstruction error values during the descent
    mem = []
        
    for epoch in range(epochs):
        #learning rate decays slowly duing descent
        lr = 0.8**epoch
        
        t = 0
        while True:
            # grab a minibatch from the observed data
            try:
                v_data = batch.get()
            except StopIteration:
                break
            
            # generate a sample from the model
            sampler = fit.SequentialMC(model, v_data) 
            sampler.update_state(steps)    
            sampler.resample_state(temperature=1.0)
            v_model = sampler.state
            
            # compute the gradients
            grad = {key: gradient(model, key, v_data, v_model) for key in model.params}

            # compute the updates using RMSprop (slide 29 lecture 6 of Geoff Hinton's coursera course) or momentum
            if method == "RMSprop":
                for key in model.params:
                    mean_square[key][:] = 0.9 * mean_square[key] + (1.0 - 0.9) * grad[key]**2
                    delta[key][:] = (lr * steps[key]) / (0.000001 + numpy.sqrt(mean_square[key])) * grad[key]
            elif method == "momentum":
                for key in model.params:
                    delta[key][:] = lr * steps[key] * grad[key] + momentum * delta[key] 
            else:
                raise ValueError("method must be one of: RMSprop, momentum")
                
            # update the parameters
            for key in model.params:
                model.params[key][:] = model.params[key] + delta[key]
        
    return (a, b, W, mem)

        
        
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
        