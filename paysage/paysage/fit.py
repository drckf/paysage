from . import models
from . import batch
import numpy
from math import exp, log
from numba import jit, vectorize
    
EPSILON = numpy.finfo(numpy.float32).eps

# ----- SAMPLER CLASS ----- #

class SequentialMC(object):
    
    def __init__(self, amodel, adataframe):
        self.model = amodel
        try:
            self.state = adataframe.as_matrix().astype(numpy.float32)
        except Exception:
            self.state = adataframe.astype(numpy.float32)
        
    @classmethod
    def from_batch(cls, amodel, abatch):
        tmp = cls(amodel, abatch.get())
        abatch.reset()
        return tmp
        
    def update_state(self, steps):
        self.state[:] = self.model.gibbs_chain(self.state, steps)

    def resample_state(self, temperature=1.0):
        energies = self.model.marginal_energy(self.state)
        weights = importance_weights(energies, numpy.float32(temperature)).clip(min=0.0)
        indices = numpy.random.choice(numpy.arange(len(self.state)), size=len(self.state), replace=True, p=weights)
        self.state[:] = self.state[list(indices)]        


def basic_train(model, batch, epochs, method="momentum", verbose=True):
    momentum = 0.0
    sampler = SequentialMC.from_batch(model, batch)
    steps = {key: 0.001 for key in model.params}
    grad = {key: numpy.zeros_like(model.params[key]) for key in model.params}
    delta = {key: numpy.zeros_like(model.params[key]) for key in model.params}
    mean_square = {key: numpy.zeros_like(model.params[key]) for key in model.params}

    # an array to store the reconstruction error values during the descent
    mem = []
        
    edist = 0
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
            sampler = SequentialMC(model, v_data) 
            sampler.update_state(2)    
            sampler.resample_state(temperature=1.0)
            v_model = sampler.state
            
            grad = gradient(model, v_data, v_model)
            
            for key in grad:
                model.params[key][:] = model.params[key] - lr * steps[key] * grad[key]
            
            if t % 100 == 0:
                sampler = SequentialMC(model, v_data) 
                sampler.update_state(1)   
                recon = numpy.sqrt(numpy.mean((v_data - sampler.state)**2))
                sampler.resample_state(temperature=1.0)
                sampler.update_state(10)
                sampler.resample_state(temperature=1.0)
                if epoch == 0:
                    edist = energy_distance(v_data, v_model)
                else:
                    edist = 0.6 * energy_distance(v_data, v_model) + 0.4 * edist
                print(epoch, t, recon, edist, data_free_energy(model, v_data))
                
            t += 1
    
    return (v_data, v_model)
        
        
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

# ----- FUNCTIONS ----- #

@jit('float32[:](float32[:])',nopython=True)
def normalize(anarray):
    return anarray / numpy.sum(numpy.abs(anarray))
    
@jit('float32[:](float32[:],float32)',nopython=True)
def importance_weights(energies, temperature):
    gauge = energies - numpy.min(energies)
    return normalize(numpy.exp(-gauge/temperature))
    
# gradient: (LatentModel, numpy.ndarray, numpy.ndarray) -> numpy.ndarray
def gradient(model, minibatch, samples):    
    positive_phase = model.derivatives(minibatch.astype(numpy.float32))
    negative_phase = model.derivatives(samples.astype(numpy.float32))
    return {key: (positive_phase[key] - negative_phase[key]) for key in positive_phase}    
    
@jit('float32(float32[:,:],float32[:,:])',nopython=True)
def energy_distance(minibatch, samples):
    d1 = numpy.float32(0)
    d2 = numpy.float32(0)
    d3 = numpy.float32(0)

    for i in range(len(minibatch)):
        for j in range(len(minibatch)):
            d1 += numpy.linalg.norm(minibatch[i] - minibatch[j])
    d1 = d1 / (len(minibatch)**2 - len(minibatch))
    
    for i in range(len(samples)):
        for j in range(len(samples)):
            d2 += numpy.linalg.norm(samples[i] - samples[j])
    d2 = d2 / (len(samples)**2 - len(samples))
    
    for i in range(len(minibatch)):
        for j in range(len(samples)):
            d3 += numpy.linalg.norm(minibatch[i] - samples[j])
    d3 = d3 / (len(minibatch)*len(samples))
    
    return 2*d3 - d2 - d1
    
@vectorize('float32(float32)',nopython=True)
def plogp(x):
    if x <= EPSILON:
        return numpy.float32(0)
    else:
        return x * numpy.log(x)
        
@jit('float32(float32[:])',nopython=True)
def entropy(vec):
    return -numpy.sum(plogp(vec))
    
def data_free_energy(model, data):
    energies = model.marginal_energy(data)
    weights = importance_weights(energies, numpy.float32(1.0)).clip(min=0.0)
    return numpy.dot(energies, weights) - entropy(weights)
        