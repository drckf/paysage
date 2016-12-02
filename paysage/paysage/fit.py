from . import models
from . import batch
import numpy
from numba import jit
    
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


class TrainingMethod(object):
    
    def __init__(self, model, batch, epochs, skip=100):
        self.model = model
        self.batch = batch
        self.epochs = epochs
        self.sampler = SequentialMC.from_batch(model, batch)
        self.optimizer = ADAM(model)
        self.monitor = ProgressMonitor(skip, batch)

        
class ContrastiveDivergence(TrainingMethod):
    
    def __init__(self, model, batch, epochs, mcsteps, skip=100):
        super().__init__(model, batch, epochs, skip=skip)
        self.mcsteps = mcsteps
        
    def train(self):
        for epoch in range(self.epochs):          
            t = 0
            while True:
                try:
                    v_data = self.batch.get(mode='train')
                except StopIteration:
                    break
                            
                # CD resets the sampler from the visible data at each iteration
                sampler = SequentialMC(self.model, v_data) 
                sampler.update_state(self.mcsteps)    
                
                # compute the gradient and update the model parameters
                self.optimizer.update(self.model, v_data, sampler.state, epoch)
                
                # monitor learning progress
                prog = self.monitor.check_progress(self.model, t)
                if prog:
                    print(epoch, *prog)
                t += 1
        
        return None
             
             
class ProgressMonitor(object):
    
    def __init__(self, skip, batch):
        self.skip = skip
        self.batch = batch
        self.num_validation_samples = batch.index.end['validate'] - batch.index.end['train']
        
    def check_progress(self, model, t):
        if not (t % self.skip):
            recon = 0
            edist = 0
            while True:
                try:
                    v_data = self.batch.get(mode='validate')
                except StopIteration:
                    break
                sampler = SequentialMC(model, v_data) 
                sampler.update_state(1)   
                recon += numpy.sum((v_data - sampler.state)**2)
                #sampler.resample_state(temperature=1.0)
                #sampler.update_state(10)
                #sampler.resample_state(temperature=1.0)
                #edist += energy_distance(v_data, sampler.state)
            recon = numpy.sqrt(recon / self.num_validation_samples)
            edist = edist / self.num_validation_samples
            return t, recon, edist
    
# ----- OPTIMIZERS ----- #        
        
class StochasticGradientDescent(object):
    
    def __init__(self, model, stepsize=0.001, lr_decay=0.5):
        self.lr_decay = lr_decay
        self.stepsize = stepsize
        self.grad = {key: numpy.zeros_like(model.params[key]) for key in model.params}
    
    def update(self, model, v_data, v_model, epoch):
        lr = self.lr_decay ** epoch
        self.grad = gradient(model, v_data, v_model)
        for key in self.grad:
            model.params[key][:] = model.params[key] - lr * self.stepsize * self.grad[key]
         
         
class Momentum(object):
    
    def __init__(self, model, stepsize=0.001, momentum=0.9, lr_decay=0.5):
        self.lr_decay = lr_decay
        self.stepsize = stepsize
        self.momentum = momentum
        self.grad = {key: numpy.zeros_like(model.params[key]) for key in model.params}
        self.delta = {key: numpy.zeros_like(model.params[key]) for key in model.params}
    
    def update(self, model, v_data, v_model, epoch):
        lr = self.lr_decay ** epoch
        self.grad = gradient(model, v_data, v_model)
        for key in self.grad:
            self.delta[key][:] = self.grad[key] + self.momentum * self.delta[key]
            model.params[key][:] = model.params[key] - lr * self.stepsize * self.delta[key]


class RMSProp(object):
    
    def __init__(self, model, stepsize=0.001, mean_square_weight=0.9):
        self.stepsize = stepsize
        self.mean_square_weight = mean_square_weight
        self.grad = {key: numpy.zeros_like(model.params[key]) for key in model.params}
        self.mean_square_grad = {key: numpy.zeros_like(model.params[key]) for key in model.params}
        self.epsilon = 10**-6
    
    def update(self, model, v_data, v_model, epoch):
        self.grad = gradient(model, v_data, v_model)
        for key in self.grad:
            self.mean_square_grad[key] = self.mean_square_weight * self.mean_square_grad[key] + (1-self.mean_square_weight)*self.grad[key]**2
            model.params[key][:] = model.params[key] - self.stepsize * self.grad[key] / numpy.sqrt(self.epsilon + self.mean_square_grad[key])
 
 
class ADAM(object):
    
    def __init__(self, model, stepsize=0.001, mean_weight=0.9, mean_square_weight=0.99):
        self.stepsize = stepsize
        self.mean_weight = mean_weight
        self.mean_square_weight = mean_square_weight
        self.grad = {key: numpy.zeros_like(model.params[key]) for key in model.params}
        self.mean_square_grad = {key: numpy.zeros_like(model.params[key]) for key in model.params}
        self.mean_grad = {key: numpy.zeros_like(model.params[key]) for key in model.params}
        self.epsilon = 10**-6
    
    def update(self, model, v_data, v_model, epoch):
        self.grad = gradient(model, v_data, v_model)
        for key in self.grad:
            self.mean_square_grad[key] = self.mean_square_weight * self.mean_square_grad[key] + (1-self.mean_square_weight)*self.grad[key]**2
            self.mean_grad[key] = self.mean_weight * self.mean_grad[key] + (1-self.mean_weight)*self.grad[key]            
            model.params[key][:] = model.params[key] - (self.stepsize / (1 - self.mean_weight)) * self.mean_grad[key] / numpy.sqrt(self.epsilon + self.mean_square_grad[key] / (1 - self.mean_square_weight))
         
         
# ----- ALIASES ----- #
         
sgd = SGD = StochasticGradientDescent   
momentum = Momentum   
rmsprop = RMSProp   
adam = ADAM
        

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
        