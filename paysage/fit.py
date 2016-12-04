from . import models
from . import batch
from . import optimizers
from .backends import numba_engine as en
import numpy
    
# -----  CLASSES ----- #

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
        
    def update_state(self, steps, resample=False, temperature=1.0):
        self.state[:] = self.model.gibbs_chain(self.state, steps, resample=resample, temperature=temperature)  


class TrainingMethod(object):
    
    def __init__(self, model, abatch, optimizer, epochs, skip=100):
        self.model = model
        self.batch = abatch
        self.epochs = epochs
        self.sampler = SequentialMC.from_batch(self.model, self.batch)
        self.optimizer = optimizer
        self.monitor = ProgressMonitor(skip, self.batch)

        
class ContrastiveDivergence(TrainingMethod):
    
    def __init__(self, model, abatch, optimizer, epochs, mcsteps, skip=100):
        super().__init__(model, abatch, optimizer, epochs, skip=skip)
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
                sampler.update_state(self.mcsteps, resample=False)    
                
                # compute the gradient and update the model parameters
                self.optimizer.update(self.model, v_data, sampler.state, epoch)
                
                # monitor learning progress
                prog = self.monitor.check_progress(self.model, t)
                if prog:
                    print('Batch {0}: Reconstruction Error: {1:.6f}, Energy Distance: {2:.6f}'.format(t, *prog))
                t += 1
            # end of epoch processing
            prog = self.monitor.check_progress(self.model, 0, store=True)
            print('End of epoch {}: '.format(epoch))
            print("-Reconstruction Error: {0:.6f}, Energy Distance: {1:.6f}".format(*prog))
        
        return None
             
             
class ProgressMonitor(object):
    
    def __init__(self, skip, abatch, update_steps=10):
        self.skip = skip
        self.batch = abatch
        self.steps = update_steps
        self.num_validation_samples = self.batch.index.end['validate'] - self.batch.index.end['train']
        self.memory = []

    def reconstruction_error(self, model, v_data):
        sampler = SequentialMC(model, v_data) 
        sampler.update_state(1)   
        return numpy.sum((v_data - sampler.state)**2)
        
    def energy_distance(self, model, v_data):
        v_model = model.random(v_data)
        sampler = SequentialMC(model, v_model) 
        sampler.update_state(self.steps, resample=False, temperature=1.0)
        return len(v_model) * en.energy_distance(v_data, sampler.state)
        
    def check_progress(self, model, t, store=False):
        if not (t % self.skip):
            recon = 0
            edist = 0
            while True:
                try:
                    v_data = self.batch.get(mode='validate')
                except StopIteration:
                    break
                recon += self.reconstruction_error(model, v_data)
                edist += self.energy_distance(model, v_data)
            recon = numpy.sqrt(recon / self.num_validation_samples)
            edist = edist / self.num_validation_samples
            if store:
                self.memory.append([recon, edist])
            return recon, edist
            
    def check_convergence(self):
        pass
    

# ----- ALIASES ----- #
         
CD = ContrastiveDivergence
        