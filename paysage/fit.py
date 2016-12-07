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
    
    def __init__(self, model, abatch, optimizer, epochs, convergence=1.0, skip=100):
        self.model = model
        self.batch = abatch
        self.epochs = epochs
        self.sampler = SequentialMC.from_batch(self.model, self.batch)
        self.optimizer = optimizer
        self.monitor = ProgressMonitor(skip, self.batch, convergence=convergence)

        
class ContrastiveDivergence(TrainingMethod):
    """ContrastiveDivergence
       CD-k algorithm for approximate maximum likelihood inference. 
    
       Hinton, Geoffrey E. "Training products of experts by minimizing contrastive divergence." Neural computation 14.8 (2002): 1771-1800.
       Carreira-Perpinan, Miguel A., and Geoffrey Hinton. "On Contrastive Divergence Learning." AISTATS. Vol. 10. 2005.
    """
    def __init__(self, model, abatch, optimizer, epochs, mcsteps, convergence=1.0, skip=100):
        super().__init__(model, abatch, optimizer, epochs, skip=skip, convergence=convergence)
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
                self.sampler = SequentialMC(self.model, v_data) 
                self.sampler.update_state(self.mcsteps, resample=False)    
                
                # compute the gradient and update the model parameters
                self.optimizer.update(self.model, v_data, self.sampler.state, epoch)
                
                # monitor learning progress
                prog = self.monitor.check_progress(self.model, t)
                if prog:
                    print('Batch {0}: Reconstruction Error: {1:.6f}, Energy Distance: {2:.6f}'.format(t, *prog))
                t += 1
                
            # end of epoch processing
            prog = self.monitor.check_progress(self.model, 0, store=True)
            print('End of epoch {}: '.format(epoch))
            print("-Reconstruction Error: {0:.6f}, Energy Distance: {1:.6f}".format(*prog))
            
            is_converged = self.monitor.check_convergence()
            if is_converged:
                print('Convergence criterion reached')
                break
        
        return None
             

class PersistentContrastiveDivergence(TrainingMethod):
    """PersistentContrastiveDivergence
       PCD-k algorithm for approximate maximum likelihood inference. 
    
       Tieleman, Tijmen. "Training restricted Boltzmann machines using approximations to the likelihood gradient." Proceedings of the 25th international conference on Machine learning. ACM, 2008.
    """
    
    def __init__(self, model, abatch, optimizer, epochs, mcsteps, convergence=1.0, skip=100):
        super().__init__(model, abatch, optimizer, epochs, skip=skip, convergence=convergence)
        self.mcsteps = mcsteps
        
    def train(self):
        for epoch in range(self.epochs):          
            t = 0
            while True:
                try:
                    v_data = self.batch.get(mode='train')
                except StopIteration:
                    break
                            
                # PCD keeps the sampler from the previous iteration
                self.sampler.update_state(self.mcsteps, resample=False)    
                
                # compute the gradient and update the model parameters
                self.optimizer.update(self.model, v_data, self.sampler.state, epoch)
                
                # monitor learning progress
                prog = self.monitor.check_progress(self.model, t)
                if prog:
                    print('Batch {0}: Reconstruction Error: {1:.6f}, Energy Distance: {2:.6f}'.format(t, *prog))
                t += 1
                
            # end of epoch processing
            prog = self.monitor.check_progress(self.model, 0, store=True)
            print('End of epoch {}: '.format(epoch))
            print("-Reconstruction Error: {0:.6f}, Energy Distance: {1:.6f}".format(*prog))
            
            is_converged = self.monitor.check_convergence()
            if is_converged:
                print('Convergence criterion reached')
                break
        
        return None
        
        
class HopfieldContrastiveDivergence(TrainingMethod):
    """HopfieldContrastiveDivergence
       Algorithm for approximate maximum likelihood inference based on the intuition that the weights of the network are stored as memories, like in the Hopfield model of associate memory.

       Unpublished. Charles K. Fisher (2016)
    """
    
    def __init__(self, model, abatch, optimizer, epochs, convergence=1.0, attractive=True, skip=100):
        super().__init__(model, abatch, optimizer, epochs, skip=skip, convergence=convergence)
        self.attractive = attractive
        
    def train(self):
        for epoch in range(self.epochs):          
            t = 0
            while True:
                try:
                    v_data = self.batch.get(mode='train')
                except StopIteration:
                    break
                            
                # sample near the weights
                v_model = self.model.layers['visible'].prox(self.attractive * self.model.params['weights']).T 
                
                # compute the gradient and update the model parameters
                self.optimizer.update(self.model, v_data, v_model, epoch)
                
                # monitor learning progress
                prog = self.monitor.check_progress(self.model, t)
                if prog:
                    print('Batch {0}: Reconstruction Error: {1:.6f}, Energy Distance: {2:.6f}'.format(t, *prog))
                t += 1
                
            # end of epoch processing
            prog = self.monitor.check_progress(self.model, 0, store=True)
            print('End of epoch {}: '.format(epoch))
            print("-Reconstruction Error: {0:.6f}, Energy Distance: {1:.6f}".format(*prog))
            
            is_converged = self.monitor.check_convergence()
            if is_converged:
                print('Convergence criterion reached')
                break
        
        return None
        
             
class ProgressMonitor(object):
    
    def __init__(self, skip, abatch, convergence=1.0, update_steps=10):
        self.skip = skip
        self.batch = abatch
        self.steps = update_steps
        self.convergence = convergence
        self.num_validation_samples = self.batch.index.nrows - self.batch.index.end
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
            return [recon, edist]
            
    def check_convergence(self):
        try:
            delta_recon = 100 * (self.memory[-1][0] - self.memory[-2][0]) / self.memory[-2][0] # percent change in reconstruction error
            delta_edist = 100 * (self.memory[-1][1] - self.memory[-2][1]) / self.memory[-2][1] # percent change in energy distance
            if (delta_recon > - self.convergence) or (delta_edist > - self.convergence):
                return True
            else:
                return False
        except Exception:
            return False
    

# ----- ALIASES ----- #
         
CD = ContrastiveDivergence
PCD = PersistentContrastiveDivergence
HCD = HopfieldContrastiveDivergence
        