from .backends import numba_engine as en
import numpy, time
    
# -----  CLASSES ----- #

class SequentialMC(object):
    """SequentialMC
       Simple class for a sequential Monte Carlo sampler. 
    
    """
    def __init__(self, amodel, adataframe, method='stochastic'):
        self.model = amodel
        self.method = method
        try:
            self.state = adataframe.as_matrix().astype(numpy.float32)
        except Exception:
            self.state = adataframe.astype(numpy.float32)
        
    @classmethod
    def from_batch(cls, amodel, abatch, method='stochastic'):
        tmp = cls(amodel, abatch.get(), method=method)
        abatch.reset('all')
        return tmp
        
    def update_state(self, steps, resample=False, temperature=1.0):
        if self.method == 'stochastic':
            self.state = self.model.markov_chain(self.state, steps, resample=resample, temperature=temperature)  
        elif self.method == 'mean_field':
            self.state = self.model.mean_field_iteration(self.state, steps)  
        elif self.method == 'deterministic':
            self.state = self.model.deterministic_iteration(self.state, steps)  
        else:
            raise ValueError("Unknown method {}".format(self.method))


class TrainingMethod(object):
    
    def __init__(self, model, abatch, optimizer, epochs, convergence=1.0, skip=100, update_method='stochastic'):
        self.model = model
        self.batch = abatch
        self.epochs = epochs
        self.update_method = update_method
        self.sampler = SequentialMC.from_batch(self.model, self.batch, method=self.update_method)
        self.optimizer = optimizer
        self.monitor = ProgressMonitor(skip, self.batch, convergence=convergence)

        
class ContrastiveDivergence(TrainingMethod):
    """ContrastiveDivergence
       CD-k algorithm for approximate maximum likelihood inference. 
    
       Hinton, Geoffrey E. "Training products of experts by minimizing contrastive divergence." Neural computation 14.8 (2002): 1771-1800.
       Carreira-Perpinan, Miguel A., and Geoffrey Hinton. "On Contrastive Divergence Learning." AISTATS. Vol. 10. 2005.
    
    """
    def __init__(self, model, abatch, optimizer, epochs, mcsteps, convergence=1.0, skip=100, update_method='stochastic'):
        super().__init__(model, abatch, optimizer, epochs, skip=skip, convergence=convergence, update_method=update_method)
        self.mcsteps = mcsteps
        
    def train(self):
        for epoch in range(self.epochs):          
            t = 0
            start_time = time.time()
            while True:
                try:
                    if not t % 100:
                        print('Sampling batch: {0}'.format(t))
                    v_data = self.batch.get(mode='train')
                except StopIteration:
                    break
                            
                # CD resets the sampler from the visible data at each iteration
                self.sampler = SequentialMC(self.model, v_data, method=self.update_method) 
                self.sampler.update_state(self.mcsteps, resample=False)    
                
                # compute the gradient and update the model parameters
                self.optimizer.update(self.model, v_data, self.sampler.state, epoch)
                t += 1
                
            # end of epoch processing            
            prog = self.monitor.check_progress(self.model, 0, store=True)
            print('End of epoch {}: '.format(epoch))
            print("-Reconstruction Error: {0:.6f}, Energy Distance: {1:.6f}".format(*prog))

            end_time = time.time()
            print('Epoch took {0:.2f} seconds'.format(end_time - start_time), end='\n\n')            
            
            # convergence check should be part of optimizer
            is_converged = self.optimizer.check_convergence()
            if is_converged:
                print('Convergence criterion reached')
                break
        
        return None
             

class PersistentContrastiveDivergence(TrainingMethod):
    """PersistentContrastiveDivergence
       PCD-k algorithm for approximate maximum likelihood inference. 
    
       Tieleman, Tijmen. "Training restricted Boltzmann machines using approximations to the likelihood gradient." Proceedings of the 25th international conference on Machine learning. ACM, 2008.
   
    """    
    def __init__(self, model, abatch, optimizer, epochs, mcsteps, convergence=1.0, skip=100, update_method='stochastic'):
       super().__init__(model, abatch, optimizer, epochs, skip=skip, convergence=convergence, update_method=update_method)
       self.mcsteps = mcsteps
    
    def train(self):
        for epoch in range(self.epochs):          
            t = 0
            start_time = time.time()
            while True:
                try:
                    if not t % 100:
                        print('Sampling batch: {0}'.format(t))
                    v_data = self.batch.get(mode='train')
                except StopIteration:
                    break
                            
                # PCD keeps the sampler from the previous iteration
                self.sampler.update_state(self.mcsteps, resample=False)    
    
                # compute the gradient and update the model parameters
                self.optimizer.update(self.model, v_data, self.sampler.state, epoch)
                t += 1
                
            # end of epoch processing            
            prog = self.monitor.check_progress(self.model, 0, store=True)
            print('End of epoch {}: '.format(epoch))
            print("-Reconstruction Error: {0:.6f}, Energy Distance: {1:.6f}".format(*prog))
            
            end_time = time.time()
            print('Epoch took {0:.2f} seconds'.format(end_time - start_time), end='\n\n')  
            
            # convergence check should be part of optimizer
            is_converged = self.optimizer.check_convergence()
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
            start_time = time.time()
            while True:
                try:
                    if not t % 100:
                        print('Sampling batch: {0}'.format(t))
                    v_data = self.batch.get(mode='train')
                except StopIteration:
                    break
                            
                # sample near the weights
                v_model = self.model.layers['visible'].prox(self.attractive * self.model.params['weights']).T 
                # compute the gradient and update the model parameters
                self.optimizer.update(self.model, v_data, v_model, epoch)
                t += 1
                
            # end of epoch processing            
            prog = self.monitor.check_progress(self.model, 0, store=True)
            print('End of epoch {}: '.format(epoch))
            print("-Reconstruction Error: {0:.6f}, Energy Distance: {1:.6f}".format(*prog))
            
            end_time = time.time()
            print('Epoch took {0:.2f} seconds'.format(end_time - start_time), end='\n\n')  
            
            # convergence check should be part of optimizer
            is_converged = self.optimizer.check_convergence()
            if is_converged:
                print('Convergence criterion reached')
                break
        
        return None
        
             
#TODO: convergence should be based on magnitude of gradient updates not validation performance
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
        """energy_distance(model, v_data)
        
           Székely, Gábor J., and Maria L. Rizzo. "Energy statistics: A class of statistics based on distances." Journal of statistical planning and inference 143.8 (2013): 1249-1272.
        
        """
        v_model = model.random(v_data)
        sampler = SequentialMC(model, v_model) 
        sampler.update_state(self.steps, resample=False, temperature=1.0)
        return len(v_model) * en.fast_energy_distance(v_data, sampler.state, downsample=100)
        
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

# ----- ALIASES ----- #
         
CD = ContrastiveDivergence
PCD = PersistentContrastiveDivergence
HCD = HopfieldContrastiveDivergence
        