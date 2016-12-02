import numpy
from . import layers
from collections import OrderedDict
from numba import jit, vectorize

#---- MODEL CLASSES ----#

class LatentModel(object):
    """LatentModel
       Abstract class for a 2-layer neural network.
       
    """
    def __init__(self):
        self.layers = {}
        self.params = {}
                
    def sample_hidden(self, visible):
        pass
    
    def sample_visible(self, hidden):
        pass        
    
    def marginal_energy(self, visible):
        pass
    
    def resample_state(self, visibile, temperature=1.0):
        energies = self.marginal_energy(visibile)
        weights = importance_weights(energies, numpy.float32(temperature)).clip(min=0.0)
        indices = numpy.random.choice(numpy.arange(len(visibile)), size=len(visibile), replace=True, p=weights)
        return visibile[list(indices)]  
    
    def gibbs_step(self, vis):
        """gibbs_step(v):
           v -> h -> v'
           return v'
        
        """
        hid = self.sample_hidden(vis)
        return self.sample_visible(hid)
        
    def gibbs_chain(self, vis, steps, resample=False, temperature=1.0):
        """gibbs_chain(v, n):
           v -> h -> v_1 -> h_1 -> ... -> v_n
           return v_n
            
        """
        new_vis = vis.astype(vis.dtype)
        for t in range(steps):
            new_vis = self.gibbs_step(new_vis)
            if resample:
                new_vis = self.resample_state(new_vis, temperature=temperature)
        return new_vis
   
   
class RestrictedBoltzmannMachine(LatentModel):
    """RestrictedBoltzmanMachine
    
    """
    def __init__(self, nvis, nhid):
        self.nvis = nvis
        self.nhid = nhid
        
        self.layers = {}
        self.layers['visible'] = layers.IsingLayer()
        self.layers['hidden'] = layers.BernoulliLayer()
        
        self.params = {}
        self.params['weights'] = numpy.random.normal(loc=0.0, scale=1.0, size=(nvis, nhid)).astype(dtype=numpy.float32)
        self.params['visible_bias'] = numpy.ones(nvis, dtype=numpy.float32)  
        self.params['hidden_bias'] = numpy.ones(nhid, dtype=numpy.float32) 
        
    def sample_hidden(self, visible):
        field = self.params['hidden_bias'] + numpy.dot(visible, self.params['weights'])
        if len(field.shape) == 2:
            return numpy.array([self.layers['hidden'].sample_state(f) for f in field], dtype=numpy.float32)       
        else:
            return self.layers['hidden'].sample_state(field)
              
    def sample_visible(self, hidden):
        field = self.params['visible_bias'] + numpy.dot(hidden, self.params['weights'].T)
        if len(field.shape) == 2:
            return numpy.array([self.layers['visible'].sample_state(f) for f in field], numpy.float32)
        else:
            return self.layers['visible'].sample_state(field)
        
    def joint_energy(self, visible, hidden):
        energy = -numpy.dot(visible, self.params['visible_bias']) - numpy.dot(hidden, self.params['hidden_bias'])
        if len(visible.shape) == 2:
            energy = energy - batch_dot(visible.astype(numpy.float32), self.params['weights'], hidden.astype(numpy.float32))
        else:
            energy =  energy - numpy.dot(visible, numpy.dot(self.params['weights'], hidden))
        return numpy.mean(energy)
   
    def marginal_energy(self, visible):
        field = self.params['hidden_bias'] + numpy.dot(visible, self.params['weights'])
        log_Z_hidden = self.layers['hidden'].log_partition_function(field)
        return -numpy.dot(visible, self.params['visible_bias']) - numpy.sum(log_Z_hidden)

    def derivatives(self, visible):
        field = self.params['hidden_bias'] + numpy.dot(visible, self.params['weights'])
        mean_hidden = self.layers['hidden'].mean(field)
        derivs = {}
        if len(mean_hidden.shape) == 2:
            derivs['visible_bias'] = -numpy.mean(visible, axis=0)
            derivs['hidden_bias'] = -numpy.mean(mean_hidden, axis=0)
            derivs['weights'] = -batch_outer(visible, mean_hidden)
        else:
            derivs['visible_bias'] = -visible
            derivs['hidden_bias'] = -mean_hidden
            derivs['weights'] = -outer(visible, mean_hidden)
        return derivs
        
    def random(self, visible):
        return self.layers['visible'].random(visible)


"""  
#TODO:
class HopfieldModel(LatentModel):
    
    def __init__(self, nvis, nhid):
        pass

#TODO:
class HookeMachine(LatentModel):
    
    def __init__(self, nvis, nhid, vis_type='gauss', hid_type='expo'):   
        pass

    """
    
# ----- ALIASES ----- #
    
RBM = RestrictedBoltzmannMachine    
    
    
# ----- FUNCTIONS ----- #
    
@jit('float32[:](float32[:,:],float32[:,:],float32[:,:])',nopython=True)
def batch_dot(vis, W, hid):
    result = numpy.zeros(len(vis), dtype=numpy.float32)
    for i in range(len(vis)):
        result[i] = numpy.dot(vis[i], numpy.dot(W, hid[i]))
    return result

@jit('float32[:,:](float32[:],float32[:])',nopython=True)
def outer(vis, hid):
    result = numpy.zeros((len(vis), len(hid)), dtype=numpy.float32)
    for i in range(len(vis)):
        for u in range(len(hid)):
            result[i][u] = vis[i] * hid[u]
    return result
    
@jit('float32[:,:](float32[:],float32[:], float32[:,:])',nopython=True)
def outer_inplace(vis, hid, result):
    for i in range(len(vis)):
        for u in range(len(hid)):
            result[i][u] += vis[i] * hid[u]
    return result
    
@jit('float32[:,:](float32[:,:],float32[:,:])',nopython=True)
def batch_outer(vis, hid):
    result = numpy.zeros((vis.shape[1], hid.shape[1]), dtype=numpy.float32)
    for i in range(len(vis)):
        outer_inplace(vis[i], hid[i], result)
    return result / len(vis)
    
@jit('float32[:](float32[:])',nopython=True)
def normalize(anarray):
    return anarray / numpy.sum(numpy.abs(anarray))
    
@jit('float32[:](float32[:],float32)',nopython=True)
def importance_weights(energies, temperature):
    gauge = energies - numpy.min(energies)
    return normalize(numpy.exp(-gauge/temperature)) 
            