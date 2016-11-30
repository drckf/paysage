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
    
    def gibbs_step(self, vis):
        """gibbs_step(v):
           v -> h -> v'
           return v'
        
        """
        hid = self.sample_hidden(vis)
        return self.sample_visible(hid)
        
    def gibbs_chain(self, vis, steps):
        """gibbs_chain(v, n):
           v -> h -> v_1 -> h_1 -> ... -> v_n
           return v_n
            
        """
        new_vis = vis.astype(vis.dtype)
        for t in range(steps):
            new_vis = self.gibbs_step(new_vis)
        return new_vis
   
   
#TODO:
class RestrictedBoltzmannMachine(LatentModel):
    
    def __init__(self, nvis, nhid):
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

"""  
#TODO:
class HopfieldModel(LatentModel):
    
    def __init__(self, nvis, nhid):
        self.layers = {}
        self.layers['visible'] = layers.IsingLayer(nvis)
        self.layers['hidden'] = layers.GaussianLayer(nhid)
        
        self.params = {}
        self.params['weights'] = numpy.random.normal(loc=0.0, scale=1.0, size=(self.layers['visible'].len, self.layers['hidden'].len)).astype(dtype=numpy.float32)
        self.params['bias'] = numpy.ones_like(self.layers['visible'].loc)  


class HookeMachine(LatentModel):
    
    def __init__(self, nvis, nhid, vis_type='gauss', hid_type='expo'):   
        pass

    """
    
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
    

            