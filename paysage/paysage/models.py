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
        return energy
   
    def marginal_energy(self, visible):
        Z_hidden = self.layers['hidden'].partition_function(self.params['hidden_bias'] + numpy.dot(visible, self.params['weights']))
        return -numpy.dot(self.params['visible_bias'], visible) - numpy.sum(numpy.log(Z_hid))

    def derivatives(self, visible):
        mean_hidden = self.layers['hidden'].mean(visible)
        derivs = {}
        self.derivs['visible_bias'] = -visible
        self.derivs['hidden_bias'] = -mean_hidden
        self.derivs['weights'] = -numpy.outer(visible, mean_hidden)
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
        assert vis_type.lower() in ['gauss', 'ising']
        assert hid_type.lower() in ['expo', 'bern']
        
        self.layers = {}
        self.layers['visible'] = layers.get(vis_type)(nvis)
        self.layers['hidden'] = layers.get(hid_type)(nhid)
        
        self.params = {}
        self.params['weights'] = numpy.random.normal(loc=0.0, scale=1.0, size=(self.layers['visible'].len, self.layers['hidden'].len)).astype(dtype=numpy.float32)
        self.params['bias'] = numpy.ones_like(self.layers['hidden'].loc)  
        self.params['T'] = numpy.ones(1, dtype=numpy.float32)
                
        self.deriv = {}
        self.deriv['weights'] = numpy.zeros_like(self.params['weights'])
        self.deriv['bias'] = numpy.zeros_like(self.params['bias'])
        self.params['T'] = numpy.zeros_like(self.params['T'])
        
        self.set_vis(numpy.zeros_like(self.layers['visible'].loc))
        
    def set_vis(self, vis):
        self.vis = vis
        self.diff = (self.params['weights'].T - vis).T
        self.squared_dist = numpy.sum(self.diff ** 2, axis=0)
        self.layers['hidden'].update_params(self.params['bias'] + self.squared_dist / (2 * self.params['T']))
        self.energy = -numpy.sum(numpy.log(self.layers['hidden'].partition_function()))        
        
    def visible_conditional_params(self, hid):
        total = numpy.sum(hid)
        loc = numpy.dot(self.params['weights'], hid) / total
        scale = self.params['T'] / total * numpy.ones_like(self.layers['visible'].loc)
        return (loc, scale)
        
    def update_visible_params(self, hid):
        self.layers['visible'].update_params(*self.visible_conditional_params(hid))
        
    def derivatives(self, vis, key):
        self.update_hidden_params(vis)
        hidden_mean = self.layers['hidden'].mean()
        if key == 'bias':
            # del H(v, k) / del b
            return hidden_mean
        elif key == 'weights':
            # del H(v, k) / del W
            return (self.difference(vis) * hidden_mean.T) / self.params['T']
        elif key == 'T':
            # del H(v,k) / del T
            return numpy.dot(hidden_mean.T, self.squared_distance(vis))
        else:
            raise ValueError('unknown key: {}'.format(key))
    """
    
# ----- FUNCTIONS ----- #
    
@jit('float32[:](float32[:,:],float32[:,:],float32[:,:])',nopython=True)
def batch_dot(vis, W, hid):
    result = numpy.zeros(len(vis), dtype=numpy.float32)
    for i in range(len(vis)):
        result[i] = numpy.dot(vis[i], numpy.dot(W, hid[i]))
    return result


            