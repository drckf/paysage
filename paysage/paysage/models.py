import numpy
from . import layers
from collections import OrderedDict

#---- MODEL CLASSES ----#

class LatentModel(object):
    """LatentModel
       Abstract class for a 2-layer neural network.
       
    """
    def __init__(self):
        self.layers = OrderedDict()
        self.params = {}
    
    def update_visible_params(self, hid):
        pass

    def update_hidden_params(self, vis):
        pass
    
    def energy(self, vis):
        pass
    
    def gradient(self, vis):
        pass
    
    def gibbs_step(self, vis):
        """gibbs_step(v):
           v -> h -> v'
           return v'
        
        """
        self.update_hidden_params(vis)
        hid = self.layers['hidden'].sample_state()
        self.update_visible_params(hid)
        return self.layers['visible'].sample_state()
        
    def gibbs_chain(self, vis, steps):
        """gibbs_chain(v, n):
           v -> h -> v_1 -> h_1 -> ... -> v_n
           return v_n
            
        """
        new_vis = vis.astype(vis.dtype)
        for t in range(steps):
            new_vis = self.gibbs_step(new_vis)
        return new_vis
    

class HopfieldModel(LatentModel):
    
    def __init__(self, nvis, nhid):
        self.layers = OrderedDict()
        self.layers['visible'] = layers.IsingLayer(nvis)
        self.layers['hidden'] = layers.GaussianLayer(nhid)
        
        self.params = {}
        self.params['weights'] = numpy.random.normal(loc=0.0, scale=1.0, size=(self.layers['visible'].len, self.layers['hidden'].len)).astype(dtype=numpy.float32)
        self.params['bias'] = numpy.ones_like(self.layers['visible'].loc)  

    
class RestrictedBoltzmannMachine(LatentModel):
    
    def __init__(self, nvis, nhid):
        self.layers = OrderedDict()
        self.layers['visible'] = layers.IsingLayer(nvis)
        self.layers['hidden'] = layers.BernoulliLayer(nhid)
        
        self.params = {}
        self.params['weights'] = numpy.random.normal(loc=0.0, scale=1.0, size=(self.layers['visible'].len, self.layers['hidden'].len)).astype(dtype=numpy.float32)
        self.params['visible_bias'] = numpy.ones_like(self.layers['visible'].loc)  
        self.params['hidden_bias'] = numpy.ones_like(self.layers['hidden'].loc)  
        
        
class HookeMachine(LatentModel):
    
    def __init__(self, nvis, nhid, vis_type='gauss', hid_type='expo'):   
        assert vis_type.lower() in ['gauss', 'ising']
        assert hid_type.lower() in ['expo', 'bern']
        
        self.layers = OrderedDict()
        self.layers['visible'] = layers.get(vis_type)(nvis)
        self.layers['hidden'] = layers.get(hid_type)(nhid)

        self.params = {}
        self.params['weights'] = numpy.random.normal(loc=0.0, scale=1.0, size=(self.layers['visible'].len, self.layers['hidden'].len)).astype(dtype=numpy.float32)
        self.params['bias'] = numpy.ones_like(self.layers['hidden'].loc)  
        self.params['T'] = numpy.float32(1.0)
        
        self.gradient = {}
        self.previous_gradient = {}
        for key in self.params:
            self.gradient[key] = numpy.zeros_like(self.params[key])
            self.previous_gradient[key] = numpy.zeros_like(self.params[key])
        
    def squared_diff(self, vis):
        return (numpy.reshape(vis, (-1,1)) - self.params['weights']) ** 2.0
        
    def squared_distance(self, vis):
        return numpy.sum(self.squared_diff(vis), axis=0, keepdims=True).T
        
    def visible_conditional_params(self, hid):
        total = numpy.sum(hid)
        loc = numpy.dot(self.params['weights'], numpy.reshape(hid, (-1,1))) / total
        scale = self.params['T'] / total * numpy.ones_like(self.layers['visible'].loc)
        return (loc, scale)
        
    def hidden_conditional_params(self, vis):
        return self.params['bias'] + self.squared_distance(vis) / (2 * self.params['T'])
        
    def update_visible_params(self, hid):
        self.layers['visible'].update_params(*self.visible_conditional_params(hid))

    def update_hidden_params(self, vis):
        self.layers['hidden'].update_params(self.hidden_conditional_params(vis))
        
    def energy(self, vis):
        return numpy.sum(numpy.log(self.params['bias'] + self.squared_distance(vis) / (2 * self.params['T'])))
        
    def gradient(self, vis):
        pass
