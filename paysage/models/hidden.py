import numpy
from .. import layers
from .. import backends as B
from ..models.initialize import init_hidden as init
from .. import constraints
from .. import penalties

#---- MODEL CLASSES ----#

class LatentModel(object):
    """LatentModel
       Abstract class for a 2-layer neural network.
       
    """
    def __init__(self):
        self.layers = {}
        self.params = {}
        self.constraints = {}
        self.penalty = {}
                
    # placeholder function -- defined in each model
    def sample_hidden(self, visible, beta=1):
        pass
    
    # placeholder function -- defined in each model
    def sample_visible(self, hidden, beta=1):
        pass        

    # placeholder function -- defined in each model    
    def marginal_free_energy(self, visible, beta=1):
        pass
    
    def add_constraints(self, cons):
        for key in cons:
            assert key in self.params
            self.constraints[key] = cons[key]
    
    def enforce_constraints(self):
        for key in self.constraints:
            getattr(constraints, self.constraints[key])(self.params[key])
            
    def add_weight_decay(self, penalty, method='l2_penalty'):
        self.penalty.update({'weights': getattr(penalties, method)(penalty)})
    
    def mcstep(self, vis, beta=1):
        """gibbs_step(v):
           v -> h -> v'
           return v'
        
        """
        hid = self.sample_hidden(vis, beta)
        return self.sample_visible(hid, beta)
        
    def markov_chain(self, vis, steps, beta=1):
        """gibbs_chain(v, n):
           v -> h -> v_1 -> h_1 -> ... -> v_n
           return v_n
            
        """
        new_vis = vis.astype(vis.dtype)
        for t in range(steps):
            new_vis = self.mcstep(new_vis, beta)
        return new_vis
        
    def mean_field_step(self, vis, beta=1):
        """mean_field_step(v):
           v -> h -> v'
           return v'
        
           It may be worth looking into extended approaches:
           Gabrié, Marylou, Eric W. Tramel, and Florent Krzakala. "Training Restricted Boltzmann Machine via the￼ Thouless-Anderson-Palmer free energy." Advances in Neural Information Processing Systems. 2015.
        
        """
        hid = self.hidden_mean(vis, beta)
        return self.visible_mean(hid, beta)   
        
    def mean_field_iteration(self, vis, steps, beta=1):
        """mean_field_iteration(v, n):
           v -> h -> v_1 -> h_1 -> ... -> v_n
           return v_n
            
        """
        new_vis = vis.astype(vis.dtype)
        for t in range(steps):
            new_vis = self.mean_field_step(new_vis, beta)
        return new_vis
        
    def deterministic_step(self, vis, beta=1):
        """deterministic_step(v):
           v -> h -> v'
           return v'
        
        """
        hid = self.hidden_mode(vis, beta)
        return self.visible_mode(hid, beta)   
        
    def deterministic_iteration(self, vis, steps, beta=1):
        """mean_field_iteration(v, n):
           v -> h -> v_1 -> h_1 -> ... -> v_n
           return v_n
            
        """
        new_vis = vis.astype(vis.dtype)
        for t in range(steps):
            new_vis = self.deterministic_step(new_vis, beta)
        return new_vis
        
    def random(self, visible):
        return self.layers['visible'].random(visible)
        
   
class RestrictedBoltzmannMachine(LatentModel):
    """RestrictedBoltzmanMachine
    
       Hinton, Geoffrey. "A practical guide to training restricted Boltzmann machines." Momentum 9.1 (2010): 926.
    
    """
    def __init__(self, nvis, nhid, vis_type='ising', hid_type='bernoulli'):
        assert vis_type in ['ising', 'bernoulli']
        assert hid_type in ['ising', 'bernoulli']
        
        super().__init__()
        
        self.nvis = nvis
        self.nhid = nhid
        
        self.layers['visible'] = layers.get(vis_type)
        self.layers['hidden'] = layers.get(hid_type)
                
        self.params['weights'] = numpy.random.normal(loc=0.0, scale=0.01, size=(nvis, nhid)).astype(dtype=numpy.float32)
        self.params['visible_bias'] = numpy.zeros(nvis, dtype=numpy.float32)  
        self.params['hidden_bias'] = numpy.zeros(nhid, dtype=numpy.float32) 
        
    def initialize(self, data, method='hinton'):
        try:
            func = getattr(init, method)
        except AttributeError:
            print('{} is not a valid initialization method for latent models'.format(method))
        func(data, self)
        self.enforce_constraints()

    def _hidden_field(self, visible, beta=1):
        W = beta * self.params['weights']
        return B.xM_plus_a(visible, W, self.params['hidden_bias'], trans=False)

    def _visible_field(self, hidden, beta=1):
        W = beta * self.params['weights']
        return B.xM_plus_a(hidden, W, self.params['visible_bias'], trans=True)
        
    def sample_hidden(self, visible, beta=1):
        return self.layers['hidden'].sample_state(self._hidden_field(visible, beta))
            
    def hidden_mean(self, visible, beta=1):
        return self.layers['hidden'].mean(self._hidden_field(visible, beta))
            
    def hidden_mode(self, visible, beta=1):
        return self.layers['hidden'].prox(self._hidden_field(visible, beta))
              
    def sample_visible(self, hidden, beta=1):
        return self.layers['visible'].sample_state(self._visible_field(hidden, beta))
            
    def visible_mean(self, hidden, beta=1):
        return self.layers['visible'].mean(self._visible_field(hidden, beta))
            
    def visible_mode(self, hidden, beta=1):
        return self.layers['visible'].prox(self._visible_field(hidden, beta))

    def derivatives(self, visible):
        mean_hidden = self.hidden_mean(visible, beta=1)
        derivs = {}
        if len(mean_hidden.shape) == 2:
            derivs['visible_bias'] = -B.mean(visible, axis=0)
            derivs['hidden_bias'] = -B.mean(mean_hidden, axis=0)
            derivs['weights'] = -B.dot(visible.T, mean_hidden) / len(visible)
        else:
            derivs['visible_bias'] = -visible
            derivs['hidden_bias'] = -mean_hidden
            derivs['weights'] = -B.outer(visible, mean_hidden)
        return derivs
        
    def joint_energy(self, visible, hidden, beta=1):
        W = beta * self.params['weights']
        energy = -B.dot(visible, self.params['visible_bias']) - B.dot(hidden, self.params['hidden_bias'])
        if len(visible.shape) == 2:
            energy -= B.batch_dot(visible.astype(numpy.float32), W, hidden.astype(numpy.float32))
        else:
            energy -=  B.quadratic_form(visible, W, hidden)
        return B.mean(energy)
   
    def marginal_free_energy(self, visible, beta=1):
        log_Z_hidden = self.layers['hidden'].log_partition_function(self._hidden_field(visible,beta=beta))
        return -B.dot(visible, self.params['visible_bias']) - B.sum(log_Z_hidden)



class HopfieldModel(LatentModel):
    """HopfieldModel
       A model of associative memory with binary visible units and Gaussian hidden units.        
       
       Hopfield, John J. "Neural networks and physical systems with emergent collective computational abilities." Proceedings of the national academy of sciences 79.8 (1982): 2554-2558.
    
    """    
    def __init__(self, nvis, nhid, vis_type='ising'):
        assert vis_type in ['ising', 'bernoulli']
        
        super().__init__()
        
        self.nvis = nvis
        self.nhid = nhid
        
        self.layers['visible'] = layers.get(vis_type)
        self.layers['hidden'] = layers.get('gaussian')
                
        self.params['weights'] = numpy.random.normal(loc=0.0, scale=0.01, size=(nvis, nhid)).astype(dtype=numpy.float32)
        self.params['visible_bias'] = numpy.zeros(nvis, dtype=numpy.float32) 
        
        # the parameters of the hidden layer are not trainable
        self.hidden_bias = numpy.zeros(nhid, dtype=numpy.float32)
        self.hidden_scale = numpy.ones(nhid, dtype=numpy.float32)
        
    def initialize(self, data, method='hinton'):
        try:
            func = getattr(init, method)
        except AttributeError:
            print('{} is not a valid initialization method for latent models'.format(method))
        func(data, self)
        self.enforce_constraints()
    
    def _hidden_loc(self, visible, beta=1):
        W = beta * self.params['weights']
        return B.xM_plus_a(visible, W, self.hidden_bias, trans=False)
    
    def _visible_field(self, hidden, beta=1):
        W = beta * self.params['weights']
        return B.xM_plus_a(hidden, W, self.params['visible_bias'], trans=True)
        
    def sample_hidden(self, visible, beta=1):
        return self.layers['hidden'].sample_state(self._hidden_loc(visible, beta), self.hidden_scale)
            
    def hidden_mean(self, visible, beta=1):
        return self.layers['hidden'].mean(self._hidden_loc(visible, beta))
            
    def hidden_mode(self, visible, beta=1):
        return self.layers['hidden'].prox(self._hidden_loc(visible, beta))
              
    def sample_visible(self, hidden, beta=1):
        return self.layers['visible'].sample_state(self._visible_field(hidden, beta))
            
    def visible_mean(self, hidden, beta=1):
        return self.layers['visible'].mean(self._visible_field(hidden, beta))
            
    def visible_mode(self, hidden, beta=1):
        return self.layers['visible'].prox(self._visible_field(hidden,beta))
        
    def derivatives(self, visible):
        mean_hidden = self.hidden_mean(visible, beta=1)
        derivs = {}
        if len(mean_hidden.shape) == 2:
            derivs['visible_bias'] = -B.mean(visible, axis=0)
            derivs['weights'] = -B.batch_outer(visible, mean_hidden) / len(visible)
        else:
            derivs['visible_bias'] = -visible
            derivs['weights'] = -B.outer(visible, mean_hidden)
        return derivs
        
    def joint_energy(self, visible, hidden, beta=1):
        W = beta * self.params['weights']
        energy = -B.dot(visible, self.params['visible_bias']) - B.msum(hidden**2, axis=1)
        if len(visible.shape) == 2:
            energy -= B.batch_dot(visible.astype(numpy.float32), W, hidden.astype(numpy.float32))
        else:
            energy -=  B.quadratic_form(visible, self.params['weights'], hidden)
        return B.mean(energy)
   
    def marginal_free_energy(self, visible, beta=1):
        W = beta * self.params['weights']
        J = B.dot(W, W.T)
        return -B.dot(visible, self.params['visible_bias']) - B.batch_dot(visible, J, visible)

   

class GaussianRestrictedBoltzmannMachine(LatentModel):
    """GaussianRestrictedBoltzmanMachine
       RBM with Gaussian visible units. 
    
       Hinton, Geoffrey. "A practical guide to training restricted Boltzmann machines." Momentum 9.1 (2010): 926.
    
    """    
    def __init__(self, nvis, nhid, hid_type='bernoulli'):
        assert hid_type in ['ising', 'bernoulli']
        
        super().__init__()
        
        self.nvis = nvis
        self.nhid = nhid
        
        self.layers['visible'] = layers.get('gaussian')
        self.layers['hidden'] = layers.get(hid_type)
                
        self.params['weights'] = numpy.random.normal(loc=0.0, scale=0.01, size=(nvis, nhid)).astype(dtype=numpy.float32)
        self.params['visible_bias'] = numpy.zeros(nvis, dtype=numpy.float32)  
        self.params['visible_scale'] = numpy.zeros(nvis, dtype=numpy.float32)  
        self.params['hidden_bias'] = numpy.zeros(nhid, dtype=numpy.float32) 
  
    def initialize(self, data, method='hinton'):
        try:
            func = getattr(init, method)
        except AttributeError:
            print('{} is not a valid initialization method for latent models'.format(method))
        func(data, self)
        self.enforce_constraints()

    def _hidden_field(self, visible, beta=1):
        W = beta * self.params['weights']
        scale = B.exp(self.params['visible_scale'])
        return B.xM_plus_a(visible / scale, W, self.params['hidden_bias'], trans=False)

    def _visible_loc(self, hidden, beta=1):
        W = beta * self.params['weights']
        return B.xM_plus_a(hidden, W, self.params['visible_bias'], trans=True)
        
    def sample_hidden(self, visible, beta=1):
        return self.layers['hidden'].sample_state(self._hidden_field(visible, beta))
            
    def hidden_mean(self, visible, beta=1):
        return self.layers['hidden'].mean(self._hidden_field(visible, beta))
            
    def hidden_mode(self, visible, beta=1):
        return self.layers['hidden'].prox(self._hidden_field(visible, beta))
              
    def sample_visible(self, hidden, beta=1):
        scale = B.exp(0.5 * self.params['visible_scale'])
        return self.layers['visible'].sample_state(self._visible_loc(hidden, beta), scale)
            
    def visible_mean(self, hidden, beta=1):
        return self.layers['visible'].mean(self._visible_loc(hidden, beta))
            
    def visible_mode(self, hidden, beta=1):
        return self.layers['visible'].prox(self._visible_loc(hidden,beta))

    def derivatives(self, visible):
        mean_hidden = self.hidden_mean(visible, beta=1)
        scale = B.exp(self.params['visible_scale'])
        v_scaled = visible / scale
        derivs = {}
        if len(mean_hidden.shape) == 2:
            derivs['visible_bias'] = -B.mean(v_scaled, axis=0)
            derivs['hidden_bias'] = -B.mean(mean_hidden, axis=0)
            derivs['weights'] = -B.dot(v_scaled.T, mean_hidden) / len(visible)
            derivs['visible_scale'] = -0.5 * B.mean((visible-self.params['visible_bias'])**2, axis=0) + B.batch_dot(mean_hidden, self.params['weights'].T, visible, axis=0) / len(visible)
            derivs['visible_scale'] /= scale
        else:
            derivs['visible_bias'] = -v_scaled
            derivs['hidden_bias'] = -mean_hidden
            derivs['weights'] = -B.outer(v_scaled, mean_hidden)
            derivs['visible_scale'] = -0.5 * (visible - self.params['visible_bias'])**2 + B.dot(self.params['weights'], mean_hidden)
            derivs['visible_scale'] /= scale     
        return derivs
        
    def joint_energy(self, visible, hidden, beta=1):
        W = beta * self.params['weights']
        scale = B.exp(self.params['visible_scale'])
        v_scaled = visible / scale
        energy = 0.5 * B.mean((visible - self.params['visible_bias'])**2 / scale, axis=1) - B.dot(hidden, self.params['hidden_bias'])
        if len(visible.shape) == 2:
            energy -= B.batch_dot(v_scaled, W, hidden)
        else:
            energy -=  B.quadratic_form(v_scaled, W, hidden)
        return B.mean(energy)
           
    def marginal_free_energy(self, visible, beta=1):
        scale = B.exp(self.params['visible_scale'])
        v_scaled = visible / scale
        log_Z_hidden = self.layers['hidden'].log_partition_function(self.hidden_field(v_scaled, beta))
        return 0.5 * B.mean((visible - self.params['visible_bias'])**2 / scale, axis=1) - B.sum(log_Z_hidden)        
  
      
#TODO:
class HookeMachine(LatentModel):
    """HookeMachine
    
       Unpublished. Charles K. Fisher (2016)
    
    """
    def __init__(self, nvis, nhid, vis_type='gauss', hid_type='expo'):   
        pass


    
# ----- ALIASES ----- #
    
RBM = RestrictedBoltzmannMachine    
GRBM = GaussianRBM = GaussianRestrictedBoltzmannMachine
            