import numpy
from . import backends as B

# ----- FUNCTIONS ----- #

class l2_penalty(object):
    
    def __init__(self, penalty):
        self.penalty = penalty
        
    def value(self, anarray):
        return 0.5 * self.penalty * numpy.sum(anarray**2)
        
    def grad(self, array):
        return self.penalty * array
    
    
class l1_penalty(object):
    
    def __init__(self, penalty):
        self.penalty = penalty
        
    def value(self, anarray):
        return self.penalty * numpy.sum(numpy.abs(anarray))
        
    def grad(self, array):
        return self.penalty * numpy.sign(array)
      
      
class log_penalty(object):
    
    def __init__(self, penalty):
        self.penalty = penalty
        
    def value(self, anarray):
        return -self.penalty * B.log(anarray)
        
    def grad(self, anarray):
        return -self.penalty / anarray
        
        
# ----- ALIASES ----- #
        
ridge = l2_penalty
lasso = l1_penalty
