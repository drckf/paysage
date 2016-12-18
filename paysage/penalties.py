import numpy

# ----- FUNCTIONS ----- #

class l2_penalty(object):
    
    def __init__(self, penalty):
        self.penalty = penalty
        
    def value(self, anarray):
        return 0.5 * self.penalty * numpy.sum(anarray**2)

    def update(self, anarray, stepsize):
        anarray = anarray - stepsize * self.penalty * anarray
    
    
class l1_penalty(object):
    
    def __init__(self, penalty):
        self.penalty = penalty
        
    def value(self, anarray):
        return self.penalty * numpy.sum(numpy.abs(anarray))

    def update(self, anarray, stepsize):
        tmp = anarray - stepsize * self.penalty * anarray
        sgn = numpy.float32(numpy.sign(tmp) == numpy.sign(anarray))
        anarray = sgn * tmp
        
        
# ----- ALIASES ----- #
        
ridge = l2_penalty
lasso = l1_penalty
