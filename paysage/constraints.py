import numpy

# ----- FUNCTIONS ----- #
        
def non_negative(anarray):
    anarray.clip(min=0.0, out=anarray).astype(numpy.float32)

def non_positive(anarray):
    anarray.clip(max=0.0, out=anarray).astype(numpy.float32)    
    