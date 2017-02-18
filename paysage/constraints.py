from . import backends as be

# ----- FUNCTIONS ----- #

def non_negative(tensor):
    be.clip_inplace(tensor, a_min=0.0)

def non_positive(tensor):
    be.clip_inplace(tensor, a_max=0.0)
