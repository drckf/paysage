from . import matrix
import torch

DEFAULT_SEED = 137

def set_seed(n=DEFAULT_SEED):
    raise NotImplementedError

def rand(shape):
    raise NotImplementedError

def randn(shape):
    raise NotImplementedError
