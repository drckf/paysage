import torch

DEFAULT_SEED = 137

def set_seed(n=DEFAULT_SEED):
    torch.manual_seed(int(n))

def rand(shape):
    return torch.rand(shape)

def randn(shape):
    return torch.randn(shape)
