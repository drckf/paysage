import os, sys, numpy, pandas

from paysage import batch
from paysage import models
from paysage import fit

if __name__ == "__main__":
    
    filepath = os.path.join(os.path.dirname(__file__), 'mnist', 'mnist.h5')
    b = batch.Batch(filepath, 'train/images', 100, transform=batch.color_to_ising, train_fraction=0.99)
    m = models.RestrictedBoltzmannMachine(b.cols, 5)
    cd = fit.ContrastiveDivergence(m, b, 5, 1, skip=200)
    cd.train()    
    