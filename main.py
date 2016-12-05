import os, sys, numpy, pandas

from paysage import batch
from paysage import models
from paysage import fit
from paysage import optimizers

if __name__ == "__main__":
    
    filepath = os.path.join(os.path.dirname(__file__), 'mnist', 'mnist.h5')
    b = batch.Batch(filepath, 'train/images', 50, transform=batch.color_to_ising, train_fraction=0.99)
    m = models.RestrictedBoltzmannMachine(b.cols, 5)
    opt = optimizers.RMSProp(m)
    cd = fit.HCD(m, b, opt, 100, 10, skip=200)
    cd.train()    
    
    b.close()