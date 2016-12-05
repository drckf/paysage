import os, sys, numpy, pandas

from paysage import batch
from paysage import models
from paysage import fit
from paysage import optimizers

if __name__ == "__main__":
    
    filepath = os.path.join(os.path.dirname(__file__), 'mnist', 'mnist.h5')
    b = batch.Batch(filepath, 'train/images', 200, transform=batch.color_to_ising, train_fraction=0.99)
    m = models.RestrictedBoltzmannMachine(b.cols, 5)
    opt = optimizers.ADAM(m)
    cd = fit.HCD(m, b, opt, 200, 0, skip=200, attractive=True)
    cd.train()    
    
    b.close()