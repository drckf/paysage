import os, sys, numpy, pandas

from paysage import batch
from paysage import models
from paysage import fit
from paysage import optimizers

import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == "__main__":
    
    filepath = os.path.join(os.path.dirname(__file__), 'mnist', 'mnist.h5')
    b = batch.Batch(filepath, 'train/images', 100, transform=batch.color_to_ising, train_fraction=0.99)
    m = models.RestrictedBoltzmannMachine(b.cols, 10)
    opt = optimizers.ADAM(m)
    
    hcd = fit.HCD(m, b, opt, 5, skip=200)
    hcd.train()    
    
    v_data = b.get()
    v_model = m.random(v_data)
    sampler = fit.SequentialMC(m, v_model) 
    sampler.update_state(1000, resample=True, temperature=1.0)
    v_model = sampler.state
    
    sns.heatmap(numpy.reshape(v_data[0], (28,28)))
    
    b.close()
    