import os, sys, numpy, pandas

from paysage import batch
from paysage import models
from paysage import fit

if __name__ == "__main__":
    
    filepath = os.path.join(os.path.dirname(__file__), 'mnist', 'mnist.h5')
    b = batch.Batch(filepath, 'train/images', 100, transform=batch.color_to_ising)
    labels = batch.Batch(filepath, 'train/labels', 100)
    m = models.RestrictedBoltzmannMachine(b.cols, 10)
    sampler = fit.SequentialMC.from_batch(m, b)
    
    minibatch = b.get()
    v0 = minibatch[0]
    
    hid = m.sample_hidden(minibatch)
    m.joint_energy(minibatch, hid)
    
    sampler.update_state(1)
    sampler.resample_state(temperature=1.0)

    #v_d, v_m = fit.basic_train(m, b, 1, method="RMSprop", verbose=True)
    
    