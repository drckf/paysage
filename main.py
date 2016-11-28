import os, sys, numpy, pandas

from paysage import batch
from paysage import models
from paysage import fit

if __name__ == "__main__":
    
    filepath = os.path.join(os.path.dirname(__file__), 'mnist', 'mnist.h5')
    b = batch.Batch(filepath, 'train/images', 100, transform=batch.color_to_ising)
    m = models.HookeMachine(b.cols, 10, vis_type = 'Ising')
    sampler = fit.SequentialMC.from_batch(m, b)
    
    minibatch = b.get()
    v0 = minibatch[0]
    
    foo = m.derivatives(v0, 'T')
    print(foo)
    #v_d, v_m = fit.basic_train(m, b, 1, method="RMSprop", verbose=True)
    
    