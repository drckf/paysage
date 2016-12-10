import os, sys, numpy, pandas

from paysage import batch
from paysage.models import hidden
from paysage import fit
from paysage import optimizers

import matplotlib.pyplot as plt
import seaborn as sns

def plot_image(image_vector, shape):
    f, ax = plt.subplots(figsize=(4,4))
    hm = sns.heatmap(numpy.reshape(image_vector, shape), ax=ax, cmap="gray_r", cbar=False)
    hm.set(yticks=[])
    hm.set(xticks=[])
    plt.show(f)
    plt.close(f)    

if __name__ == "__main__":
    
    # set up the batch, model, and optimizer objects
    filepath = os.path.join(os.path.dirname(__file__), 'mnist', 'mnist.h5')
    b = batch.Batch(filepath, 'train/images', 100, transform=batch.color_to_ising, train_fraction=0.99)
    num_hidden_units = 200   
    m = hidden.RestrictedBoltzmannMachine(b.ncols, num_hidden_units)
    opt = optimizers.ADAM(m)
    """
    # train the model with contrastive divergence
    print('training with hopfield contrastive divergence')
    cd = fit.HCD(m, b, opt, 10, skip=200, convergence=0.0)
    cd.train()  
    """
    
    print('\ncontinuing training with contrastive divergence')
    cd = fit.CD(m, b, opt, 1, 1, skip=200, convergence=0.0)
    cd.train()    
    
    """
    # plot some reconstructions
    v_data = b.get()
    sampler = fit.SequentialMC(m, v_data) 
    sampler.update_state(1, resample=False, temperature=1.0)
    v_model = sampler.state
    
    plot_image(v_data[0], (28,28))
    plot_image(v_model[0], (28,28))
    
    # plot some fantasy particles
    v_model = m.random(v_data)
    sampler = fit.SequentialMC(m, v_model) 
    sampler.update_state(1000, resample=False, temperature=1.0)
    v_model = sampler.state
    
    plot_image(v_data[0], (28,28))
    plot_image(v_model[0], (28,28))
    """
    # close the HDF5 store
    b.close()
    