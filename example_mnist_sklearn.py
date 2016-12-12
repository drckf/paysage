import os, sys, numpy, pandas
from paysage.backends import numba_engine as en

from paysage import batch
from sklearn.neural_network import BernoulliRBM

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
    num_hidden_units = 500   
    batch_size = 50
    num_epochs = 20
    learning_rate = 0.1
    
    # set up the batch, model, and optimizer objects
    filepath = os.path.join(os.path.dirname(__file__), 'mnist', 'mnist.h5')
    data = batch.Batch(filepath, 'train/images', batch_size, 
                    transform=batch.binarize_color, train_fraction=0.99)
                    
    rbm = BernoulliRBM(n_components=num_hidden_units, 
                       learning_rate=learning_rate, 
                       batch_size=batch_size, 
                       n_iter=num_epochs, 
                       verbose=1)
                       
    rbm.fit(data.chunk['train'])
    
    # plot some reconstructions
    v_data = data.chunk['validate']
    v_model = rbm.gibbs(v_data)

    recon = numpy.sqrt(numpy.sum((v_data - v_model)**2) / len(v_data))
        
    plot_image(v_data[0], (28,28))
    plot_image(v_model[0], (28,28))
    
    # plot some fantasy particles
    for t in range(1000):
        v_model = rbm.gibbs(v_model)
    
    plot_image(v_data[0], (28,28))
    plot_image(v_model[0], (28,28))
    
    edist = en.fast_energy_distance(v_data.astype(numpy.float32), v_model.astype(numpy.float32), downsample=100)
    
    print('Reconstruction error:  {0:.2f}'.format(recon))
    print('Energy distance:  {0:.2f}'.format(edist))
    
    # close the HDF5 store
    data.close()
    