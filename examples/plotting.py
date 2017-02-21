import numpy
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import seaborn as sns
from paysage import backends as be

def plot_image(image_vector, shape):
    f, ax = plt.subplots(figsize=(4,4))
    array = be.to_numpy_array(image_vector)
    hm = sns.heatmap(numpy.reshape(array, shape), ax=ax, cmap="gray_r", cbar=False)
    hm.set(yticks=[])
    hm.set(xticks=[])
    plt.show(f)
    plt.close(f)

def plot_image_grid(image_array, shape, vmin=0, vmax=1):
    array = be.to_numpy_array(image_array)
    nrows, ncols = array.shape[:-1]
    f = plt.figure(figsize=(2*ncols, 2*nrows))
    grid = gs.GridSpec(nrows, ncols)
    axes = [[plt.subplot(grid[i,j]) for j in range(ncols)] for i in range(nrows)]
    for i in range(nrows):
        for j in range(ncols):
            sns.heatmap(numpy.reshape(array[i][j], shape),
                ax=axes[i][j], cmap="gray_r", cbar=False, vmin=vmin, vmax=vmax)
            axes[i][j].set(yticks=[])
            axes[i][j].set(xticks=[])
    plt.show(f)
    plt.close(f)
