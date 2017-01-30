import numpy
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import seaborn as sns

def plot_image(image_vector, shape):
    f, ax = plt.subplots(figsize=(4,4))
    hm = sns.heatmap(numpy.reshape(image_vector, shape), ax=ax, cmap="gray_r", cbar=False)
    hm.set(yticks=[])
    hm.set(xticks=[])
    plt.show(f)
    plt.close(f)

def plot_image_grid(image_array, shape, vmin=0, vmax=1):
    nrows, ncols = image_array.shape[:-1]
    f = plt.figure(figsize=(2*ncols, 2*nrows))
    grid = gs.GridSpec(nrows, ncols)
    axes = [[plt.subplot(grid[i,j]) for j in range(ncols)] for i in range(nrows)]
    for i in range(nrows):
        for j in range(ncols):
            sns.heatmap(numpy.reshape(image_array[i][j], shape),
                        ax=axes[i][j], cmap="gray_r", cbar=False, vmin=vmin, vmax=vmax)
            axes[i][j].set(yticks=[])
            axes[i][j].set(xticks=[])
    plt.show(f)
    plt.close(f)
