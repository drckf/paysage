import numpy
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import matplotlib.cm as cm

from paysage import backends as be

def plot_image(image_vector, shape, vmin=0, vmax=1, filename=None, show=True,
               cmap=cm.gray, nan_color='red'):
    f, ax = plt.subplots(figsize=(4,4))

    # reshape the data and cast to a numpy array
    img = numpy.reshape(be.to_numpy_array(image_vector), shape)

    # make the plot
    ax.imshow(img, interpolation='none', cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set(yticks=[])
    ax.set(xticks=[])

    if show:
        plt.show(f)
    if filename is not None:
        f.savefig(filename)
    plt.close(f)

def plot_image_grid(image_array, shape, vmin=0, vmax=1, filename=None, show=True,
                    cmap=cm.gray, nan_color='red'):
    # cast to a numpy array
    img_array = be.to_numpy_array(image_array)
    nrows, ncols = img_array.shape[:-1]

    f = plt.figure(figsize=(2*ncols, 2*nrows))
    grid = gs.GridSpec(nrows, ncols)
    axes = [[plt.subplot(grid[i,j]) for j in range(ncols)] for i in range(nrows)]
    for i in range(nrows):
        for j in range(ncols):
            axes[i][j].imshow(numpy.reshape(img_array[i][j], shape), cmap=cmap,
                              interpolation='none', vmin=vmin, vmax=vmax)
            axes[i][j].set(yticks=[])
            axes[i][j].set(xticks=[])
    plt.tight_layout(pad=0.5, h_pad=0.2, w_pad=0.2)
    if show:
        plt.show(f)
    if filename is not None:
        f.savefig(filename)
    plt.close(f)
