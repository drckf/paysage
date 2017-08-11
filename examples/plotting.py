import numpy
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import matplotlib.cm as cm
from paysage import backends as be

def replace_nan(x, value=0):
    tensor = numpy.copy(x)
    tensor[numpy.where(x!=x)] = value
    return tensor

def plot_image(image_vector, shape, vmin=0, vmax=1, filename=None, show=True,
               cmap=cm.gray_r, nan_color='red'):
    f, ax = plt.subplots(figsize=(4,4))

    # reshape the data and cast to a numpy array
    img = numpy.reshape(be.to_numpy_array(image_vector), shape)
    # construct a masked numpy array from the data in case of nan
    img = numpy.ma.array(img, mask=numpy.isnan(img))

    # choose the color map and the color for nan
    cmap.set_bad(nan_color,1.)

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
                    cmap=cm.gray_r, nan_color='red'):
    # cast to a numpy array
    img_array = be.to_numpy_array(image_array)
    # construct a masked numpy array from the data in case of nan
    img_array = numpy.ma.array(img_array, mask=numpy.isnan(img_array))
    nrows, ncols = img_array.shape[:-1]

    # choose the color map and the color for nan
    cmap.set_bad(nan_color, 1.)

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
