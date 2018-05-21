import numpy
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import matplotlib.cm as cm

from .. import backends as be

def plot_metrics(history, filename=None, show=True):
    """
    Plots a set of metrics in a history (List of dicts of {metric name : value}).

    Args:
        history (List: ProgressMonitor.memory attribute)
        filename (optional; str)
        show (optional; bool)

    Returns:
        None

    """

    # collate the data out
    metric_names = sorted(list(history[0].keys()))
    metrics = {name : [h[name] for h in history] for name in metric_names}

    # filter out any metrics that are None
    metrics = {name: metrics[name] for name in metrics if not
               any(m is None for m in metrics[name])}
    metric_names = sorted(list(metrics.keys()))

    # build the container for the figures
    num_metrics = len(metric_names)
    num_rows = int(numpy.ceil(numpy.sqrt(num_metrics)))
    num_cols = int(numpy.ceil(num_metrics / num_rows))
    f, ax = plt.subplots(num_rows, num_cols, figsize=(3*num_cols, 2*num_rows),
                         squeeze=False)

    for k, name in enumerate(metric_names):
        i, j = k // num_cols, k % num_cols
        # convert to numpy arrays and make each row correspond to a metric
        # from a given epoch (e.g., take the transpose)
        vals = numpy.array([m for m in metrics[name]]).T

        markersize = max(2.5, 5*min(1, 10/len(vals)))
        ax[i][j].set_title(name)

        num_epochs = len(vals)
        ax[i][j].plot(range(1,1+num_epochs), vals, 'o-',
                      markersize=markersize, c='b')

        if i+1 == num_rows:
            ax[i][j].set_xlabel("epoch")

        vmin, vmax = min(numpy.ravel(vals)), max(numpy.ravel(vals))
        if vmax < 0:
            vmin *= 1.1
            vmax = 0
        if vmin > 0:
            vmin = 0
            vmax *= 1.1
        ax[i][j].set_ylim([vmin, vmax])

    # plot the blank spots
    for k in range(len(metric_names), num_rows*num_cols):
        i, j = k // num_cols, k % num_cols
        ax[i][j].scatter([-1], [-1])
        ax[i][j].set_xlim([0, 1])
        ax[i][j].set_ylim([0, 1])
        ax[i][j].set_axis_off()

    plt.tight_layout(pad=0.5, h_pad=0.2, w_pad=0.2)
    if show:
        plt.show(f)
    if filename is not None:
        f.savefig(filename)
    plt.close(f)
