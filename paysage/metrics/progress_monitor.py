import os
import numpy
import pandas
from collections import OrderedDict

from . import generator_metrics as M
from . import plotting
from .model_assessment import ModelAssessment

from .. import backends as be

class ProgressMonitor(object):
    """
    Monitor the progress of training by computing statistics on the
    validation set.

    """
    def __init__(self, generator_metrics = [M.ReconstructionError(),
                                            M.EnergyCoefficient(),
                                            M.HeatCapacity(),
                                            M.WeightSparsity(),
                                            M.WeightSquare(),
                                            M.KLDivergence(),
                                            M.ReverseKLDivergence()]
                 ):
        """
        Create a progress monitor.

        Args:
            metrics (list[metric object]): list of metrics objects to compute with

        Returns:
            ProgressMonitor

        """
        self.generator_metrics = generator_metrics

        self.metdict = {}
        self.memory = []
        self.save_conditions = []

    def reset_metrics(self):
        """
        Reset the state of the metrics.

        Notes:
            Modifies the metrics in place!

        Args:
            None

        Returns:
            None

        """
        for metric in self.generator_metrics:
            metric.reset()


    def batch_update(self, assessment):
        """
        Update the metrics on a batch.

        Args:
            assessment (ModelAssessment)

        Returns:
            None

        """
        # update generative metrics
        for metric in self.generator_metrics:
            try:
                metric.update(assessment)
            except Exception:
                pass

    def _get_metric_dict(self):
        """
        Get the metrics in dictionary form.

        Args:
            None

        Returns:
            OrderedDict

        """
        metrics = [(m.name, m.value())
                   for m in self.generator_metrics]
        return OrderedDict(metrics)

    def get_metric_dict(self, filter_none=True):
        """
        Get the metrics in dictionary form.

        Args:
            filter_none (bool): remove none values from metric output

        Returns:
            OrderedDict

        """
        metdict = self._get_metric_dict()
        return OrderedDict((key, metdict[key]) for key in metdict
                    if metdict[key] is not None or filter_none == False)

    def epoch_update(self, batch, generator, fantasy_steps=10,
                     store=False, show=False, filter_none=True, reset=True):
        """
        Outputs metric stats, and returns the metric dictionary

        Args:
            batch (paysage.batch object): data batcher
            generator (paysage.models model): generative model
            fantasy_steps (int): num steps to sample generator for fantasy particles
            store (bool): if true, store the metrics in a list
                and check if the model should be saved
            show (bool): if true, print the metrics to the screen
            filter_none (bool): remove none values from metric output
            reset (bool): reset the metrics on epoch update

        Returns:
            metdict (dict): an ordered dictionary with the metrics

        """
        # update the generator and classifier metrics
        batch.reset_generator(mode='validate')
        while True:
            try:
                v_data = batch.get(mode='validate')
            except StopIteration:
                break

            assessment = ModelAssessment(v_data, generator,
                                         fantasy_steps=fantasy_steps)

            self.batch_update(assessment)

        # compute metric dictionary
        self.metdict = self.get_metric_dict(filter_none)

        if show:
            for metric in self.metdict:
                try:
                    print("-{0}: {1:.6f}".format(metric, self.metdict[metric]))
                except TypeError:
                    print("-{0}: TypeError".format(metric))

            print("")

        # store the metrics for later
        if store:
            self.memory.append(self.metdict)
            # check if the model should be saved
            self.check_save_conditions(generator)

        # reset the metrics
        if reset:
            self.reset_metrics()

        return self.metdict

    def save_best(self, filename, metric, extremum="min"):
        """
        Save the model when a given metric is extremal.
        The filename will have the extremum and metric name appended,
            e.g. "_min_EnergyCoefficient".

        Notes:
            Modifies save_conditions in place.

        Args:
            filename (str): the filename.
            metric (str): the metric name.
            extremum (str): "min" or "max"

        Returns:
            None

        """
        assert (extremum in ["min", "max"])
        filename_parts = os.path.splitext(filename)
        model_filename = filename_parts[0] + "_{}_{}".format(extremum, metric) \
                            + filename_parts[1]
        def save(model):
            extremal_func = pandas.Series.idxmin \
                                if extremum == "min" else pandas.Series.idxmax
            metrics = pandas.DataFrame(self.memory)
            num_epochs = len(metrics)
            do_save = \
                (extremal_func(metrics[metric].apply(numpy.mean)) == num_epochs-1)
            if do_save:
                print("Epoch {}: Saving model as {}.  Best value of {}.\n".format(
                        num_epochs, model_filename, metric))
                store = pandas.HDFStore(model_filename, "w")
                model.save(store)
                store.put("metrics", pandas.DataFrame(metrics))
                store.close()
        self.save_conditions.append(save)

    def save_every(self, filename, epoch_period=1):
        """
        Save the model every N epochs.
        The filename will have "_epoch<N>" appended.

        Notes:
            Modifies save_conditions in place.

        Args:
            filename (str): the filename. "_epoch<N>" will be appended.
            epoch_period (int): the period for saving the model. For example,
                if epoch_period=2, the model is saved on epochs 2, 4, 6, ...

        Returns:
            None

        """
        filename_parts = os.path.splitext(filename)
        filename_template = filename_parts[0] + "_epoch{}" + filename_parts[1]
        def save(model):
            metrics = pandas.DataFrame(self.memory)
            num_epochs = len(metrics)
            if num_epochs % epoch_period == 0:
                model_filename = filename_template.format(num_epochs)
                print("Epoch {}: Saving model as {}. Periodic save.\n".format(
                        num_epochs, model_filename))
                store = pandas.HDFStore(model_filename, "w")
                model.save(store)
                store.put("metrics", metrics)
                store.close()
        self.save_conditions.append(save)

    def check_save_conditions(self, model):
        """
        Checks any save conditions.
        Each check will save the model if it passes.

        Args:
            model (paysage.models model): generative model

        Returns:
            None

        """
        for check_save in self.save_conditions:
            check_save(model)

    def plot_metrics(self, filename=None, show=True):
        """
        Plot the metric memory.

        Args:
            filename (optional; str)
            show (optional; bool)

        Returns:
            None

        """
        plotting.plot_metrics(self.memory, filename=filename, show=show)
