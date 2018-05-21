import time
import numpy as np

from .. import backends as be
from .. import metrics as M
from .. import samplers
from . import methods

class StochasticGradientDescent(object):
    """Stochastic gradient descent with minibatches"""
    def __init__(self, model, batch, fantasy_steps=10):
        """
        Create a StochasticGradientDescent object.

        Args:
            model: a model object
            batch: a batch object
            fantasy_steps (int): the number of steps for fantasy particles
                in the progress monitor.

        Returns:
            StochasticGradientDescent

        """
        self.model = model
        self.batch = batch
        self.fantasy_steps = fantasy_steps
        self.monitor = M.ProgressMonitor()

    def train(self, optimizer, num_epochs, mcsteps=1, update_method='markov_chain',
              method=methods.pcd, beta_std=0.6, negative_phase_batch_size=None,
              verbose=True, burn_in=0):
        """
        Train the model.

        Notes:
            Updates the model parameters in place.

        Args:
            optimizer: an optimizer object
            num_epochs (int): the number of epochs
            mcsteps (int; optional): the number of Monte Carlo steps per gradient
            update_method (str; optional): the method used to update the state
                [markov_chain, deterministic_iteration, mean_field_iteration]
            method (fit.methods obj; optional): the method used to approximate the likelihood
                               gradient [cd, pcd, tap]
            beta_std (float; optional): the standard deviation of the inverse
                temperature of the SequentialMC sampler
            negative_phase_batch_size (int; optional): the batch size for the negative phase.
                If None, matches the positive_phase batch size.
            verbose (bool; optional): print output to stdout
            burn_in (int; optional): the number of initial epochs during which
                the beta_std will be set to 0

        Returns:
            None

        """
        neg_batch_size = negative_phase_batch_size \
            if negative_phase_batch_size is not None else self.batch.output_batch_size

        positive_phase = samplers.SequentialMC.from_batch(self.model,
                                                          self.batch,
                                                          updater=update_method,
                                                          clamped=[0],
                                                          beta_std=0,
                                                          mcsteps=mcsteps)

        negative_phase = samplers.SequentialMC.from_model(self.model,
                                                          neg_batch_size,
                                                          updater=update_method,
                                                          beta_std=0,
                                                          mcsteps=mcsteps)

        be.maybe_print('Before training:', verbose=verbose)
        if self.monitor is not None:
            self.monitor.epoch_update(self.batch, self.model,
                                      fantasy_steps=self.fantasy_steps,
                                      store=False, show=verbose)

        for epoch in range(1, 1+num_epochs):
            time_last = time.time()

            if epoch > burn_in:
                 negative_phase.beta_sampler.set_std(beta_std)

            optimizer.update_lr()

            while True:
                try:
                    v_data = self.batch.get(mode='train')
                except StopIteration:
                    break

                optimizer.update(self.model,
                    method(v_data, self.model, positive_phase, negative_phase)
                )

            # end of epoch processing
            time_now = time.time()
            iteration_time = time_now - time_last
            time_last = time_now
            be.maybe_print('End of epoch {}: '.format(epoch), verbose=verbose)
            be.maybe_print('Time elapsed {}s'.format(np.around(iteration_time, 3)),
                           verbose=verbose)

            if self.monitor is not None:
                self.monitor.epoch_update(self.batch, self.model,
                                          fantasy_steps=self.fantasy_steps,
                                          store=True, show=verbose)

            # convergence check should be part of optimizer
            is_converged = optimizer.check_convergence()
            if is_converged:
                be.maybe_print('Convergence criterion reached', verbose=verbose)
                break

        return None

# alias
SGD = StochasticGradientDescent
