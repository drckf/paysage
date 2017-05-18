import time
from collections import OrderedDict
from itertools import tee

from . import backends as be
from . import metrics as M
from . import schedules
from .models.model_utils import State

class Sampler(object):
    """Base class for the sequential Monte Carlo samplers"""
    def __init__(self, model, **kwargs):
        """
        Create a sampler.

        Args:
            model: a model object
            kwargs (optional)

        Returns:
            sampler

        """
        self.model = model
        self.pos_state = None
        self.neg_state = None
        self.updater = self.model.markov_chain

    def set_positive_state(self, state):
        """
        Set up the positive state for each of the Markov Chains.
        The initial state is randomly initialized.

        Notes:
            Modifies the state attribute in place.

        Args:
            shape (tuple): shape if the visible layer

        Returns:
            None

        """
        self.pos_state = state

    def set_negative_state(self, state):
        """
        Set up the initial states for each of the Markov Chains.
        The initial state is randomly initialized.

        Notes:
            Modifies the state attribute in place.

        Args:
            shape (tuple): shape if the visible layer

        Returns:
            None

        """
        self.neg_state = state

    def get_states(self):
        """
        Retrieve the states.

        """
        return self.pos_state, self.neg_state

    @classmethod
    def from_batch(cls, model, batch, **kwargs):
        """
        Create a sampler from a batch object.

        Args:
            model: a Model object
            batch: a Batch object
            method (str; optional): how to update the particles
            kwargs (optional)

        Returns:
            sampler

        """
        tmp = cls(model, **kwargs)
        vdata = batch.get('train')
        tmp.set_positive_state(State.from_visible(vdata, model))
        tmp.set_negative_state(State.from_visible(vdata, model))
        batch.reset_generator('all')
        return tmp


class SequentialMC(Sampler):
    """Basic sequential Monte Carlo sampler"""
    def __init__(self, model):
        """
        Create a sequential Monte Carlo sampler.

        Args:
            model: a model object
            method (str; optional): how to update the particles

        Returns:
            SequentialMC

        """
        super().__init__(model)

    def update_positive_state(self, steps):
        """
        Update the positive state of the particles.

        Notes:
            Modifies the state attribute in place.

        Args:
            steps (int): the number of Monte Carlo steps

        Returns:
            None

        """
        if not self.pos_state:
            raise AttributeError(
                'You must call the initialize(self, array_or_shape)'
                +' method to set the initial state of the Markov Chain')
        self.pos_state = self.updater(steps, self.pos_state)

    def update_negative_state(self, steps):
        """
        Update the negative state of the particles.

        Notes:
            Modifies the state attribute in place.

        Args:
            steps (int): the number of Monte Carlo steps

        Returns:
            None

        """
        if not self.neg_state:
            raise AttributeError(
                'You must call the initialize(self, array_or_shape)'
                +' method to set the initial state of the Markov Chain')
        self.neg_state = self.updater(steps, self.neg_state)


class DrivenSequentialMC(Sampler):
    """An accelerated sequential Monte Carlo sampler"""
    def __init__(self, model, beta_momentum=0.9, beta_std=0.6,
                 schedule=schedules.constant(initial=1.0)):
        """
        Create a sequential Monte Carlo sampler.

        Args:
            model: a model object
            beta_momentum (float in [0,1]): autoregressive coefficient of beta
            beta_std (float > 0): the standard deviation of beta
            schedule (generator; optional)

        Returns:
            DrivenSequentialMC

        """
        super().__init__(model)

        from numpy.random import gamma, poisson
        self.gamma = gamma
        self.poisson = poisson

        self.std = beta_std
        self.var = self.std**2

        self.phi = beta_momentum # autocorrelation
        self.nu = 1 / self.var # location parameter
        self.c = (1-self.phi) * self.var # scale parameter

        self.beta = None
        self.has_beta = False
        self.schedule = schedule

    def _anneal(self):
        t = next(self.schedule)
        return self.nu / t, self.c * t

    def _update_beta(self):
        """
        Update beta with an autoregressive Gamma process.

        beta_0 ~ Gamma(nu,c/(1-phi)) = Gamma(nu, var)
        h_t ~ Possion( phi/c * h_{t-1})
        beta_t ~ Gamma(nu + z_t, c)

        Achieves a stationary distribution with mean 1 and variance var:
        Gamma(nu, var) = Gamma(1/var, var)

        Notes:
            Modifies the folling attributes in place:
                has_beta, beta_shape, beta

        Args:
            None

        Returns:
            None

        """
        nu, c = self._anneal()
        if not self.has_beta:
            self.has_beta = True
            if self.pos_state:
                self.beta_shape = (be.shape(self.pos_state.units[0])[0], 1)
            else:
                self.beta_shape = (be.shape(self.neg_state.units[0])[0], 1)
            self.beta = self.gamma(nu, c/(1-self.phi), size=self.beta_shape)
        z = self.poisson(lam=self.beta * self.phi/c)
        self.beta = self.gamma(nu + z, c)

    def _beta(self):
        """Return beta in the appropriate tensor format."""
        return be.float_tensor(self.beta)

    def update_positive_state(self, steps):
        """
        Update the state of the particles.

        Notes:
            Modifies the state attribute in place.

        Args:
            steps (int): the number of Monte Carlo steps

        Returns:
            None

        """
        if not self.pos_state:
            raise AttributeError(
                'You must call the initialize(self, array_or_shape)'
                +' method to set the initial state of the Markov Chain')
        self.pos_state = self.updater(steps, self.pos_state, beta=None)

    def update_negative_state(self, steps):
        """
        Update the negative state of the particles.

        Notes:
            Modifies the state attribute in place.
            Calls _update_beta() method.

        Args:
            steps (int): the number of Monte Carlo steps

        Returns:
            None

        """
        if not self.neg_state:
            raise AttributeError(
                'You must call the initialize(self, array_or_shape)'
                +' method to set the initial state of the Markov Chain')
        for _ in range(steps):
            self._update_beta()
            self.neg_state = self.updater(1, self.neg_state, self._beta())


class ProgressMonitor(object):
    """
    Monitor the progress of training by computing statistics on the
    validation set.

    """
    def __init__(self, batch, metrics=['ReconstructionError']):
        """
        Create a progress monitor.

        Args:
            batch (int): the
            metrics (list[str]): list of metrics to compute

        Returns:
            ProgressMonitor

        """
        self.batch = batch
        self.update_steps = 10
        self.metrics = [M.__getattribute__(m)() for m in metrics]
        self.memory = []

    def check_progress(self, model, store=False, show=False):
        """
        Compute the metrics from a model on the validaiton set.

        Args:
            model: a model object
            store (bool): if true, store the metrics in a list
            show (bool): if true, print the metrics to the screen

        Returns:
            metdict (dict): an ordered dictionary with the metrics

        """
        sampler = SequentialMC(model)

        for metric in self.metrics:
            metric.reset()

        while True:
            try:
                v_data = self.batch.get(mode='validate')
            except StopIteration:
                break

            # set up the positive state
            data_state = State.from_visible(v_data, model)
            sampler.set_positive_state(data_state)
            # set up the negative state
            random_samples = model.random(v_data)
            model_state = State.from_visible(random_samples, model)
            sampler.set_negative_state(model_state)

            # update the states
            sampler.update_positive_state(1)
            sampler.update_negative_state(self.update_steps)

            metric_state = M.MetricState(minibatch=data_state,
                                         reconstructions=sampler.pos_state,
                                         random_samples=model_state,
                                         samples=sampler.neg_state,
                                         model=model)

            # update metrics
            for metric in self.metrics:
                metric.update(metric_state)

        # compute metric dictionary
        metdict = OrderedDict([(m.name, m.value()) for m in self.metrics])

        if show:
            for metric in metdict:
                print("-{0}: {1:.6f}".format(metric, metdict[metric]))

        if store:
            self.memory.append(metdict)

        return metdict


def contrastive_divergence(vdata, model, sampler, steps=1):
    """
    Compute an approximation to the likelihood gradient using the CD-k
    algorithm for approximate maximum likelihood inference.

    Hinton, Geoffrey E.
    "Training products of experts by minimizing contrastive divergence."
    Neural computation 14.8 (2002): 1771-1800.

    Carreira-Perpinan, Miguel A., and Geoffrey Hinton.
    "On Contrastive Divergence Learning."
    AISTATS. Vol. 10. 2005.

    Notes:
        Modifies the state of the sampler.
        Modifies the sampling attributes of the model's compute graph.

    Args:
        vdata (tensor): observed visible units
        model: a model object
        sampler: a sampler object
        steps (int): the number of Monte Carlo steps

    Returns:
        gradient

    """
    # build the states
    data_state = State.from_visible(vdata, model)
    model_state = State.from_visible(vdata, model)

    # CD resets the sampler from the visible data at each iteration
    sampler.set_positive_state(data_state)
    sampler.set_negative_state(model_state)
    model.graph.set_clamped_sampling([0])
    sampler.update_positive_state(steps)
    model.graph.set_clamped_sampling([])
    sampler.update_negative_state(steps)

    # compute the conditional sampling on all visible-side layers,
    # inclusive over hidden-side layers
    layer_list = range(model.num_layers)

    for i in range(1, len(layer_list) - 1):
        model.graph.set_clamped_sampling(layer_list[:i])
        sampler.update_positive_state(steps)
        sampler.update_negative_state(steps)

    # make a mean field step to copmute the expectation on the last layer
    model.graph.set_clamped_sampling(layer_list[:-1])
    grad_data_state = model.mean_field_iteration(1, sampler.pos_state)
    grad_model_state = model.mean_field_iteration(1, sampler.neg_state)

    # reset the sampling clamping
    model.graph.set_clamped_sampling([])

    # compute the gradient
    return model.gradient(grad_data_state, grad_model_state)

# alias
cd = contrastive_divergence

def persistent_contrastive_divergence(vdata, model, sampler, steps=1):
    """
    PCD-k algorithm for approximate maximum likelihood inference.

    Tieleman, Tijmen.
    "Training restricted Boltzmann machines using approximations to the
    likelihood gradient."
    Proceedings of the 25th international conference on Machine learning.
    ACM, 2008.

    Notes:
        Modifies the state of the sampler.
        Modifies the sampling attributes of the model's compute graph.

    Args:
        vdata (tensor): observed visible units
        model: a model object
        sampler: a sampler object
        steps (int): the number of Monte Carlo steps

    Returns:
        gradient

    """
    # PCD persists the state of the sampler from the previous iteration
    data_state = State.from_visible(vdata, model)
    sampler.set_positive_state(data_state)
    sampler.update_negative_state(steps)

    # step through the hidden layers, up to the last
    # for each, compute the conditional sampling on all visible-side layers,
    # inclusive over hidden-side layers
    layer_list = range(model.num_layers)

    for i in range(1, len(layer_list) - 1):
        model.graph.set_clamped_sampling(layer_list[:i])
        sampler.update_positive_state(steps)
        sampler.update_negative_state(steps)

    # make a mean field step to copmute the expectation on the last layer
    model.graph.set_clamped_sampling(layer_list[:-1])
    grad_data_state = model.mean_field_iteration(1, sampler.pos_state)
    grad_model_state = model.mean_field_iteration(1, sampler.neg_state)

    # reset the sampling clamping
    model.graph.set_clamped_sampling([])

    # compute the gradient
    return model.gradient(grad_data_state, grad_model_state)

# alias
pcd = persistent_contrastive_divergence

def tap(vdata, model, sampler=None, steps=None):
    """
    Compute the gradient using the Thouless-Anderson-Palmer (TAP)
    mean field approximation.

    Eric W Tramel, Marylou Gabrie, Andre Manoel, Francesco Caltagirone,
    and Florent Krzakala
    "A Deterministic and Generalized Framework for Unsupervised Learning
    with Restricted Boltzmann Machines"


    Args:
        vdata (tensor): observed visible units
        model: a model object
        sampler (default to None): not required
        steps (default to None): not requires

    Returns:
        gradient

    """
    data_state = State.from_visible(vdata, model)
    return model.gradient(data_state, None)


class StochasticGradientDescent(object):
    """Stochastic gradient descent with minibatches"""
    def __init__(self, model, batch, optimizer, epochs, method=pcd,
                 sampler=SequentialMC, mcsteps=1, monitor=None):
        """
        Create a StochasticGradientDescent object.

        Args:
            model: a model object
            batch: a batch object
            optimizer: an optimizer object
            epochs (int): the number of epochs
            method (optional): the method used to approximate the likelihood
                               gradient [cd, pcd, ortap]
            sampler (optional): a sampler object
            mcsteps (int, optional): the number of Monte Carlo steps per gradient
            monitor (optional): a progress monitor

        Returns:
            StochasticGradientDescent

        """
        self.model = model
        self.batch = batch
        self.epochs = epochs
        self.grad_approx = method
        self.sampler = sampler
        self.mcsteps = mcsteps
        self.optimizer = optimizer
        self.monitor = monitor

    def train(self):
        """
        Train the model.

        Notes:
            Updates the model parameters in place.

        Args:
            None

        Returns:
            None

        """
        for epoch in range(self.epochs):
            t = 0
            start_time = time.time()

            self.optimizer.update_lr()

            while True:
                try:
                    v_data = self.batch.get(mode='train')
                except StopIteration:
                    break

                self.optimizer.update(
                    self.model,
                    self.grad_approx(
                        v_data,
                        self.model,
                        self.sampler,
                        self.mcsteps
                    )
                )

                t += 1

            # end of epoch processing
            print('End of epoch {}: '.format(epoch))
            if self.monitor is not None:
                self.monitor.check_progress(self.model, store=True, show=True)

            end_time = time.time()
            print('Epoch took {0:.2f} seconds'.format(end_time - start_time),
                  end='\n\n')

            # convergence check should be part of optimizer
            is_converged = self.optimizer.check_convergence()
            if is_converged:
                print('Convergence criterion reached')
                break

        return None


    def train_layerwise(self):
        """
        Train the model layerwise.

        Notes:
            Updates the model parameters in place.

        Args:
            None

        Returns:
            None

        """
        # the main loop over layers to train
        for end_layer in range(2, self.model.num_layers+1):
            fixed_layer = end_layer - 1 if end_layer > 2 else 0
            trainable_layers = list(range(fixed_layer, end_layer))
            if end_layer > 2:
                trainable_layers = [0] + trainable_layers

            print("~~~~~~~~~~~~~~~~~~~~")
            print("layerwise training")
            print(" - training layers: {}".format(list(trainable_layers)))
            print("~~~~~~~~~~~~~~~~~~~~")

            # fork the learning rate schedule, set one copy
            lr_schedule, lr_schedule_cache = tee(self.optimizer.stepsize, 2)
            self.optimizer.stepsize = lr_schedule

            # set the compute graph attributes
            self.model.graph.set_trainable_layers(trainable_layers)

            # train in this configuration
            self.train()

            # reset the learning rate schedule
            self.optimizer.stepsize = lr_schedule_cache

        return None


# alias
sgd = SGD = StochasticGradientDescent
