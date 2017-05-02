import time, math
from collections import OrderedDict
from . import backends as be
from . import metrics as M
from paysage.models.model import State


class Sampler(object):
    """Base class for the sequential Monte Carlo samplers"""
    def __init__(self, model, method='stochastic', **kwargs):
        """
        Create a sampler.

        Args:
            model: a model object
            method (str; optional): how to update the particles
            kwargs (optional)

        Returns:
            sampler

        """
        self.model = model
        self.pos_state = None
        self.neg_state = None

        self.method = method
        if self.method == 'stochastic':
            self.updater = self.model.markov_chain
        elif self.method == 'mean_field':
            self.updater = self.model.mean_field_iteration
        elif self.method == 'deterministic':
            self.updater = self.model.deterministic_iteration
        else:
            raise ValueError("Unknown method {}".format(self.method))

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
    def from_batch(cls, model, batch, method='stochastic', **kwargs):
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
        tmp = cls(model, method=method, **kwargs)
        vdata = batch.get('train')
        tmp.set_positive_state(State.from_visible(vdata, model))
        tmp.set_negative_state(State.from_visible(vdata, model))
        batch.reset_generator('all')
        return tmp


class SequentialMC(Sampler):
    """Basic sequential Monte Carlo sampler"""
    def __init__(self, model, method='stochastic'):
        """
        Create a sequential Monte Carlo sampler.

        Args:
            model: a model object
            method (str; optional): how to update the particles

        Returns:
            SequentialMC

        """
        super().__init__(model, method=method)

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
    def __init__(self, model, beta_momentum=0.9, beta_std=0.2,
                 method='stochastic'):
        """
        Create a sequential Monte Carlo sampler.

        Args:
            model: a model object
            beta_momentum (float in [0,1]): autoregressive coefficient of beta
            beta_std (float > 0): the standard deviation of beta
            method (str; optional): how to update the particles

        Returns:
            SequentialMC

        """
        super().__init__(model, method=method)
        self.beta_momentum = beta_momentum
        self.beta_std = beta_std
        self.beta = None
        self.has_beta = False

    def _update_beta(self):
        """
        Update beta with an AR(1) process.

        AR(1) process: X_t = momentum * X_(t-1) + loc + scale * noise
        E[X] = loc / (1 - momentum)
             -> loc = E[X] * (1 - momentum)
        Var[X] = scale ** 2 / (1 - momentum**2)
               -> scale = sqrt(Var[X] * (1 - momentum**2))

        Notes:
            Modifies the folling attributes in place:
                has_beta, beta_shape, beta_loc, beta_scale, beta

        Args:
            None

        Returns:
            None

        """
        if not self.has_beta:
            self.has_beta = True
            if self.pos_state:
                self.beta_shape = (be.shape(self.pos_state.units[0])[0], 1)
            else:
                self.beta_shape = (be.shape(self.neg_state.units[0])[0], 1)
            self.beta_loc = (1-self.beta_momentum) * be.ones(self.beta_shape)
            self.beta_scale = self.beta_std * math.sqrt(1-self.beta_momentum**2)
            self.beta = be.ones(self.beta_shape)

        self.beta *= self.beta_momentum
        self.beta += self.beta_loc
        self.beta += self.beta_scale * be.randn(self.beta_shape)

    def update_positive_state(self, steps):
        """
        Update the state of the particles.

        Notes:
            Modifies the state attribute in place.
            Calls _update_beta() method.

        Args:
            steps (int): the number of Monte Carlo steps

        Returns:
            None

        """
        if not self.pos_state:
            raise AttributeError(
                  'You must call the initialize(self, array_or_shape)'
                  +' method to set the initial state of the Markov Chain')
        self.pos_state = self.updater(steps, self.pos_state, self.beta)

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
        self._update_beta()
        self.neg_state = self.updater(steps, self.neg_state, self.beta)


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

        for m in self.metrics:
            m.reset()

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
            for m in self.metrics:
                m.update(metric_state)

        # compute metric dictionary
        metdict = OrderedDict([(m.name, m.value()) for m in self.metrics])
        if show:
            for m in metdict:
                print("-{0}: {1:.6f}".format(m, metdict[m]))

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
    sampler.update_positive_state(steps)
    sampler.update_negative_state(steps)

    # compute the gradient
    return model.gradient(*sampler.get_states())

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

    # compute the gradient
    return model.gradient(*sampler.get_states())

# alias
pcd = persistent_contrastive_divergence

def tap(vdata, model, sampler, positive_steps=1):
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
        sampler: for marginal free energy
        positive_steps: steps to sample MCMC for positive phase

    Returns:
        gradient

    """
    data_state = State.from_visible(vdata, model)
    sampler.set_positive_state(data_state)
    sampler.update_positive_state(positive_steps)
    return model.TAP_gradient(data_state, None)


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
            while True:
                try:
                    v_data = self.batch.get(mode='train')
                except StopIteration:
                    break

                self.optimizer.update(self.model,
                self.grad_approx(v_data, self.model, self.sampler, self.mcsteps),
                epoch)

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

# alias
sgd = SGD = StochasticGradientDescent
