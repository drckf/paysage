import time, math
from collections import OrderedDict
from . import backends as be
from . import metrics as M


#TODO: should import the State class from model.py

# -----  CLASSES ----- #

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
        self.state = None
        self.has_state = False

        self.method = method
        if self.method == 'stochastic':
            self.updater = self.model.markov_chain
        elif self.method == 'mean_field':
            self.updater = self.model.mean_field_iteration
        elif self.method == 'deterministic':
            self.updater = self.model.deterministic_iteration
        else:
            raise ValueError("Unknown method {}".format(self.method))

    #TODO: use State
    # should use hidden.State object
    def randomize_state(self, shape):
        """
        Set up the inital states for each of the Markov Chains.
        The initial state is randomly initalized.

        Notes:
            Modifies the state attribute in place.

        Args:
            shape (tuple): shape if the visible layer

        Returns:
            None

        """
        self.state = self.model.random(shape)
        self.has_state = True

    #TODO: use State
    # should use hidden.State object
    def set_state(self, tensor):
        """
        Set up the inital states for each of the Markov Chains.

        Notes:
            Modifies state attribute in place.

        Args:
            tensor: the observed visible units

        Returns:
            None

        """
        self.state = be.float_tensor(tensor)
        self.has_state = True

    #TODO: use State
    # should use hidden.State object
    @classmethod
    def from_batch(cls, model, batch, method='stochastic', **kwargs):
        """
        Create a sampler from a batch object.

        Args:
            model: a model object
            batch: a batch object
            method (str; optional): how to update the particles
            kwargs (optional)

        Returns:
            sampler

        """
        tmp = cls(model, method=method, **kwargs)
        tmp.set_state(batch.get('train'))
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

    #TODO: use State
    # should use hidden.State object
    def update_state(self, steps):
        """
        Update the state of the particles.

        Notes:
            Modifies the state attribute in place.

        Args:
            steps (int): the number of Monte Carlo steps

        Returns:
            None

        """
        if not self.has_state:
            raise AttributeError(
                  'You must set the initial state of the Markov Chain')
        self.state = self.updater(self.state, steps)

    #TODO: use State
    # should use hidden.State object
    def get_state(self):
        """
        Return the state attribute.

        Args:
            None

        Returns:
            state (tensor)

        """
        return self.state


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
            self.beta_shape = (len(self.state), 1)
            self.beta_loc = (1-self.beta_momentum) * be.ones(self.beta_shape)
            self.beta_scale = self.beta_std * math.sqrt(1-self.beta_momentum**2)
            self.beta = be.ones(self.beta_shape)

        self.beta *= self.beta_momentum
        self.beta += self.beta_loc
        self.beta += self.beta_scale * be.randn(self.beta_shape)

    #TODO: use State
    # should use hidden.State object
    def update_state(self, steps):
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
        if not self.has_state:
            raise AttributeError(
                  'You must call the initialize(self, array_or_shape)'
                  +' method to set the initial state of the Markov Chain')
        self._update_beta()
        self.state = self.updater(self.state, steps, self.beta)

    #TODO: use State
    # should use hidden.State object
    def get_state(self):
        """
        Return the state attribute.

        Args:
            None

        Returns:
            state (tensor)

        """
        return self.state


class TrainingMethod(object):
    """Base training method class"""
    def __init__(self, model, batch, optimizer, sampler, epochs, skip=100,
                 metrics=['ReconstructionError', 'EnergyDistance']):
        """
        Create a training method object.

        Args:
            model: a model object
            batch: a batch object
            optimizer: an optimizer object
            sampler: a sampler object
            epochs (int): the number of epochs
            skip (int): the number of minibatches between metric calculations
            metrics (list[str]): list of metrics to compute

        Returns:
            TrainingMethod

        """
        self.model = model
        self.batch = batch
        self.epochs = epochs
        self.sampler = sampler
        self.optimizer = optimizer
        self.monitor = ProgressMonitor(skip, batch, metrics=metrics)


class ContrastiveDivergence(TrainingMethod):
    """
    CD-k algorithm for approximate maximum likelihood inference.

    Hinton, Geoffrey E.
    "Training products of experts by minimizing contrastive divergence."
    Neural computation 14.8 (2002): 1771-1800.

    Carreira-Perpinan, Miguel A., and Geoffrey Hinton.
    "On Contrastive Divergence Learning."
    AISTATS. Vol. 10. 2005.

    """
    def __init__(self, model, batch, optimizer, sampler, epochs, mcsteps=1,
                 skip=100, metrics=['ReconstructionError', 'EnergyDistance']):
        """
        Create a contrastive divergence object.

        Args:
            model: a model object
            batch: a batch object
            optimizer: an optimizer object
            sampler: a sampler object
            epochs (int): the number of epochs
            skip (int): the number of minibatches between metric calculations
            metrics (list[str]): list of metrics to compute

        Returns:
            ContrastiveDivergence

        """
        super().__init__(model, batch, optimizer, sampler, epochs, skip=skip,
                        metrics=metrics)
        self.mcsteps = mcsteps

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

                #TODO: use State
                # should use hidden.State objects
                # note that we will need two states
                # one for the positive phase (with visible units as observed)
                # one for the negative phase (with visible units sampled from the model)

                # CD resets the sampler from the visible data at each iteration
                self.sampler.set_state(v_data)
                self.sampler.update_state(self.mcsteps)

                # compute the gradient and update the model parameters
                v_model = self.sampler.get_state()
                grad = self.model.gradient(v_data, v_model)
                self.optimizer.update(self.model, grad, epoch)
                t += 1

            # end of epoch processing
            print('End of epoch {}: '.format(epoch))
            self.monitor.check_progress(self.model, 0, store=True, show=True)

            end_time = time.time()
            print('Epoch took {0:.2f} seconds'.format(end_time - start_time),
                  end='\n\n')

            # convergence check should be part of optimizer
            is_converged = self.optimizer.check_convergence()
            if is_converged:
                print('Convergence criterion reached')
                break

        return None


class PersistentContrastiveDivergence(TrainingMethod):
    """
    PCD-k algorithm for approximate maximum likelihood inference.

    Tieleman, Tijmen.
    "Training restricted Boltzmann machines using approximations to the
    likelihood gradient."
    Proceedings of the 25th international conference on Machine learning.
    ACM, 2008.

    """
    def __init__(self, model, abatch, optimizer, sampler, epochs, mcsteps=1,
                 skip=100, metrics=['ReconstructionError', 'EnergyDistance']):
        """
        Create a persistent contrastive divergence object.

        Args:
            model: a model object
            batch: a batch object
            optimizer: an optimizer object
            sampler: a sampler object
            epochs (int): the number of epochs
            skip (int): the number of minibatches between metric calculations
            metrics (list[str]): list of metrics to compute

        Returns:
            PeristentContrastiveDivergence

        """
        super().__init__(model, abatch, optimizer, sampler, epochs, skip=skip,
                        metrics=metrics)
        self.mcsteps = mcsteps

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

                #TODO: use State
                # should use hidden.State objects
                # note that we will need two states
                # one for the positive phase (with visible units as observed)
                # one for the negative phase (with visible units sampled from the model)

                # PCD keeps the sampler from the previous iteration
                self.sampler.update_state(self.mcsteps)

                # compute the gradient and update the model parameters
                v_model = self.sampler.get_state()
                grad = self.model.gradient(v_data, v_model)
                self.optimizer.update(self.model, grad, epoch)
                t += 1

            # end of epoch processing
            print('End of epoch {}: '.format(epoch))
            self.monitor.check_progress(self.model, 0, store=True, show=True)

            end_time = time.time()
            print('Epoch took {0:.2f} seconds'.format(end_time - start_time),
                  end='\n\n')

            # convergence check should be part of optimizer
            is_converged = self.optimizer.check_convergence()
            if is_converged:
                print('Convergence criterion reached')
                break

        return None

class StochasticGradientDescent(TrainingMethod):
    """Stochastic gradient descent with minibatches"""
    def __init__(self, model, batch, optimizer, epochs, skip=100,
                 metrics=['ReconstructionError', 'EnergyDistance']):
        """
        Create a StochasticGradientDescent object.

        Args:
            model: a model object
            batch: a batch object
            optimizer: an optimizer object
            sampler: a sampler object
            epochs (int): the number of epochs
            skip (int): the number of minibatches between metric calculations
            metrics (list[str]): list of metrics to compute

        Returns:
            StochasticGradientDescent

        """
        super().__init__(model, batch, optimizer, None, epochs, skip=skip,
                        metrics=metrics)

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

                # compute the gradient and update the model parameters
                v_model = None
                grad = self.model.gradient(v_data, v_model)
                self.optimizer.update(self.model, grad, epoch)
                t += 1

            # end of epoch processing
            print('End of epoch {}: '.format(epoch))
            self.monitor.check_progress(self.model, 0, store=True, show=True)

            end_time = time.time()
            print('Epoch took {0:.2f} seconds'.format(end_time - start_time),
                  end='\n\n')

            # convergence check should be part of optimizer
            is_converged = self.optimizer.check_convergence()
            if is_converged:
                print('Convergence criterion reached')
                break

        return None


class ProgressMonitor(object):
    """
    Monitor the progress of training by computing statistics on the
    validation set.

    """
    def __init__(self, skip, batch,
                 metrics=['ReconstructionError', 'EnergyDistance']):
        """
        Create a progress monitor.

        Args:
            skip (int): how often to compute the updates
            batch (int): the
            metrics (list[str]): list of metrics to compute

        Returns:
            ProgressMonitor

        """
        self.skip = skip
        self.batch = batch
        self.update_steps = 10
        self.metrics = [M.__getattribute__(m)() for m in metrics]
        self.memory = []

    def check_progress(self, model, t, store=False, show=False):
        """
        Compute the metrics from a model on the validaiton set.

        Args:
            model: a model object
            t (int): the iteration number
            store (bool): if true, store the metrics in a list
            show (bool): if true, print the metrics to the screen

        Returns:
            metdict (dict): an ordered dictionary with the metrics

        """
        if not self.skip or not (t % self.skip):

            sampler = SequentialMC(model)

            for m in self.metrics:
                m.reset()

            while True:
                try:
                    v_data = self.batch.get(mode='validate')
                except StopIteration:
                    break

                #TODO: use State
                # should use hidden.State objects
                # note that we will need two states
                # one for the positive phase (with visible units as observed)
                # one for the fantasy particles (with visible units sampled from the model)

                # compute the reconstructions
                sampler.set_state(v_data)
                sampler.update_state(1)
                reconstructions = sampler.state

                # compute the fantasy particles
                random_samples = model.random(v_data)
                sampler.set_state(random_samples)
                sampler.update_state(self.update_steps)
                fantasy_particles = sampler.state

                metric_state = M.MetricState(minibatch=v_data,
                                             reconstructions=reconstructions,
                                             random_samples=random_samples,
                                             samples=fantasy_particles,
                                             amodel=model)

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

# ----- ALIASES ----- #

CD = ContrastiveDivergence
PCD = PersistentContrastiveDivergence
sgd = SGD = StochasticGradientDescent
