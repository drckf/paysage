import time
from collections import OrderedDict

from . import backends as be
from . import metrics as M
from . import schedules
from .models.model_utils import State
from . import layers
from.models.model import Model

class Sampler(object):
    """Base class for the sequential Monte Carlo samplers"""
    def __init__(self, model, updater='markov_chain', **kwargs):
        """
        Create a sampler.

        Args:
            model: a model object
            kwargs (optional)

        Returns:
            sampler

        """
        self.model = model
        self.state = None
        self.clamped = []
        self.update_method = updater
        self.updater = getattr(model, updater)

    def set_state(self, state):
        """
        Set the state.

        Notes:
            Modifies the state attribute in place.

        Args:
            state (State): The state of the units.

        Returns:
            None

        """
        self.state = state

    def set_state_from_batch(self, batch):
        """
        Set the state of the sampler using a sample of visible vectors.

        Notes:
            Modifies the sampler.state attribute in place.

        Args:
            batch: a Batch object

        Returns:
            None

        """
        vdata = batch.get('train')
        self.set_state(State.from_visible(vdata, self.model))
        batch.reset_generator('all')

    @classmethod
    def from_batch(cls, model, batch, **kwargs):
        """
        Create a sampler from a batch object.

        Args:
            model: a Model object
            batch: a Batch object
            kwargs (optional)

        Returns:
            sampler

        """
        tmp = cls(model, **kwargs)
        tmp.set_state_from_batch(batch)
        return tmp


class SequentialMC(Sampler):
    """Basic sequential Monte Carlo sampler"""
    def __init__(self, model, clamped=None, updater='markov_chain'):
        """
        Create a sequential Monte Carlo sampler.

        Args:
            model: a model object
            method (str; optional): how to update the particles

        Returns:
            SequentialMC

        """
        super().__init__(model, updater=updater)
        if clamped is not None:
            self.clamped = clamped

    def update_state(self, steps, dropout_mask=None):
        """
        Update the positive state of the particles.

        Notes:
            Modifies the state attribute in place.

        Args:
            steps (int): the number of Monte Carlo steps
            dropout_mask (optioonal; State): mask on model units; 1=on, 0=dropped

        Returns:
            None

        """
        if not self.state:
            raise AttributeError(
                'You must call the initialize(self, array_or_shape)'
                +' method to set the initial state of the Markov Chain')
        clamping = self.model.graph.clamped_sampling
        self.model.graph.set_clamped_sampling(self.clamped)
        self.state = self.updater(steps, self.state, dropout_mask)
        self.model.graph.set_clamped_sampling(clamping)

    def state_for_grad(self, target_layer, dropout_mask=None):
        """
        Peform a mean field update of the target layer.

        Args:
            target_layer (int): the layer to update
            dropout_mask (State): mask on model units

        Returns:
            state

        """
        layer_list = range(self.model.num_layers)
        clamping = self.model.graph.clamped_sampling
        self.model.graph.set_clamped_sampling([i for i in layer_list if i != target_layer])
        grad_state = self.model.mean_field_iteration(1, self.state, dropout_mask)
        self.model.graph.set_clamped_sampling(clamping)
        return grad_state

    def reset(self):
        """
        Reset the sampler state.

        Notes:
            Modifies sampler.state attribute in place.

        Args:
            None

        Returns:
            None

        """
        self.state = None


class DrivenSequentialMC(Sampler):
    """An accelerated sequential Monte Carlo sampler"""
    def __init__(self, model, clamped=None, updater='markov_chain', beta_momentum=0.9,
                 beta_std=0.6, schedule=schedules.Constant(initial=1.0)):
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
        super().__init__(model, updater=updater)
        if clamped is not None:
            self.clamped = clamped

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
            self.beta_shape = (be.shape(self.state.units[0])[0], 1)
            self.beta = self.gamma(nu, c/(1-self.phi), size=self.beta_shape)
        z = self.poisson(lam=self.beta * self.phi/c)
        self.beta = self.gamma(nu + z, c)

    def _beta(self):
        """Return beta in the appropriate tensor format."""
        return be.float_tensor(self.beta)

    def update_state(self, steps, dropout_mask=None):
        """
        Update the state of the particles.

        Notes:
            Modifies the state attribute in place.
            Calls _update_beta() method.

        Args:
            steps (int): the number of Monte Carlo steps
            dropout_mask (State object): mask on model units
                for positive phase dropout, 1: on 0: dropped-out

        Returns:
            None

        """
        if not self.state:
            raise AttributeError(
                'You must call the initialize(self, array_or_shape)'
                +' method to set the initial state of the Markov Chain')
        for _ in range(steps):
            self._update_beta()
            clamping = self.model.graph.clamped_sampling
            self.model.graph.set_clamped_sampling(self.clamped)
            self.state = self.updater(1, self.state, dropout_mask, beta=self._beta())
            self.model.graph.set_clamped_sampling(clamping)

    def state_for_grad(self, target_layer, dropout_mask=None):
        """
        Peform a mean field update of the target layer.

        Args:
            target_layer (int): the layer to update
            dropout_mask (State): mask on model units

        Returns:
            state

        """
        layer_list = range(self.model.num_layers)
        clamping = self.model.graph.clamped_sampling
        self.model.graph.set_clamped_sampling([i for i in layer_list if i != target_layer])
        grad_state = self.model.mean_field_iteration(1, self.state, dropout_mask)
        self.model.graph.set_clamped_sampling(clamping)
        return grad_state

    def reset(self):
        """
        Reset the sampler state.

        Notes:
            Modifies sampler.state attribute in place.

        Args:
            None

        Returns:
            None

        """
        self.state = None
        self.beta = None
        self.has_beta = False


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
        dropout_scale = State.dropout_rescale(model)

        for metric in self.metrics:
            metric.reset()

        while True:
            try:
                v_data = self.batch.get(mode='validate')
            except StopIteration:
                break

            # compute the reconstructions
            reconstructions = SequentialMC(model)
            data_state = State.from_visible(v_data, model)
            reconstructions.set_state(data_state)
            reconstructions.update_state(1, dropout_scale)

            # compute the fantasy particles
            fantasy = SequentialMC(model)
            random_state = State.from_visible(model.random(v_data), model)
            fantasy.set_state(random_state)
            fantasy.update_state(self.update_steps, dropout_scale)

            metric_state = M.MetricState(minibatch=data_state,
                                         reconstructions=reconstructions.state,
                                         random_samples=random_state,
                                         samples=fantasy.state,
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

def contrastive_divergence(vdata, model, positive_phase, negative_phase,
                           positive_dropout=None, steps=1):
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
        positive_phase: a sampler object
        negative_phase: a sampler object
        positive_dropout (State object): mask on model units
         for dropout 1: on 0: dropped out
        steps (int): the number of Monte Carlo steps

    Returns:
        gradient

    """
    target_layer = model.num_layers - 1

    # compute the update of the positive phase
    data_state = State.from_visible(vdata, model)
    positive_phase.set_state(data_state)
    positive_phase.update_state(steps, positive_dropout)
    grad_data_state = positive_phase.state_for_grad(target_layer, positive_dropout)

    # CD resets the sampler from the visible data at each iteration
    model_state = State.from_visible(vdata, model)
    negative_phase.set_state(model_state)
    negative_phase.update_state(steps, positive_dropout)
    grad_model_state = negative_phase.state_for_grad(target_layer, positive_dropout)

    # compute the gradient
    return model.gradient(grad_data_state, grad_model_state, positive_dropout, positive_dropout)

# alias
cd = contrastive_divergence

def persistent_contrastive_divergence(vdata, model, positive_phase, negative_phase,
                                      positive_dropout=None, steps=1):
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
        positive_phase: a sampler object
        negative_phase: a sampler object
        positive_dropout (State object): mask on model units for positive phase dropout
         1: on 0: dropped-out
        steps (int): the number of Monte Carlo steps

    Returns:
        gradient

    """
    target_layer = model.num_layers - 1

    # compute the update of the positive phase
    data_state = State.from_visible(vdata, model)
    positive_phase.set_state(data_state)
    positive_phase.update_state(steps, positive_dropout)
    grad_data_state = positive_phase.state_for_grad(target_layer, positive_dropout)

    # PCD persists the state of the sampler from the previous iteration
    dropout_scale = State.dropout_rescale(model)
    negative_phase.update_state(steps, dropout_scale)
    grad_model_state = negative_phase.state_for_grad(target_layer, dropout_scale)

    return model.gradient(grad_data_state, grad_model_state, positive_dropout, dropout_scale)

# alias
pcd = persistent_contrastive_divergence

def tap(vdata, model, positive_phase, negative_phase=None, steps=1, init_lr_EMF=0.1,
        tolerance_EMF=1e-4, max_iters_EMF=25, positive_dropout=None):
    """
    Compute the gradient using the Thouless-Anderson-Palmer (TAP)
    mean field approximation.

    Slight modifications on the methods in

    Eric W Tramel, Marylou Gabrie, Andre Manoel, Francesco Caltagirone,
    and Florent Krzakala
    "A Deterministic and Generalized Framework for Unsupervised Learning
    with Restricted Boltzmann Machines"

    Args:
        vdata (tensor): observed visible units
        model: a model object
        positive_phase: a sampler object
        negative_phase (unused; default=None): a sampler object
        steps: steps to sample MCMC for positive phase
        positive_dropout (State object): mask on model units for positive phase dropout
         1: on 0: dropped-out

        TAP free energy computation parameters:
            init_lr float: initial learning rate which is halved whenever necessary to enforce descent.
            tol float: tolerance for quitting minimization.
            max_iters: maximum gradient decsent steps

    Returns:
        gradient object

    """
    # compute the positive phase
    data_state = State.from_visible(vdata, model)
    positive_phase.set_state(data_state)
    positive_phase.update_state(steps, positive_dropout)

    grad_data_state = positive_phase.state_for_grad(model.num_layers-1, positive_dropout)

    return model.TAP_gradient(grad_data_state, init_lr_EMF, tolerance_EMF,
                              max_iters_EMF, positive_dropout)


class StochasticGradientDescent(object):
    """Stochastic gradient descent with minibatches"""
    def __init__(self, model, batch, optimizer, epochs, sampler, method=pcd,
                 mcsteps=1, monitor=None):
        """
        Create a StochasticGradientDescent object.

        Args:
            model: a model object
            batch: a batch object
            optimizer: an optimizer object
            epochs (int): the number of epochs
            sampler: a sampler object
            method (optional): the method used to approximate the likelihood
                               gradient [cd, pcd, tap]
            mcsteps (int, optional): the number of Monte Carlo steps per gradient
            monitor (optional): a progress monitor

        Returns:
            StochasticGradientDescent

        """
        self.model = model
        self.batch = batch
        self.epochs = epochs
        self.grad_approx = method

        self.positive_phase = SequentialMC.from_batch(model, batch,
                                                      updater=sampler.update_method,
                                                      clamped=[0])
        self.negative_phase = sampler

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
            start_time = time.time()
            self.optimizer.update_lr()

            while True:
                try:
                    v_data = self.batch.get(mode='train')
                except StopIteration:
                    break

                # the dropout mask is fixed for each batch
                self.optimizer.update(
                    self.model,
                    self.grad_approx(
                        v_data,
                        self.model,
                        self.positive_phase,
                        self.negative_phase,
                        positive_dropout=State.dropout_mask(self.model, be.shape(v_data)[0]),
                        steps=self.mcsteps
                    )
                )

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

class LayerwisePretrain(object):
    """
    Pretrain a model in layerwise fashion using the method from:

    "Deep Boltzmann Machines" by Ruslan Salakhutdinov and Geoffrey Hinton

    """
    def __init__(self, model, batch, optimizer, epochs, method=pcd,
                 mcsteps=1, metrics=None):
        """
        Create a LayerwisePretrain object.

        Args:
            model: a model object
            batch: a batch object
            optimizer: an optimizer object
            epochs (int): the number of epochs
            method (optional): the method used to approximate the likelihood
                               gradient [cd, pcd, tap]
            mcsteps (int, optional): the number of Monte Carlo steps per gradient
            metrics (List, optional): a list of metrics

        Returns:
            LayerwisePretrain

        """
        self.model = model
        self.batch = batch
        self.epochs = epochs
        self.grad_approx = method
        self.mcsteps = mcsteps
        self.optimizer = optimizer
        self.metrics = metrics

    def _create_transform(self, model, basic_transform):
        """
        Closure that creates a transform function from a model.

        Args:
            model (Model)

        Returns:
            transform (callable)

        """
        def transform(data):
            # create a state and dropout mask
            state = State.from_visible(basic_transform(data), model)
            dropout = State.dropout_rescale(model)
            # cache the model clamping
            clamping = model.graph.clamped_sampling
            # clamp the visible units
            model.graph.set_clamped_sampling([0])
            # perform a mean field iteration
            state = model.mean_field_iteration(1, state, dropout)
            # reset the model clamping
            model.graph.set_clamped_sampling(clamping)
            # return the units of the last layer
            return state.units[-1]
        return transform

    def _copy_params_from_submodels(self, submodels):
        """
        Copy the parameters from a list of submodels into a single model.

        Notes:
            Modifies the parameters fo the model attribute in place.

        Args:
            submodels (List[Model]): a stack of RBMs

        Returns:
            None

        """
        # copy the params from the zeroth layer
        for j in range(len(submodels[0].layers[0].params)):
            self.model.layers[0].params[j][:] = submodels[0].layers[0].params[j]
        # copy the rest of the layer and weight parameters
        for i in range(len(submodels)):
            for j in range(len(submodels[i].layers[1].params)):
                self.model.layers[i+1].params[j][:] = submodels[i].layers[1].params[j]
            if (i == 0) or (i == len(submodels)-1):
                # keep the weights of the zeroth layer and the last layer
                self.model.weights[i].params.matrix[:] = submodels[i].weights[0].W()
            else:
                # halve the weights of the other layers
                self.model.weights[i].params.matrix[:] = 0.5 * submodels[i].weights[0].W()

    def train(self):
        """
        Train the model layerwise.

        Notes:
            Updates the model parameters in place.

        Args:
            None

        Returns:
            None

        """
        # create the submodels
        submodels = [Model(
                [layers.layer_from_config(self.model.layers[i].get_config()),
                 layers.layer_from_config(self.model.layers[i+1].get_config())],
                 [layers.weights_from_config(self.model.weights[i].get_config())]
                ) for i in range(self.model.num_layers-1)]

        # set the multipliers to double the effect of the visible and the last
        # hidden layers
        submodels[0].multipliers = be.float_tensor([2,1])
        submodels[-1].multipliers = be.float_tensor([1,2])

        # cache the learning rate and the transform
        lr_schedule_cache = self.optimizer.stepsize.copy()
        transform_cache = self.batch.transform

        for i in range(len(submodels)):
            print('training model {}'.format(i), end="\n\n")

            # update the transform
            basic_transform = self.batch.transform
            if i > 0:
                self.batch.transform = self._create_transform(submodels[i-1], basic_transform)
                # set the parameters of the zeroth layer using the
                # parameters of the first layer of the previous model
                submodels[i].layers[0].set_params(submodels[i-1].layers[1].params)
                submodels[i].layers[0].set_fixed_params(list(submodels[i-1].layers[1].params._fields))

            # initialize the submodel
            submodels[i].initialize(self.batch, method="glorot_normal")

            # set up a progress monitor
            perf = ProgressMonitor(self.batch, metrics=self.metrics)

            # reset the state of the optimizer
            self.optimizer.reset()
            self.optimizer.stepsize = lr_schedule_cache

            # set up a sampler
            sampler = DrivenSequentialMC.from_batch(submodels[i], self.batch)
            trainer = StochasticGradientDescent(submodels[i], self.batch, self.optimizer,
                                                self.epochs, sampler,
                                                method=self.grad_approx,
                                                mcsteps=self.mcsteps,
                                                monitor=perf)
            trainer.train()

        # reset the transform
        self.batch.transform = transform_cache

        # update the model
        self._copy_params_from_submodels(submodels)
