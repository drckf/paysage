import time, math
from collections import OrderedDict
from . import backends as be
from . import metrics as M
from paysage.models.model import State


# -----  CLASSES ----- #

class Sampler(object):
    """
    A base class for the sequential Monte Carlo samplers

    """
    def __init__(self, amodel,
                 method='stochastic',
                 **kwargs):
        self.model = amodel
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

        """
        self.pos_state = state

    def set_negative_state(self, state):
        """
        Set up the negative state for each of the Markov Chains.

        """
        self.neg_state = state

    def get_states(self):
        """
        Retrieve the states.

        """
        return self.pos_state, self.neg_state

    @classmethod
    def from_batch(cls, amodel, abatch,
                   method='stochastic',
                   **kwargs):
        """
        Build a Sampler object from a Model, Batch object, and update method.

        Args:
            amodel (Model): a model
            abatch (Batch): a data batcher
            method (str): an update method

        Returns:
            A Sampler object

        """
        tmp = cls(amodel, method=method, **kwargs)
#        tmp.set_state(abatch.get('train')) # TODO: fix
        abatch.reset_generator('all')
        return tmp


class SequentialMC(Sampler):
    """
    Simple class for a sequential Monte Carlo sampler.

    """
    def __init__(self, amodel,
                 method='stochastic'):
        super().__init__(amodel, method=method)

    def update_state(self, steps):
        if not (self.pos_state and self.neg_state):
            raise AttributeError(
                  'You must call the initialize(self, array_or_shape)'
                  +' method to set the initial state of the Markov Chain')
        self.pos_state = self.updater(steps, self.pos_state)
        self.neg_state = self.updater(steps, self.neg_state)


class DrivenSequentialMC(Sampler):

    def __init__(self, amodel,
                 beta_momentum=0.9,
                 beta_scale=0.2,
                 method='stochastic'):
        super().__init__(amodel, method=method)
        self.beta_momentum = beta_momentum
        self.beta_scale = beta_scale
        self.has_beta = False

    def update_beta(self):
        """
        Update beta with an AR(1) process

        """
        if not self.has_beta:
            self.has_beta = True
            # AR(1) process: X_t = momentum * X_(t-1) + loc + scale * noise
            # E[X] = loc / (1 - momentum)
            #     -> loc = E[X] * (1 - momentum)
            # Var[X] = scale ** 2 / (1 - momentum**2)
            #        -> scale = sqrt(Var[X] * (1 - momentum**2))
            self.beta_shape = (len(self.pos_state.units[0]), 1)
            self.beta_loc = (1-self.beta_momentum) * be.ones(self.beta_shape)
            self.beta_scale *= math.sqrt(1-self.beta_momentum**2)
            self.beta = be.ones(self.beta_shape)

        self.beta *= self.beta_momentum
        self.beta += self.beta_loc
        self.beta += self.beta_scale * be.randn(self.beta_shape)

    def update_state(self, steps):
        if not (self.pos_state and self.neg_state):
            raise AttributeError(
                  'You must call the initialize(self, array_or_shape)'
                  +' method to set the initial state of the Markov Chain')
        self.update_beta()
        self.pos_state = self.updater(steps, self.pos_state, self.beta)
        self.neg_state = self.updater(steps, self.neg_state, self.beta)


class TrainingMethod(object):

    def __init__(self, model, abatch, optimizer, sampler, epochs,
                 skip=100,
                 metrics=['ReconstructionError', 'EnergyDistance']):
        self.model = model
        self.batch = abatch
        self.epochs = epochs
        self.sampler = sampler
        self.optimizer = optimizer
        self.monitor = ProgressMonitor(skip, abatch,
                                       metrics=metrics)


class ContrastiveDivergence(TrainingMethod):
    """
    ContrastiveDivergence
    CD-k algorithm for approximate maximum likelihood inference.

    Hinton, Geoffrey E.
    "Training products of experts by minimizing contrastive divergence."
    Neural computation 14.8 (2002): 1771-1800.

    Carreira-Perpinan, Miguel A., and Geoffrey Hinton.
    "On Contrastive Divergence Learning."
    AISTATS. Vol. 10. 2005.

    """
    def __init__(self, model, abatch, optimizer, sampler, epochs,
                 mcsteps=1,
                 skip=100,
                 metrics=['ReconstructionError', 'EnergyDistance']):
        super().__init__(model, abatch, optimizer, sampler, epochs,
                        skip=skip,
                        metrics=metrics)
        self.mcsteps = mcsteps

    def train(self):
        for epoch in range(self.epochs):
            t = 0
            start_time = time.time()
            while True:
                try:
                    v_data = self.batch.get(mode='train')
                except StopIteration:
                    break

                # build the states
                data_state = State.from_visible(v_data, self.model)
                model_state = State.from_visible(v_data, self.model)

                # CD resets the sampler from the visible data at each iteration
                self.sampler.set_positive_state(data_state)
                self.sampler.set_negative_state(model_state)
                self.sampler.update_state(self.mcsteps)

                # compute the gradient and update the model parameters
                data_state, model_state = self.sampler.get_states()
                self.optimizer.update(self.model, data_state, model_state, epoch)
                t += 1

            # end of epoch processing
            print('End of epoch {}: '.format(epoch))
            prog = self.monitor.check_progress(self.model, 0,
                                               store=True,
                                               show=True)

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
    """PersistentContrastiveDivergence
       PCD-k algorithm for approximate maximum likelihood inference.

       Tieleman, Tijmen.
       "Training restricted Boltzmann machines using approximations to the
       likelihood gradient."
       Proceedings of the 25th international conference on Machine learning.
       ACM, 2008.

    """
    def __init__(self, model, abatch, optimizer, sampler, epochs,
                 mcsteps=1,
                 skip=100,
                 metrics=['ReconstructionError', 'EnergyDistance']):
       super().__init__(model, abatch, optimizer, sampler, epochs,
                        skip=skip,
                        metrics=metrics)
       self.mcsteps = mcsteps

    def train(self):
        # build the negative state
        model_state = State.from_model(self.batch.batch_size, self.model)
        self.sampler.set_negative_state(model_state)

        for epoch in range(self.epochs):
            t = 0
            start_time = time.time()
            while True:
                try:
                    v_data = self.batch.get(mode='train')
                except StopIteration:
                    break

                # build the positive state
                data_state = State.from_visible(v_data, self.model)

                # PCD keeps the sampler from the previous iteration
                self.sampler.set_positive_state(data_state)
                self.sampler.update_state(self.mcsteps)

                # compute the gradient and update the model parameters
                data_state, model_state = self.sampler.get_states()
                self.optimizer.update(self.model, data_state, model_state, epoch)
                t += 1

            # end of epoch processing
            print('End of epoch {}: '.format(epoch))
            prog = self.monitor.check_progress(self.model, 0,
                                               store=True,
                                               show=True)

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
    """
    Stochastic gradient descent with minibatches
    """
    def __init__(self, model, abatch, optimizer, epochs,
                 skip=100,
                 metrics=['ReconstructionError', 'EnergyDistance']):
        super().__init__(model, abatch, optimizer, None, epochs,
                        skip=skip,
                        metrics=metrics)

    def train(self):
        for epoch in range(self.epochs):
            t = 0
            start_time = time.time()
            while True:
                try:
                    v_data = self.batch.get(mode='train')
                except StopIteration:
                    break

                # compute the gradient and update the model parameters
                self.optimizer.update(self.model, v_data, None, epoch)
                t += 1

            # end of epoch processing
            print('End of epoch {}: '.format(epoch))
            prog = self.monitor.check_progress(self.model, 0,
                                               store=True,
                                               show=True)

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

    def __init__(self, skip, batch,
                 metrics=['ReconstructionError', 'EnergyDistance']):
        self.skip = skip
        self.batch = batch
        self.update_steps = 10
        self.metrics = [M.__getattribute__(m)() for m in metrics]
        self.memory = []

    def check_progress(self, model, t,
                       store=False,
                       show=False):
        if not self.skip or not (t % self.skip):

            sampler = SequentialMC(model)

            for m in self.metrics:
                m.reset()

            while True:
                try:
                    v_data = self.batch.get(mode='validate')
                except StopIteration:
                    break

                # set up the states
                data_state = State.from_visible(v_data, model)
                sampler.set_positive_state(data_state)
                random_samples = model.random(v_data)
                model_state = State.from_visible(random_samples, model)
                sampler.set_negative_state(model_state)

                # compute the reconstructions
                sampler.update_state(1)
                reconstructions = sampler.pos_state.units[0]

                # compute the fantasy particles
                random_samples = model.random(v_data)
                model_state = State.from_visible(random_samples, model)
                sampler.set_negative_state(model_state)
                sampler.update_state(self.update_steps)
                fantasy_particles = sampler.neg_state.units[0]

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
