import numpy, pandas, time
from . import backends as B
from . import metrics as M

# -----  CLASSES ----- #

class SequentialMC(object):
    """SequentialMC
       Simple class for a sequential Monte Carlo sampler.

    """
    def __init__(self, amodel,
                 adataframe = None,
                 size = (),
                 method='stochastic'):
        self.model = amodel
        self.method = method
        """
        if isinstance(adataframe, pandas.DataFrame):
            self.state = adataframe.as_matrix().astype(numpy.float32)
        elif isinstance(adataframe, numpy.ndarray):
            self.state = adataframe.astype(numpy.float32)
        else:
            self.state = amodel.random()
        """
        try:
            self.state = adataframe.as_matrix().astype(numpy.float32)
        except Exception:
            self.state = adataframe.astype(numpy.float32)

    @classmethod
    def from_batch(cls, amodel, abatch,
                   method='stochastic'):
        tmp = cls(amodel, abatch.get('train'), method=method)
        abatch.reset_generator('all')
        return tmp

    def update_state(self, steps):
        if self.method == 'stochastic':
            self.state = self.model.markov_chain(self.state, steps)
        elif self.method == 'mean_field':
            self.state = self.model.mean_field_iteration(self.state, steps)
        elif self.method == 'deterministic':
            self.state = self.model.deterministic_iteration(self.state, steps)
        else:
            raise ValueError("Unknown method {}".format(self.method))

    def get_state(self):
        return self.state

class DrivenSequentialMC(object):

    def __init__(self, amodel, adataframe,
                 beta_momentum=0.9,
                 beta_scale=0.2,
                 method='stochastic'):
        self.model = amodel
        self.method = method
        try:
            self.state = adataframe.as_matrix().astype(numpy.float32)
        except Exception:
            self.state = adataframe.astype(numpy.float32)

        # for an AR(1) process X_t = momentum * X_(t-1) + loc + scale * noise
        # E[X] = loc / (1 - momentum)
        #     -> loc = E[X] * (1 - momentum)
        # Var[X] = scale ** 2 / (1 - momentum**2)
        #        -> scale = sqrt(Var[X] * (1 - momentum**2))
        self.beta_momentum = beta_momentum
        self.beta_loc = (1-self.beta_momentum) * numpy.ones(
                                                (len(self.state), 1),
                                                dtype=numpy.float32
                                                )
        self.beta_scale = numpy.sqrt(1-self.beta_momentum**2) * beta_scale
        self.beta = numpy.ones((len(self.state), 1), dtype=numpy.float32)

    @classmethod
    def from_batch(cls, amodel, abatch,
                   method='stochastic'):
        tmp = cls(amodel, abatch.get('train'), method=method)
        abatch.reset_generator('all')
        return tmp

    def update_beta(self):
        """ update beta with an AR(1) process

        """
        self.beta *= self.beta_momentum
        self.beta += self.beta_loc
        self.beta += self.beta_scale * numpy.random.randn(len(self.beta),1)

    def update_state(self, steps):
        self.update_beta()
        if self.method == 'stochastic':
            self.state = self.model.markov_chain(self.state,
                                                 steps,
                                                 self.beta)
        elif self.method == 'mean_field':
            self.state = self.model.mean_field_iteration(self.state,
                                                         steps,
                                                         self.beta)
        elif self.method == 'deterministic':
            self.state = self.model.deterministic_iteration(self.state,
                                                            steps,
                                                            self.beta)
        else:
            raise ValueError("Unknown method {}".format(self.method))

    def get_state(self):
        return self.state


class TrainingMethod(object):

    def __init__(self, model, abatch, optimizer, epochs,
                 skip=100,
                 update_method='stochastic',
                 sampler='SequentialMC',
                 metrics=['ReconstructionError', 'EnergyDistance']):
        self.model = model
        self.batch = abatch
        self.epochs = epochs
        self.update_method = update_method
        #self.sampler = SequentialMC.from_batch(self.model, self.batch, method=self.update_method)
        self.sampler = DrivenSequentialMC.from_batch(self.model, self.batch,
                                                     method=self.update_method)
        self.optimizer = optimizer
        self.monitor = ProgressMonitor(skip, abatch, metrics=metrics)



class ContrastiveDivergence(TrainingMethod):
    """ContrastiveDivergence
       CD-k algorithm for approximate maximum likelihood inference.

       Hinton, Geoffrey E. "Training products of experts by minimizing contrastive divergence." Neural computation 14.8 (2002): 1771-1800.
       Carreira-Perpinan, Miguel A., and Geoffrey Hinton. "On Contrastive Divergence Learning." AISTATS. Vol. 10. 2005.

    """
    def __init__(self, model, abatch, optimizer, epochs, mcsteps,
                 skip=100,
                 update_method='stochastic',
                 sampler='SequentialMC',
                 metrics=['ReconstructionError', 'EnergyDistance']):
        super().__init__(model, abatch, optimizer, epochs, skip=skip, update_method=update_method, sampler=sampler, metrics=metrics)
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

                # CD resets the sampler from the visible data at each iteration
                self.sampler = SequentialMC(self.model, v_data, method=self.update_method)
                self.sampler.update_state(self.mcsteps)

                # compute the gradient and update the model parameters
                v_model = self.sampler.get_state()
                self.optimizer.update(self.model, v_data, v_model, epoch)
                t += 1

            # end of epoch processing
            print('End of epoch {}: '.format(epoch))
            prog = self.monitor.check_progress(self.model, 0, store=True, show=True)

            end_time = time.time()
            print('Epoch took {0:.2f} seconds'.format(end_time - start_time), end='\n\n')

            # convergence check should be part of optimizer
            is_converged = self.optimizer.check_convergence()
            if is_converged:
                print('Convergence criterion reached')
                break

        return None


class PersistentContrastiveDivergence(TrainingMethod):
    """PersistentContrastiveDivergence
       PCD-k algorithm for approximate maximum likelihood inference.

       Tieleman, Tijmen. "Training restricted Boltzmann machines using approximations to the likelihood gradient." Proceedings of the 25th international conference on Machine learning. ACM, 2008.

    """
    def __init__(self, model, abatch, optimizer, epochs, mcsteps,
                 skip=100,
                 update_method='stochastic',
                 sampler='SequentialMC',
                 metrics=['ReconstructionError', 'EnergyDistance']):
       super().__init__(model, abatch, optimizer, epochs, skip=skip, update_method=update_method, sampler=sampler, metrics=metrics)
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

                # PCD keeps the sampler from the previous iteration
                self.sampler.update_state(self.mcsteps)

                # compute the gradient and update the model parameters
                self.optimizer.update(self.model, v_data, self.sampler.state, epoch)
                t += 1

            # end of epoch processing
            print('End of epoch {}: '.format(epoch))
            prog = self.monitor.check_progress(self.model, 0, store=True, show=True)

            end_time = time.time()
            print('Epoch took {0:.2f} seconds'.format(end_time - start_time), end='\n\n')

            # convergence check should be part of optimizer
            is_converged = self.optimizer.check_convergence()
            if is_converged:
                print('Convergence criterion reached')
                break

        return None


class HopfieldContrastiveDivergence(TrainingMethod):
    """HopfieldContrastiveDivergence
       Algorithm for approximate maximum likelihood inference based on the intuition that the weights of the network are stored as memories, like in the Hopfield model of associate memory.

       Unpublished. Charles K. Fisher (2016)

    """
    def __init__(self, model, abatch, optimizer, epochs,
                 attractive=True,
                 skip=100,
                 metrics=['ReconstructionError', 'EnergyDistance']):
        super().__init__(model, abatch, optimizer, epochs, skip=skip, metrics=metrics)
        self.attractive = attractive

    def train(self):
        for epoch in range(self.epochs):
            t = 0
            start_time = time.time()
            while True:
                try:
                    v_data = self.batch.get(mode='train')
                except StopIteration:
                    break

                # sample near the weights
                v_model = self.model.layers['visible'].prox(self.attractive * self.model.params['weights']).T
                # compute the gradient and update the model parameters
                self.optimizer.update(self.model, v_data, v_model, epoch)
                t += 1

            # end of epoch processing
            print('End of epoch {}: '.format(epoch))
            prog = self.monitor.check_progress(self.model, 0, store=True, show=True)

            end_time = time.time()
            print('Epoch took {0:.2f} seconds'.format(end_time - start_time), end='\n\n')

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

            for m in self.metrics:
                m.reset()

            while True:
                try:
                    v_data = self.batch.get(mode='validate')
                except StopIteration:
                    break

                # compute the reconstructions
                sampler = SequentialMC(model, v_data)
                sampler.update_state(1)
                reconstructions = sampler.state

                # compute the fantasy particles
                random_samples = model.random(v_data)
                sampler = SequentialMC(model, random_samples)
                sampler.update_state(self.update_steps)
                fantasy_particles = sampler.state

                # compile argdict
                argdict = {
                'minibatch': v_data,
                'reconstructions': reconstructions,
                'random_samples': random_samples,
                'samples': fantasy_particles,
                'amodel': model
                }

                # update metrics
                for m in self.metrics:
                    m.update(**argdict)

            # compute metric dictionary
            metdict = {m.name: m.value() for m in self.metrics}
            if show:
                for m in metdict:
                    print("-{0}: {1:.6f}".format(m, metdict[m]))

            if store:
                self.memory.append(metdict)

            return metdict

# ----- ALIASES ----- #

CD = ContrastiveDivergence
PCD = PersistentContrastiveDivergence
HCD = HopfieldContrastiveDivergence
