from . import backends as be
from . import schedules
from .models import state as model_state

class AutoregressiveGammaSampler(object):
    """Sampler from an autoregressive Gamma process."""
    def __init__(self, beta_momentum=0.9, beta_std=0.6,
                 schedule=schedules.Constant(initial=1.0)):
        """
        Create an autoregressive gamma sampler.
        Can be used to sample inverse temperatures for MC sampling.

        Args:
            beta_momentum (float in [0,1]; optional): autoregressive coefficient
                the inverse temperature, beta.
            beta_std (float >= 0; optional): the standard deviation of the
                inverse temperature, beta.
            schedule (generator; optional)

        Returns:
            An AutoregressiveGammaSampler instance.

        """
        from numpy.random import gamma, poisson
        self.gamma = gamma
        self.poisson = poisson

        self.schedule = schedule
        self.set_std(beta_std, beta_momentum)

    def set_std(self, std, momentum=0.9):
        """
        Set the parameters based off the standard deviation.

        Notes:
            Modifies many layer attributes in place!

        Args:
            std (float)

        Returns:
            None

        """
        self.std = std
        self.var = self.std**2
        self.use_driven = (self.std > 0)

        if self.use_driven:
            self.phi = momentum # autocorrelation
            self.nu = 1 / self.var # location parameter
            self.c = (1-self.phi) * self.var # scale parameter

        self.beta = None
        self.has_beta = False
        self.schedule.reset()

    def set_schedule(self, value):
        """
        Change the value of the learning rate schedule.

        Notes:
            Modifies the schedule.value attribute in place!

        Args:
            value (float)

        Returns:
            None

        """
        self.schedule.set_value(value)

    def _anneal(self):
        """
        Get the next value from the learning rate schedule and update
        the parameters of the Gamma generating process.

        Args:
            None

        Return:
            nu (float), c (float)

        """
        t = be.EPSILON + next(self.schedule)
        return self.nu / t, self.c * t

    def update_beta(self, num_samples):
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
            num_samples (int): the number of samples to generate for beta

        Returns:
            None

        """
        if self.use_driven:
            nu, c = self._anneal()
            if not self.has_beta or self.beta_shape[0] != num_samples:
                self.has_beta = True
                self.beta_shape = (num_samples, 1)
                self.beta = self.gamma(nu, c/(1-self.phi), size=self.beta_shape)
            z = self.poisson(lam=self.beta * self.phi/c)
            self.beta = self.gamma(nu + z, c)

    def get_beta(self):
        """Return beta in the appropriate tensor format."""
        if self.use_driven:
            return be.float_tensor(self.beta)
        return None


class SequentialMC(object):
    """An accelerated sequential Monte Carlo sampler"""
    def __init__(self, model, mcsteps=1, clamped=None, updater='markov_chain',
                 beta_momentum=0.9, beta_std=0.6,
                 schedule=schedules.Constant(initial=1.0)):
        """
        Create a sequential Monte Carlo sampler.

        Args:
            model (BoltzmannMachine)
            mcsteps (int; optional): the number of Monte Carlo steps
            clamped (List[int]; optional): list of layers to clamp
            updater (str; optional): method for updating the state
            beta_momentum (float in [0,1]; optional): autoregressive coefficient
                the inverse temperature of beta
            beta_std (float >= 0; optional): the standard deviation of the
                inverse temperature beta
            schedule (generator; optional)

        Returns:
            SequentialMC

        """
        self.model = model
        self.state = None
        self.update_method = updater
        self.updater = getattr(model, updater)
        self.mcsteps = mcsteps

        self.clamped = []
        if clamped is not None:
            self.clamped = clamped

        self.beta_sampler = AutoregressiveGammaSampler(beta_momentum,
                                                       beta_std,
                                                       schedule)

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

    def set_state_from_visible(self, vdata):
        """
        Set the state of the sampler using a sample of visible vectors.

        Notes:
            Modifies the sampler.state attribute in place.

        Args:
            vdata (tensor~(num_samples,num_units)): a visible state

        Returns:
            None

        """
        self.set_state(model_state.State.from_visible(vdata, self.model))

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
        self.beta_sampler.beta = None

    def update_state(self, steps=None):
        """
        Update the state of the particles.

        Notes:
            Modifies the state attribute in place.
            Calls the beta_sampler.update_beta() method.

        Args:
            steps (int): the number of Monte Carlo steps

        Returns:
            None

        """
        if not self.state:
            raise AttributeError(
                'You must call the initialize(self, array_or_shape)'
                +' method to set the initial state of the Markov Chain')
        STEPS = self.mcsteps if steps is None else steps
        for _ in range(STEPS):
            self.beta_sampler.update_beta(be.shape(self.state[0])[0])
            clamping = self.model.clamped_sampling
            self.model.set_clamped_sampling(self.clamped)
            self.state = self.updater(1, self.state, beta=self.beta_sampler.get_beta())
            self.model.set_clamped_sampling(clamping)

    def state_for_grad(self, target_layer):
        """
        Peform a mean field update of the target layer.

        Args:
            target_layer (int): the layer to update

        Returns:
            state

        """
        layer_list = range(self.model.num_layers)
        clamping = self.model.clamped_sampling
        self.model.set_clamped_sampling([i for i in layer_list if i != target_layer])
        grad_state = self.model.mean_field_iteration(1, self.state)
        self.model.set_clamped_sampling(clamping)
        return grad_state

    @classmethod
    def from_batch(cls, model, batch, **kwargs):
        """
        Create a sampler from a batch object.

        Args:
            model: a BoltzmannMachine object
            batch: a Batch object
            kwargs (optional)

        Returns:
            sampler

        """
        tmp = cls(model, **kwargs)
        tmp.set_state_from_visible(batch.get('train'))
        batch.reset_generator('all')
        return tmp

    @classmethod
    def from_visible(cls, model, vdata, **kwargs):
        """
        Create a sampler initialized from visible data.

        Args:
            model: a BoltzmannMachine object
            vdata: visible data
            kwargs (optional)

        Returns:
            sampler

        """
        tmp = cls(model, **kwargs)
        tmp.set_state_from_visible(vdata)
        return tmp

    @classmethod
    def from_model(cls, model, batch_size, **kwargs):
        """
        Create a sampler from a model object.

        Args:
            model: a BoltzmannMachine object
            batch_size: the batch size
            kwargs (optional)

        Returns:
            sampler

        """
        tmp = cls(model, **kwargs)
        tmp.set_state(model_state.State.from_model_envelope(
                batch_size, model))
        return tmp

    @classmethod
    def generate_fantasy_state(cls, model, batch_size, update_steps,
                               **kwargs):
        """
        Generate fantasy particles from a model.

        Args:
            model: A BoltzmannMachine object.
            batch_size (int): the number of fantasy particles to generate
            update_steps (int): how many monte carlo steps to run
            kwargs: other keyword arguments for SequentialMC

        Returns:
            State

        """
        sampler = cls.from_model(model, batch_size, **kwargs)
        sampler.update_state(update_steps)
        return sampler.state
