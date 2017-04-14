from . import backends as be
from cytoolz import identity, partial
from .models import gradient_util as gu

# ----- CLASSES ----- #

class GradientMemory(object):
    """
    Many optimizers like RMSProp or ADAM keep track of moving averages
    of the gradients. This class computes the first two moments of the
    gradients as running averages.

    """

    def __init__(self, mean_weight=0.9, mean_square_weight=0.0):
        """
        Create a gradient memory object to keep track of the first two
        moments of the gradient.

        Args:
            mean_weight (float \in (0,1); optional):
                how strongly to weight the previous gradient
            mean_square_weight (float \in (0,1); optional)
                how strongly to weight the square of the previous gradient

        Returns:
            GradientMemory

        """
        self.mean_weight = be.float_scalar(mean_weight)
        self.mean_square_weight = be.float_scalar(mean_square_weight)

        self.mean_gradient = None
        self.mean_square_gradient = None

        self.mixer_ = partial(be.mix_inplace, self.mean_weight)
        self.square_mixer_ = partial(be.square_mix_inplace, self.mean_square_weight)

    def update_mean(self, grad):
        """
        Update the running average of the model gradients.

        Args:
            grad (a Gradient object)

        Returns:
            None

        """
        if self.mean_gradient is None:
            self.mean_gradient = gu.grad_apply(identity, grad)
        else:
            gu.grad_mapzip_(self.mixer_, self.mean_gradient, grad)

    def update_mean_square(self, grad):
        """
        Update the running average of the squared model gradients.

        Args:
            grad (a Gradient object)

        Returns:
            None

        """
        if self.mean_square_gradient is None:
            self.mean_square_gradient = gu.grad_apply(be.square, grad)
        else:
            gu.grad_mapzip_(self.square_mixer_, self.mean_square_gradient, grad)

    def update(self, grad):
        """
        Update the running average of the model gradients and the running
        average of the squared model gradients.

        Notes:
            Modifies mean_weight and mean_square_weight attributes in place.

        Args:
            grad (a Gradient object)

        Returns:
            None

        """
        if self.mean_weight:
            self.update_mean(grad)
        if self.mean_square_weight:
            self.update_mean_square(grad)

    def normalize(self, grad, unbiased=False):
        """
        Divide grad by the square root of the mean square gradient.

        Notes:
            A running average is biased due to autoregressive correlations
            between adjacent timepoints. The bias can be corrected by
            dividing the results by appropriate weights that reflect
            the degree of autocorrelation.

            Acts like the identity function if mean_square_weight = 0.

        Args:
            grad (a Gradient object)
            unbiased (bool): whether to unbias the estimates

        Returns:
            normalized Gradient object

        """
        if not self.mean_square_gradient:
            return grad

        if unbiased:
            mean_norm = be.float_scalar(1 - self.mean_weight)
            mean_square_norm = be.float_scalar(1 - self.mean_square_weight)
            def normalizer(mean, mean_square):
                return be.sqrt_div(mean / mean_norm,
                                   mean_square / mean_square_norm)
        else:
            def normalizer(mean, mean_square):
                return be.sqrt_div(mean, mean_square)

        return gu.grad_mapzip(normalizer, grad, self.mean_square_gradient)



class Scheduler(object):
    """Base class for the learning rate schedulers"""
    def __init__(self):
        """
        Create a scheduler object.

        Args:
            None

        Returns:
            Scheduler

        """
        self.lr = 1
        self.iter = 0
        self.epoch = 0

    def increment(self, epoch):
        """
        Update the iter and epoch attributes.

        Notes:
            Modifies iter and epoch attributes in place.

        Args:
            epoch (int): the current epoch

        Returns:
            None

        """
        self.iter += 1
        self.epoch = epoch


class ExponentialDecay(Scheduler):
    """Learning rate that decays exponentially per epoch"""
    def __init__(self, lr_decay=0.9):
        """
        Create an exponential decay learning rate schedule.
        Larger lr_decay -> slower decay.

        Args:
            lr_decay (float \in (0,1))

        Returns:
            ExponentialDecay

        """
        super().__init__()
        self.lr_decay = lr_decay

    def get_lr(self):
        """
        Compute the current value of the learning rate.

        Args:
            None

        Returns:
            lr (float)

        """
        self.lr = (self.lr_decay ** self.epoch)
        return self.lr


class PowerLawDecay(Scheduler):
    """Learning rate that decays with a power law per epoch"""
    def __init__(self, lr_decay=0.1):
        """
        Create a power law decay learning rate schedule.
        Larger lr_decay -> faster decay.

        Args:
            lr_decay (float \in (0,1))

        Returns:
            PowerLawDecay

        """
        super().__init__()
        self.lr_decay = lr_decay

    def get_lr(self):
        """
        Compute the current value of the learning rate.

        Args:
            None

        Returns:
            lr (float)

        """
        self.lr = 1 / (1 + self.lr_decay * self.epoch)
        return self.lr



class Optimizer(object):
    """Base class for the optimizer methods."""
    def __init__(self, scheduler=PowerLawDecay(), tolerance=1e-7):
        """
        Create an optimizer object:

        Args:
            scheduler (a learning rate schedule object; optional)
            tolerance (float; optional):
                the gradient magnitude to declar convergence

        Returns:
            Optimizer

        """
        self.scheduler = scheduler
        self.tolerance = tolerance
        self.delta = {}

    def check_convergence(self):
        """
        Check the convergence criterion.

        Args:
            None

        Returns:
            bool: True if converged, False if not
        """
        mag = gu.grad_magnitude(self.delta)
        return mag <= self.tolerance


class Gradient(Optimizer):
    """Vanilla gradient optimizer"""
    def __init__(self,
                 stepsize=0.001,
                 scheduler=PowerLawDecay(),
                 tolerance=1e-7,
                 ascent=False):
        """
        Create a gradient ascent/descent optimizer.

        Aliases:
            gradient

        Args:
            model: a Model object to optimize
            stepsize (float; optional): the initial stepsize
            scheduler (a learning rate scheduler object; optional)
            tolerance (float; optional):
                the gradient magnitude to declar convergence

        Returns:
            StochasticGradientDescent

        """
        super().__init__(scheduler, tolerance)
        self.stepsize = stepsize
        if (ascent):
            self.grad_multiplier = -1.0
        else:
            self.grad_multiplier = 1.0

    def update(self, model, v_data, v_model, epoch):
        """
        Update the model parameters with a gradient step.

        Notes:
            Changes parameters of model in place.

        Args:
            model: a Model object to optimize
            v_data (tensor): observations
            v_mdoel (tensor): samples from the model
            epoch (int): the current epoch

        Returns:
            None

        """
        self.scheduler.increment(epoch)
        lr_ = partial(be.tmul_,
                      be.float_scalar(self.grad_multiplier * self.scheduler.get_lr() * self.stepsize))

        self.delta = model.gradient(v_data, v_model)
        gu.grad_apply_(lr_, self.delta)
        model.parameter_update(self.delta)

class Momentum(Optimizer):
    """
    Stochastic gradient descent with momentum.
    Qian, N. (1999).
    On the momentum term in gradient descent learning algorithms.
    Neural Networks, 12(1), 145–151

    """
    def __init__(self,
                 stepsize=0.001,
                 momentum=0.9,
                 scheduler=PowerLawDecay(),
                 tolerance=1e-7,
                 ascent=False):
        """
        Create a stochastic gradient descent with momentum optimizer.

        Aliases:
            momentum

        Args:
            model: a Model object to optimize
            stepsize (float; optional): the initial stepsize
            momentum (float; optional): the amount of momentum
            scheduler (a learning rate scheduler object; optional)
            tolerance (float; optional):
                the gradient magnitude to declar convergence

        Returns:
            Momentum

        """
        super().__init__(scheduler, tolerance)
        self.stepsize = stepsize
        self.memory = GradientMemory(mean_weight=momentum,
                                     mean_square_weight=0)
        if (ascent):
            self.grad_multiplier = -1.0
        else:
            self.grad_multiplier = 1.0

    def update(self, model, v_data, v_model, epoch):
        """
        Update the model parameters with a gradient step.

        Notes:
            Changes parameters of model in place.

        Args:
            model: a Model object to optimize
            v_data (tensor): observations
            v_mdoel (tensor): samples from the model
            epoch (int): the current epoch

        Returns:
            None

        """
        self.scheduler.increment(epoch)
        lr = partial(be.tmul,
                      be.float_scalar(self.grad_multiplier * self.scheduler.get_lr() * self.stepsize))

        grad = model.gradient(v_data, v_model)
        self.memory.update(grad)
        self.delta = gu.grad_apply(lr, self.memory.mean_gradient)
        model.parameter_update(self.delta)


class RMSProp(Optimizer):
    """
    Stochastic gradient descent with RMSProp.
    Geoffrey Hinton's Coursera Course Lecture 6e

    """
    def __init__(self,
                 stepsize=0.001,
                 mean_square_weight=0.9,
                 scheduler=PowerLawDecay(),
                 tolerance=1e-7,
                 ascent=False):
        """
        Create a stochastic gradient descent with RMSProp optimizer.

        Aliases:
            rmsprop

        Args:
            model: a Model object to optimize
            stepsize (float; optional): the initial stepsize
            mean_square_weight (float; optional):
                for computing the running average of the mean-square gradient
            scheduler (a learning rate scheduler object; optional)
            tolerance (float; optional):
                the gradient magnitude to declar convergence

        Returns:
            RMSProp

        """
        super().__init__(scheduler, tolerance)
        self.stepsize = be.float_scalar(stepsize)

        self.memory = GradientMemory(mean_weight=0,
                                     mean_square_weight=mean_square_weight)
        if (ascent):
            self.grad_multiplier = -1.0
        else:
            self.grad_multiplier = 1.0

    def update(self, model, v_data, v_model, epoch):
        """
        Update the model parameters with a gradient step.

        Notes:
            Changes parameters of model in place.

        Args:
            model: a Model object to optimize
            v_data (tensor): observations
            v_mdoel (tensor): samples from the model
            epoch (int): the current epoch

        Returns:
            None

        """
        self.scheduler.increment(epoch)
        lr_ = partial(be.tmul_,
                      be.float_scalar(self.grad_multiplier * self.scheduler.get_lr() * self.stepsize))

        grad = model.gradient(v_data, v_model)
        self.memory.update(grad)
        self.delta = self.memory.normalize(grad, unbiased=True)
        gu.grad_apply_(lr_, self.delta)
        model.parameter_update(self.delta)


class ADAM(Optimizer):
    """
    Stochastic gradient descent with Adaptive Moment Estimation algorithm.

    Kingma, D. P., & Ba, J. L. (2015).
    Adam: a Method for Stochastic Optimization.
    International Conference on Learning Representations, 1–13.

    """
    def __init__(self,
                 stepsize=0.001,
                 mean_weight=0.9,
                 mean_square_weight=0.999,
                 scheduler=PowerLawDecay(),
                 tolerance=1e-7,
                 ascent=False):
        """
        Create a stochastic gradient descent with ADAM optimizer.

        Aliases:
            adam

        Args:
            model: a Model object to optimize
            stepsize (float; optional): the initial stepsize
            mean_weight (float; optional):
                for computing the running average of the mean gradient
            mean_square_weight (float; optional):
                for computing the running average of the mean-square gradient
            scheduler (a learning rate scheduler object; optional)
            tolerance (float; optional):
                the gradient magnitude to declar convergence

        Returns:
            ADAM

        """
        super().__init__(scheduler, tolerance)
        self.stepsize = be.float_scalar(stepsize)

        self.memory = GradientMemory(mean_weight=mean_weight,
                                     mean_square_weight=mean_square_weight)
        if (ascent):
            self.grad_multiplier = -1.0
        else:
            self.grad_multiplier = 1.0

    def update(self, model, v_data, v_model, epoch):
        """
        Update the model parameters with a gradient step.

        Notes:
            Changes parameters of model in place.

        Args:
            model: a Model object to optimize
            v_data (tensor): observations
            v_mdoel (tensor): samples from the model
            epoch (int): the current epoch

        Returns:
            None

        """
        self.scheduler.increment(epoch)
        lr_ = partial(be.tmul_,
                      be.float_scalar(self.grad_multiplier * self.scheduler.get_lr() * self.stepsize))

        grad = model.gradient(v_data, v_model)
        self.memory.update(grad)
        self.delta = self.memory.normalize(self.memory.mean_gradient,
                                           unbiased=True)
        gu.grad_apply_(lr_, self.delta)
        model.parameter_update(self.delta)


# ----- ALIASES ----- #

gradient = Gradient
momentum = Momentum
rmsprop = RMSProp
adam = ADAM
