from . import backends as be
from cytoolz import identity, partial
from .models import hidden

# ----- GRADIENT ----- #

class GradientMemory(object):
    """
    Many optimizers like RMSProp or ADAM keep track of moving averages
    of the gradients. This class computes the first two moments of the
    gradients as running averages.

    """

    def __init__(self, mean_weight=0.9, mean_square_weight=0.0):

        self.mean_weight = be.float_scalar(mean_weight)
        self.mean_square_weight = be.float_scalar(mean_square_weight)

        self.mean_gradient = None
        self.mean_square_gradient = None

        from cytoolz import partial
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
            self.mean_gradient = hidden.grad_apply(identity, grad)
        else:
            hidden.grad_mapzip_(self.mixer_, self.mean_gradient, grad)

    def update_mean_square(self, grad):
        """
        Update the running average of the squared model gradients.

        Args:
            grad (a Gradient object)

        Returns:
            None

        """
        if self.mean_square_gradient is None:
            self.mean_square_gradient = hidden.grad_apply(be.square, grad)
        else:
            hidden.grad_mapzip_(self.square_mixer_, self.mean_square_gradient, grad)

    def update(self, grad):
        """
        Update the running average of the model gradients and the running
        average of the squared model gradients.

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

        return hidden.grad_mapzip(normalizer, grad, self.mean_square_gradient)


# ----- LEARNING RATE SCHEDULERS ----- #

class Scheduler(object):

    def __init__(self):
        self.lr = 1
        self.iter = 0
        self.epoch = 0

    def increment(self, epoch):
        self.iter += 1
        self.epoch = epoch


class ExponentialDecay(Scheduler):

    def __init__(self, lr_decay=0.9):
        super().__init__()
        self.lr_decay = lr_decay

    def get_lr(self):
        self.lr = (self.lr_decay ** self.epoch)
        return self.lr


class PowerLawDecay(Scheduler):

    def __init__(self, lr_decay=0.1):
        super().__init__()
        self.lr_decay = lr_decay

    def get_lr(self):
        self.lr = 1 / (1 + self.lr_decay * self.epoch)
        return self.lr


# ----- OPTIMIZERS ----- #

class Optimizer(object):

    def __init__(self,
                 scheduler=PowerLawDecay(),
                 tolerance=1e-7):
        self.scheduler = scheduler
        self.tolerance = tolerance
        self.delta = {}

    def check_convergence(self):
        mag = hidden.grad_magnitude(self.delta)
        if mag <= self.tolerance:
            return True
        else:
            return False


class StochasticGradientDescent(Optimizer):
    """StochasticGradientDescent
       Basic algorithm of gradient descent with minibatches.

    """
    def __init__(self, model,
                 stepsize=0.001,
                 scheduler=PowerLawDecay(),
                 tolerance=1e-7):
        super().__init__(scheduler, tolerance)
        self.stepsize = stepsize

    def update(self, model, v_data, v_model, epoch):
        self.scheduler.increment(epoch)
        lr_ = partial(be.tmul_,
                      be.float_scalar(self.scheduler.get_lr() * self.stepsize))

        self.delta = model.gradient(v_data, v_model)
        hidden.grad_apply_(lr_, self.delta)
        model.parameter_update(self.delta)


class Momentum(Optimizer):
    """Momentum
       Stochastic gradient descent with momentum.
       Qian, N. (1999).
       On the momentum term in gradient descent learning algorithms.
       Neural Networks, 12(1), 145–151

    """
    def __init__(self, model,
                 stepsize=0.001,
                 momentum=0.9,
                 scheduler=PowerLawDecay(),
                 tolerance=1e-7):
        super().__init__(scheduler, tolerance)
        self.stepsize = stepsize
        self.memory = GradientMemory(mean_weight=momentum,
                                     mean_square_weight=0)

    def update(self, model, v_data, v_model, epoch):
        self.scheduler.increment(epoch)
        lr = partial(be.tmul,
                      be.float_scalar(self.scheduler.get_lr() * self.stepsize))

        grad = model.gradient(v_data, v_model)
        self.memory.update(grad)
        self.delta = hidden.grad_apply(lr, self.memory.mean_gradient)
        model.parameter_update(self.delta)


class RMSProp(Optimizer):
    """RMSProp
       Geoffrey Hinton's Coursera Course Lecture 6e

    """
    def __init__(self, model,
                 stepsize=0.001,
                 mean_square_weight=0.9,
                 scheduler=PowerLawDecay(),
                 tolerance=1e-7):
        super().__init__(scheduler, tolerance)
        self.stepsize = be.float_scalar(stepsize)

        self.memory = GradientMemory(mean_weight=0,
                                     mean_square_weight=mean_square_weight)

    def update(self, model, v_data, v_model, epoch):
        self.scheduler.increment(epoch)
        lr = self.scheduler.get_lr() * self.stepsize

        grad = model.gradient(v_data, v_model)
        self.memory.update(grad)
        self.delta = self.memory.normalize(grad, unbiased=True)

        for l in self.delta['layers']:
            be.multiply_dict_inplace(l, lr)

        for l in self.delta['weights']:
            be.multiply_dict_inplace(l, lr)

        model.parameter_update(self.delta)


class ADAM(Optimizer):
    """ADAM
       Adaptive Moment Estimation algorithm.
       Kingma, D. P., & Ba, J. L. (2015).
       Adam: a Method for Stochastic Optimization.
       International Conference on Learning Representations, 1–13.

    """
    def __init__(self, model,
                 stepsize=0.001,
                 mean_weight=0.9,
                 mean_square_weight=0.999,
                 scheduler=PowerLawDecay(),
                 tolerance=1e-7):
        super().__init__(scheduler, tolerance)
        self.stepsize = be.float_scalar(stepsize)

        self.memory = GradientMemory(mean_weight=mean_weight,
                                     mean_square_weight=mean_square_weight)

    def update(self, model, v_data, v_model, epoch):
        self.scheduler.increment(epoch)
        lr = self.scheduler.get_lr() * self.stepsize

        grad = model.gradient(v_data, v_model)
        self.memory.update(grad)
        self.delta = self.memory.normalize(self.memory.mean_gradient,
                                           unbiased=True)

        for l in self.delta['layers']:
            be.multiply_dict_inplace(l, lr)

        for l in self.delta['weights']:
            be.multiply_dict_inplace(l, lr)

        model.parameter_update(self.delta)


# ----- ALIASES ----- #

sgd = SGD = StochasticGradientDescent
momentum = Momentum
rmsprop = RMSProp
adam = ADAM
