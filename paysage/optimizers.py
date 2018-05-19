from copy import deepcopy

from cytoolz import identity, partial

from . import backends as be
from .models import gradient_util as gu
from . import schedules

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

        self.mixer_ = partial(be.mix_, self.mean_weight)
        self.square_mixer_ = partial(be.square_mix_, self.mean_square_weight)


    def reset(self):
        """
        Reset the accululated mean and mean square gradients.

        Notes:
            Modifies mean_gradient and mean_square_gradient in place.

        Args:
            None

        Returns:
            None

        """
        self.mean_gradient = None
        self.mean_square_gradient = None

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


class Optimizer(object):
    """Base class for the optimizer methods."""
    def __init__(self,
                 stepsize=schedules.Constant(initial=0.001),
                 tolerance=1e-7):
        """
        Create an optimizer object:

        Args:
            model: a BoltzmannMachine object to optimize
            stepsize (generator; optional): the stepsize schedule
            tolerance (float; optional):
                the gradient magnitude to declar convergence

        Returns:
            Optimizer

        """
        self.stepsize = stepsize
        self.tolerance = tolerance
        self.delta = {}
        self.lr_ = partial(be.tmul_, stepsize)

    def check_convergence(self):
        """
        Check the convergence criterion.

        Args:
            None

        Returns:
            bool: True if converged, False if not
        """
        mag = gu.grad_rms(self.delta)
        return mag <= self.tolerance

    def update_lr(self):
        """
        Update the current value of the stepsize:

        Notes:
            Modifies stepsize attribute in place.

        Args:
            None

        Returns:
            None

        """
        lr = be.float_scalar(next(self.stepsize))
        self.lr_ = partial(be.tmul_, lr)


class Gradient(Optimizer):
    """Vanilla gradient optimizer"""
    def __init__(self,
                 stepsize=schedules.Constant(initial=0.001),
                 tolerance=1e-7):
        """
        Create a gradient descent optimizer.

        Aliases:
            gradient

        Args:
            model: a BoltzmannMachine object to optimize
            stepsize (generator; optional): the stepsize schedule
            tolerance (float; optional):
                the gradient magnitude to declar convergence

        Returns:
            StochasticGradientDescent

        """
        super().__init__(stepsize, tolerance)

    def reset(self):
        """
        Reset the gradient memory (does nothing for vanilla gradient).

        Notes:
            Modifies gradient memory in place.

        Args:
            None

        Returns:
            None

        """
        pass

    def update(self, model, grad):
        """
        Update the model parameters with a gradient step.

        Notes:
            Changes parameters of model in place.

        Args:
            model: a BoltzmannMachine object to optimize
            grad: a Gradient object
            epoch (int): the current epoch

        Returns:
            None

        """
        self.delta = deepcopy(grad)
        gu.grad_apply_(self.lr_, self.delta)
        model.parameter_update(self.delta)


class Momentum(Optimizer):
    """
    Stochastic gradient descent with momentum.
    Qian, N. (1999).
    On the momentum term in gradient descent learning algorithms.
    Neural Networks, 12(1), 145–151

    """
    def __init__(self,
                 stepsize=schedules.Constant(initial=0.001),
                 momentum=0.9,
                 tolerance=1e-7):
        """
        Create a stochastic gradient descent with momentum optimizer.

        Aliases:
            momentum

        Args:
            model: a BoltzmannMachine object to optimize
            stepsize (generator; optional): the stepsize schedule
            momentum (float; optional): the amount of momentum
            tolerance (float; optional):
                the gradient magnitude to declar convergence

        Returns:
            Momentum

        """
        super().__init__(stepsize, tolerance)
        self.memory = GradientMemory(mean_weight=momentum,
                                     mean_square_weight=0)

    def reset(self):
        """
        Reset the gradient memory.

        Notes:
            Modifies gradient memory in place.

        Args:
            None

        Returns:
            None

        """
        self.memory.reset()

    def update(self, model, grad):
        """
        Update the model parameters with a gradient step.

        Notes:
            Changes parameters of model in place.

        Args:
            model: a BoltzmannMachine object to optimize
            grad: a Gradient object
            epoch (int): the current epoch

        Returns:
            None

        """
        self.memory.update(grad)
        self.delta = deepcopy(self.memory.mean_gradient)
        gu.grad_apply_(self.lr_, self.delta)
        model.parameter_update(self.delta)


class RMSProp(Optimizer):
    """
    Stochastic gradient descent with RMSProp.
    Geoffrey Hinton's Coursera Course Lecture 6e

    """
    def __init__(self,
                 stepsize=schedules.Constant(initial=0.001),
                 mean_square_weight=0.9,
                 tolerance=1e-7):
        """
        Create a stochastic gradient descent with RMSProp optimizer.

        Aliases:
            rmsprop

        Args:
            model: a BoltzmannMachine object to optimize
            stepsize (generator; optional): the stepsize schedule
            mean_square_weight (float; optional):
                for computing the running average of the mean-square gradient
            tolerance (float; optional):
                the gradient magnitude to declar convergence

        Returns:
            RMSProp

        """
        super().__init__(stepsize, tolerance)
        self.memory = GradientMemory(mean_weight=0,
                                     mean_square_weight=mean_square_weight)

    def reset(self):
        """
        Reset the gradient memory.

        Notes:
            Modifies gradient memory in place.

        Args:
            None

        Returns:
            None

        """
        self.memory.reset()

    def update(self, model, grad):
        """
        Update the model parameters with a gradient step.

        Notes:
            Changes parameters of model in place.

        Args:
            model: a BoltzmannMachine object to optimize
            grad: a Gradient object
            epoch (int): the current epoch

        Returns:
            None

        """
        self.memory.update(grad)
        self.delta = self.memory.normalize(grad, True)
        gu.grad_apply_(self.lr_, self.delta)
        model.parameter_update(self.delta)


class ADAM(Optimizer):
    """
    Stochastic gradient descent with Adaptive Moment Estimation algorithm.

    Kingma, D. P., & Ba, J. L. (2015).
    Adam: a Method for Stochastic Optimization.
    International Conference on Learning Representations, 1–13.

    """
    def __init__(self,
                 stepsize=schedules.Constant(initial=0.001),
                 mean_weight=0.9,
                 mean_square_weight=0.999,
                 tolerance=1e-7):
        """
        Create a stochastic gradient descent with ADAM optimizer.

        Aliases:
            adam

        Args:
            model: a BoltzmannMachine object to optimize
            stepsize (generator; optional): the stepsize schedule
            mean_weight (float; optional):
                for computing the running average of the mean gradient
            mean_square_weight (float; optional):
                for computing the running average of the mean-square gradient
            tolerance (float; optional):
                the gradient magnitude to declar convergence

        Returns:
            ADAM

        """
        super().__init__(stepsize, tolerance)
        self.memory = GradientMemory(mean_weight=mean_weight,
                                     mean_square_weight=mean_square_weight)

    def reset(self):
        """
        Reset the gradient memory.

        Notes:
            Modifies gradient memory in place.

        Args:
            None

        Returns:
            None

        """
        self.memory.reset()

    def update(self, model, grad):
        """
        Update the model parameters with a gradient step.

        Notes:
            Changes parameters of model in place.

        Args:
            model: a BoltzmannMachine object to optimize
            grad: a Gradient object
            epoch (int): the current epoch

        Returns:
            None

        """
        self.memory.update(grad)
        self.delta = self.memory.normalize(self.memory.mean_gradient, True)
        gu.grad_apply_(self.lr_, self.delta)
        model.parameter_update(self.delta)


# ----- ALIASES ----- #

gradient = Gradient
momentum = Momentum
rmsprop = RMSProp
adam = ADAM
