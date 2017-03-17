from . import backends as be
from math import sqrt
from copy import deepcopy

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

    def update_mean(self, grad):
        """
        Update the running average of the model gradients.

        """

        if self.mean_gradient is None:
            self.mean_gradient = deepcopy(grad)
        else:
            for key in grad:
                # grad[key] is a list
                for i in range(len(grad[key])):
                    # grad[key][i] is a dict
                    for p in grad[key][i]:
                        # grad[key][i][p] is a tensor
                        be.mix_inplace(self.mean_weight,
                        self.mean_gradient[key][i][p],
                        grad[key][i][p]
                        )

    def update_mean_square(self, grad):
        """
        Update the running average of the squared model gradients.

        """

        if self.mean_square_gradient is None:
            for key in grad:
                # grad[key] is a list
                for i in range(len(grad[key])):
                    # grad[key][i] is a dict
                    for p in grad[key][i]:
                        # grad[key][i][p] is a tensor
                        self.mean_square_gradient[key][i][p] = be.square(
                                                                grad[key][i][p]
                                                                )
        else:
            for key in grad:
                # grad[key] is a list
                for i in range(len(grad[key])):
                    # grad[key][i] is a dict
                    for p in grad[key][i]:
                        # grad[key][i][p] is a tensor
                        be.square_mix_inplace(self.mean_weight,
                        self.mean_gradient[key][i][p],
                        grad[key][i][p]
                        )

    def update(self, grad):
        """
        Update the running average of the model gradients and the running
        average of the squared model gradients.

        """
        if self.mean_weight:
            self.update_mean(grad)
        if self.mean_square_weight:
            self.update_mean_square(grad)

    def normalize(self, grad):
        """
        Divide grad by the square root of the mean square gradient.

        """
        for key in grad:
            # grad[key] is a list
            for i in range(len(grad[key])):
                # grad[key][i] is a dict
                for p in grad[key][i]:
                    # grad[key][i][p] is a tensor
                    be.sqrt_div(grad[key][i][p],
                                self.mean_square_gradient[key][i][p])


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
                 tolerance=1e-3):
        self.scheduler = scheduler
        self.tolerance = tolerance
        self.delta = {}

    def check_convergence(self):
        mag = gradient_magnitude(self.delta)
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
                 tolerance=1e-3):
        super().__init__(scheduler, tolerance)
        self.stepsize = stepsize

    def update(self, model, v_data, v_model, epoch):
        self.scheduler.increment(epoch)
        lr = self.scheduler.get_lr() * self.stepsize

        self.delta = model.gradient(v_data, v_model)

        for l in self.delta['layers']:
            be.multiply_dict_inplace(l, lr)

        for l in self.delta['weights']:
            be.multiply_dict_inplace(l, lr)

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
                 tolerance=1e-6):
        super().__init__(scheduler, tolerance)
        self.stepsize = stepsize
        self.memory = GradientMemory(mean_weight=momentum,
                                     mean_square_weight=0)

    def update(self, model, v_data, v_model, epoch):
        self.scheduler.increment(epoch)
        lr = self.scheduler.get_lr() * self.stepsize

        grad = model.gradient(v_data, v_model)
        self.memory.update(grad)
        self.delta = deepcopy(self.memory.mean_gradient)

        for l in self.delta['layers']:
            be.multiply_dict_inplace(l, lr)

        for l in self.delta['weights']:
            be.multiply_dict_inplace(l, lr)

        model.parameter_update(self.delta)


class RMSProp(Optimizer):
    """RMSProp
       Geoffrey Hinton's Coursera Course Lecture 6e

    """
    def __init__(self, model,
                 stepsize=0.001,
                 mean_square_weight=0.9,
                 scheduler=PowerLawDecay(),
                 tolerance=1e-6):
        super().__init__(scheduler, tolerance)
        self.stepsize = be.float_scalar(stepsize)

        self.memory = GradientMemory(mean_weight=0,
                                     mean_square_weight=mean_square_weight)

    def update(self, model, v_data, v_model, epoch):
        self.scheduler.increment(epoch)
        lr = self.scheduler.get_lr() * self.stepsize

        grad = model.gradient(v_data, v_model)
        self.memory.update(grad)

        # this deepcopy is probably not needed
        self.delta = deepcopy(grad)

        for l in self.delta['layers']:

            be.multiply_dict_inplace(l, lr)

        for l in self.delta['weights']:
            be.multiply_dict_inplace(l, lr)

        model.parameter_update(self.delta)

        for key in self.grad:
            be.square_mix_inplace(self.mean_square_weight, self.mean_square_grad[key], self.grad[key])
            model.params[key] -= lr * be.sqrt_div(self.grad[key], self.mean_square_grad[key])


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
                 tolerance=1e-6):
        super().__init__(scheduler, tolerance)
        self.stepsize = be.float_scalar(stepsize)
        self.mean_weight = be.float_scalar(mean_weight)
        self.mean_square_weight = be.float_scalar(mean_square_weight)
        self.grad = {key: be.zeros_like(model.params[key])
                        for key in model.params}
        self.mean_square_grad = {key: be.zeros_like(model.params[key])
                                    for key in model.params}
        self.mean_grad = {key: be.zeros_like(model.params[key])
                            for key in model.params}
        self.epsilon = be.float_scalar(1e-6)

    def update(self, model, v_data, v_model, epoch):
        self.scheduler.increment(epoch)
        lr = self.scheduler.get_lr() * self.stepsize

        self.grad = gradient(model, v_data, v_model)
        if model.penalty:
            for key in model.penalty:
                self.grad[key] += model.penalty[key].grad(model.params[key])
        for key in self.grad:
            be.square_mix_inplace(self.mean_square_weight, self.mean_square_grad[key], self.grad[key])
            be.mix_inplace(self.mean_weight, self.mean_grad[key], self.grad[key])
            model.params[key] -= (lr / (1 - self.mean_weight)) * be.sqrt_div(
            self.mean_grad[key], self.epsilon + self.mean_square_grad[key]/(1 - self.mean_square_weight)
            )


# ----- ALIASES ----- #

sgd = SGD = StochasticGradientDescent
momentum = Momentum
rmsprop = RMSProp
adam = ADAM


# ----- FUNCTIONS ----- #

def gradient_magnitude(grad) -> float:

    # for an rbm
    # grad looks someting like like
    # {'layers:
    # [{'loc': visible_derivative: Tensor},
    # {'loc': hidden_divative: Tensor}],
    # 'weights':
    # [{'matrix': weights_derivative: Tensor}]}

    mag = 0
    norm = 0
    for key in grad:
        for layer in grad[key]:
            for param in layer:
                norm += 1
                mag += be.norm(layer[param]) ** 2 / len(layer[param])
    return sqrt(mag / norm)
