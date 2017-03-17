from . import backends as be
from math import sqrt

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
        self.grad = {}

    def check_convergence(self):
        mag = gradient_magnitude(self.grad) * self.scheduler.lr
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
        self.grad = {key: be.zeros_like(model.params[key])
                        for key in model.params}

    def update(self, model, v_data, v_model, epoch):
        self.scheduler.increment(epoch)
        lr = self.scheduler.get_lr() * self.stepsize

        self.grad = gradient(model, v_data, v_model)
        if model.penalty:
            for key in model.penalty:
                self.grad[key] += model.penalty[key].grad(model.params[key])
        for key in self.grad:
            model.params[key] -= lr * self.grad[key]
        model.enforce_constraints()


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
        self.momentum = momentum
        self.grad = {key: be.zeros_like(model.params[key])
                        for key in model.params}
        self.delta = {key: be.zeros_like(model.params[key])
                        for key in model.params}

    def update(self, model, v_data, v_model, epoch):
        self.scheduler.increment(epoch)
        lr = self.scheduler.get_lr() * self.stepsize

        self.grad = gradient(model, v_data, v_model)
        if model.penalty:
            for key in model.penalty:
                self.grad[key] += model.penalty[key].grad(model.params[key])
        for key in self.grad:
            self.delta[key] = self.grad[key] + self.momentum * self.delta[key]
            model.params[key] -= lr * self.delta[key]
        model.enforce_constraints()


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
        self.mean_square_weight = be.float_scalar(mean_square_weight)
        self.grad = {key: be.zeros_like(model.params[key])
                        for key in model.params}
        self.mean_square_grad = {key: be.zeros_like(model.params[key])
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
            model.params[key] -= lr * be.sqrt_div(self.grad[key], self.epsilon + self.mean_square_grad[key])
        model.enforce_constraints()


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
        model.enforce_constraints()


# ----- ALIASES ----- #

sgd = SGD = StochasticGradientDescent
momentum = Momentum
rmsprop = RMSProp
adam = ADAM


# ----- FUNCTIONS ----- #

def gradient(model, minibatch, samples):
    positive_phase = model.derivatives(minibatch)
    grad = {}
    grad[model.weights]['val'] = model.weights.derivs['val']
    for layer in model.layers:
        for key in layer.derivs:
            grad[layer][key] = layer.derivs[key]
    negative_phase = model.derivatives(samples)
    grad[model.weights]['val'] = model.weights.derivs['val']
    for layer in model.layers:
        for key in layer.derivs:
            grad[layer][key] = layer.derivs[key]



    positive_phase = model.derivatives(minibatch)
    negative_phase = model.derivatives(samples)
    return {key: (positive_phase[key] - negative_phase[key])
                for key in positive_phase}

def gradient_magnitude(grad):
    mag = 0
    for key in grad:
        mag += be.norm(grad[key])**2 / len(grad[key])
    return sqrt(mag / len(grad))
