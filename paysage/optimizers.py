import numpy
from . import backends as B


# ----- OPTIMIZERS ----- #

class Optimizer(object):

    def __init__(self, lr_decay=0.9, tolerance=1e-3):
        self.lr_decay = lr_decay
        self.lr = 1
        self.tolerance = tolerance
        self.grad = {}

    def update_lr(self, epoch):
        self.lr = (self.lr_decay ** epoch)
        return self.lr

    def check_convergence(self):
        mag = gradient_magnitude(self.grad) * self.lr
        print(mag)
        if mag <= self.tolerance:
            return True
        else:
            return False


class StochasticGradientDescent(Optimizer):
    """StochasticGradientDescent
       Basic algorithm of gradient descent with minibatches.

    """
    def __init__(self, model, stepsize=0.001, lr_decay=0.5, tolerance=1e-3):
        super().__init__(lr_decay, tolerance)
        self.stepsize = stepsize
        self.grad = {key: numpy.zeros_like(model.params[key]) for key in model.params}

    def update(self, model, v_data, v_model, epoch):
        lr = self.update_lr(epoch) * self.stepsize
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
       Qian, N. (1999). On the momentum term in gradient descent learning algorithms.
          Neural Networks : The Official Journal of the International Neural Network Society, 12(1), 145–151

    """
    def __init__(self, model, stepsize=0.001, momentum=0.9, lr_decay=0.5, tolerance=1e-6):
        super().__init__(lr_decay, tolerance)
        self.stepsize = stepsize
        self.momentum = momentum
        self.grad = {key: numpy.zeros_like(model.params[key]) for key in model.params}
        self.delta = {key: numpy.zeros_like(model.params[key]) for key in model.params}

    def update(self, model, v_data, v_model, epoch):
        lr = self.update_lr(epoch) * self.stepsize
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
    def __init__(self, model, stepsize=0.001, mean_square_weight=0.9, lr_decay=0.9, tolerance=1e-6):
        super().__init__(lr_decay, tolerance)
        self.stepsize = numpy.float32(stepsize)
        self.mean_square_weight = numpy.float32(mean_square_weight)
        self.grad = {key: numpy.zeros_like(model.params[key]) for key in model.params}
        self.mean_square_grad = {key: numpy.zeros_like(model.params[key]) for key in model.params}
        self.epsilon = numpy.float32(1e-6)

    def update(self, model, v_data, v_model, epoch):
        lr = self.update_lr(epoch) * self.stepsize
        self.grad = gradient(model, v_data, v_model)
        if model.penalty:
            for key in model.penalty:
                self.grad[key] += model.penalty[key].grad(model.params[key])
        for key in self.grad:
            B.square_mix_inplace(self.mean_square_weight, self.mean_square_grad[key], self.grad[key])
            model.params[key] -= lr * B.sqrt_div(self.grad[key], self.epsilon + self.mean_square_grad[key])
        model.enforce_constraints()


class ADAM(Optimizer):
    """ADAM
       Adaptive Moment Estimation algorithm.
       Kingma, D. P., & Ba, J. L. (2015). Adam: a Method for Stochastic Optimization. International Conference on Learning Representations, 1–13.

    """
    def __init__(self, model, stepsize=0.001, mean_weight=0.9, mean_square_weight=0.999, lr_decay=0.9, tolerance=1e-6):
        super().__init__(lr_decay, tolerance)
        self.stepsize = numpy.float32(stepsize)
        self.mean_weight = numpy.float32(mean_weight)
        self.mean_square_weight = numpy.float32(mean_square_weight)
        self.grad = {key: numpy.zeros_like(model.params[key]) for key in model.params}
        self.mean_square_grad = {key: numpy.zeros_like(model.params[key]) for key in model.params}
        self.mean_grad = {key: numpy.zeros_like(model.params[key]) for key in model.params}
        self.epsilon = numpy.float32(1e-6)

    def update(self, model, v_data, v_model, epoch):
        lr = self.update_lr(epoch) * self.stepsize
        self.grad = gradient(model, v_data, v_model)
        if model.penalty:
            for key in model.penalty:
                self.grad[key] += model.penalty[key].grad(model.params[key])
        for key in self.grad:
            B.square_mix_inplace(self.mean_square_weight, self.mean_square_grad[key], self.grad[key])
            B.mix_inplace(self.mean_weight, self.mean_grad[key], self.grad[key])
            model.params[key] -= (lr / (1 - self.mean_weight)) * B.sqrt_div(self.mean_grad[key], self.epsilon + self.mean_square_grad[key]/(1 - self.mean_square_weight))
        model.enforce_constraints()


# ----- ALIASES ----- #

sgd = SGD = StochasticGradientDescent
momentum = Momentum
rmsprop = RMSProp
adam = ADAM


# ----- FUNCTIONS ----- #

# gradient: (LatentModel, numpy.ndarray, numpy.ndarray) -> numpy.ndarray
def gradient(model, minibatch, samples):
    positive_phase = model.derivatives(minibatch.astype(numpy.float32))
    negative_phase = model.derivatives(samples.astype(numpy.float32))
    return {key: (positive_phase[key] - negative_phase[key]) for key in positive_phase}

# gradient_magnitude: (dict) -> float:
def gradient_magnitude(grad):
    mag = 0
    for key in grad:
        # numba doesn't seem to speed this up
        mag += numpy.linalg.norm(grad[key])**2 / len(grad[key])
    return numpy.sqrt(mag / len(grad))
