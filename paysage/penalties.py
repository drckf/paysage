import sys

from . import backends as be

# ----- PENALTY OBJECTS ----- #

class Penalty(object):
    """
    Base penalty class.
    Derived classes should define `value` and `grad` functions.

    """
    def __init__(self, penalty):
        self.penalty = penalty

    def value(self, tensor):
        """
        The value of the penalty function on a tensor.

        """
        raise NotImplementedError

    def grad(self, tensor):
        """
        The value of the gradient of the penalty function on a tensor.

        """
        raise NotImplementedError

    def get_config(self):
        """
        Returns a config for the penalty.

        """
        return {"name": self.__class__.__name__,
                "penalty": self.penalty
               }

class l2_penalty(Penalty):

    def __init__(self, penalty):
        super().__init__(penalty)

    def value(self, tensor):
        return 0.5 * self.penalty * be.tsum(tensor**2)

    def grad(self, tensor):
        return self.penalty * tensor


class l1_penalty(Penalty):

    def __init__(self, penalty):
        super().__init__(penalty)

    def value(self, tensor):
        return self.penalty * be.tsum(be.tabs(tensor))

    def grad(self, tensor):
        return self.penalty * be.sign(tensor)


class log_penalty(Penalty):

    def __init__(self, penalty):
        super().__init__(penalty)

    def value(self, tensor):
        return -self.penalty * be.log(tensor)

    def grad(self, tensor):
        return -self.penalty / tensor


class logdet_penalty(Penalty):

    def __init__(self, penalty):
        super().__init__(penalty)

    def value(self, tensor):
        J = be.dot(be.transpose(tensor), tensor)
        return -self.penalty * be.logdet(J)

    def grad(self, tensor):
        return -2 * self.penalty * be.transpose(be.pinv(tensor))


# ----- FUNCTIONS ----- #

def from_config(config):
    """
    Builds an instance from a config.

    """
    pen = getattr(sys.modules[__name__], config["name"])
    return pen(config["penalty"])


# ----- ALIASES ----- #

ridge = l2_penalty
lasso = l1_penalty
