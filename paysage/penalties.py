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


# ----- ALIASES ----- #

ridge = l2_penalty
lasso = l1_penalty
