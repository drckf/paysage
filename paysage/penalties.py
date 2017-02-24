from . import backends as be

# ----- FUNCTIONS ----- #

class l2_penalty(object):

    def __init__(self, penalty):
        self.penalty = penalty

    def value(self, tensor):
        return 0.5 * self.penalty * be.tsum(tensor**2)

    def grad(self, tensor):
        return self.penalty * tensor


class l1_penalty(object):

    def __init__(self, penalty):
        self.penalty = penalty

    def value(self, tensor):
        return self.penalty * be.tsum(be.tabs(tensor))

    def grad(self, tensor):
        return self.penalty * be.sign(tensor)


class log_penalty(object):

    def __init__(self, penalty):
        self.penalty = penalty

    def value(self, tensor):
        return -self.penalty * be.log(tensor)

    def grad(self, tensor):
        return -self.penalty / tensor


# ----- ALIASES ----- #

ridge = l2_penalty
lasso = l1_penalty
