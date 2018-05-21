import sys

from . import backends as be


class Penalty(object):
    """
    Base penalty class.
    Derived classes should define `value` and `grad` functions.

    """
    def __init__(self, penalty, slice_tuple = (slice(None,None,None),)):
        """
        Create a base Penalty object.

        Args:
            penalty (float): strength of the penalty
            slice_tuple (slice): list of slices that define the parts of the
                tensor to which the penalty will be applied

        Returns:
            Penalty

        """
        self.penalty = penalty
        self.slice_tuple = slice_tuple

    def value(self, tensor):
        """
        The value of the penalty function on a tensor.

        Args:
            tensor

        Returns:
            float

        """
        raise NotImplementedError

    def grad(self, tensor):
        """
        The value of the gradient of the penalty function on a tensor.

        Args:
            tensor

        Returns:
            tensor

        """
        raise NotImplementedError

    def get_config(self):
        """
        Returns a config for the penalty.

        Args:
            None

        Returns:
            dict

        """
        return {"name": self.__class__.__name__,
                "penalty": self.penalty,
                "slice": self.slice_tuple
               }


class trivial_penalty(Penalty):
    """A penalty that does nothing."""
    def __init__(self, penalty_unused=0, slice_tuple_unused=(slice(None,None,None),)):
        """
        Create a base trivial penalty.

        Args:
            penalty_unused (float): strength of the penalty
            slice_tuple_unused (slice): list of slices that define the parts of the
                tensor to which the penalty will be applied

        Returns:
            trivial_penalty

        """
        super().__init__(penalty_unused)

    def value(self, tensor):
        """
        The value of the penalty function on a tensor.

        Args:
            tensor

        Returns:
            float

        """
        return 0.0

    def grad(self, tensor):
        """
        The value of the gradient of the penalty function on a tensor.

        Args:
            tensor

        Returns:
            tensor

        """
        return be.zeros_like(tensor)


class l2_penalty(Penalty):
    """
    An L2 penalty encourages small values of the parameters.
    Also known as a "ridge" penalty.

    """
    def __init__(self, penalty, slice_tuple=(slice(None,None,None),)):
        """
        Create an l2 penalty.

        Args:
            penalty (float): strength of the penalty
            slice_tuple (slice): list of slices that define the parts of the
                tensor to which the penalty will be applied

        Returns:
            l2_penalty

        """
        super().__init__(penalty, slice_tuple)

    def value(self, tensor):
        """
        The value of the penalty function on a tensor.

        Args:
            tensor

        Returns:
            float

        """
        return 0.5 * self.penalty * be.tsum(tensor[self.slice_tuple]**2)

    def grad(self, tensor):
        """
        The value of the gradient of the penalty function on a tensor.

        Args:
            tensor

        Returns:
            tensor

        """
        retval = be.zeros_like(tensor)
        retval[self.slice_tuple] = self.penalty * tensor[self.slice_tuple]
        return retval


class l1_penalty(Penalty):
    """
    An l1 penalty encourages small values of the parameters,
    and tends to produce solutions that are more sparse than an l2 penalty.
    Also known as "lasso".

    """
    def __init__(self, penalty, slice_tuple=(slice(None,None,None),)):
        """
        Create an l1 penalty.

        Args:
            penalty (float): strength of the penalty
            slice_tuple (slice): list of slices that define the parts of the
                tensor to which the penalty will be applied

        Returns:
            l1_penalty

        """
        super().__init__(penalty, slice_tuple)

    def value(self, tensor):
        """
        The value of the penalty function on a tensor.

        Args:
            tensor

        Returns:
            float

        """
        return self.penalty * be.tsum(be.tabs(tensor[self.slice_tuple]))

    def grad(self, tensor):
        """
        The value of the gradient of the penalty function on a tensor.

        Args:
            tensor

        Returns:
            tensor

        """
        retval = be.zeros_like(tensor)
        retval[self.slice_tuple] = self.penalty * be.sign(tensor[self.slice_tuple])
        return retval


class exp_l2_penalty(Penalty):
    """
    Puts an l2 penalty on the exponentiated parameters.
    Useful when the parameters are represented in a logarithmic space.
    For example, encouraging the variances of a GaussianLayer to be small.

    """
    def __init__(self, penalty, slice_tuple=(slice(None,None,None),)):
        """
        Create an exp l2 penalty.

        Args:
            penalty (float): strength of the penalty
            slice_tuple (slice): list of slices that define the parts of the
                tensor to which the penalty will be applied

        Returns:
            exp_l2_penalty

        """
        super().__init__(penalty, slice_tuple)

    def value(self, tensor):
        """
        The value of the penalty function on a tensor.

        Args:
            tensor

        Returns:
            float

        """
        return 0.5 * self.penalty * be.tsum(be.exp(2.0*tensor[self.slice_tuple]))

    def grad(self, tensor):
        """
        The value of the gradient of the penalty function on a tensor.

        Args:
            tensor

        Returns:
            tensor

        """
        retval = be.zeros_like(tensor)
        retval[self.slice_tuple] = self.penalty * be.exp(2.0*tensor[self.slice_tuple])
        return retval


class l1_adaptive_decay_penalty_2(Penalty):
    """
    Modified form of the l1 penalty which regularizes more for weight rows
    with larger coupling to target layer by way of a quadratic power

    Tubiana, J., Monasson, R. (2017)
    Emergence of Compositional Representations in Restricted Boltzmann Machines,
    PRL 118, 138301 (2017), Supplemental Material I.D

    Note: expects to operate on a tensor with two degrees of freedom (eg. a weight matrix)

    """
    def __init__(self, penalty, slice_tuple=(slice(None,None,None),)):
        """
        Create an adaptive l1 penalty.

        Args:
            penalty (float): strength of the penalty
            slice_tuple (slice): list of slices that define the parts of the
                tensor to which the penalty will be applied

        Returns:
            l1_adaptive_decay_penalty_2

        """
        super().__init__(penalty, slice_tuple)

    def value(self, tensor):
        """
        The value of the penalty function on a tensor.

        Args:
            tensor

        Returns:
            float

        """
        return 0.5 * self.penalty * \
               be.tsum(be.square(be.tsum(be.tabs(tensor[self.slice_tuple]),axis=0)))

    def grad(self, tensor):
        """
        The value of the gradient of the penalty function on a tensor.

        Args:
            tensor

        Returns:
            tensor

        """
        retval = be.zeros_like(tensor)
        retval[self.slice_tuple] = self.penalty * \
               be.multiply(be.tsum(be.tabs(tensor[self.slice_tuple]),axis=0),
                           be.sign(tensor[self.slice_tuple]))
        return retval


class log_penalty(Penalty):
    """A logarithmic penalty forces the parameters to be positive."""
    def __init__(self, penalty, slice_tuple=(slice(None,None,None),)):
        """
        Create a log penalty.

        Args:
            penalty (float): strength of the penalty
            slice_tuple (slice): list of slices that define the parts of the
                tensor to which the penalty will be applied

        Returns:
            log_penalty

        """
        super().__init__(penalty, slice_tuple)

    def value(self, tensor):
        """
        The value of the penalty function on a tensor.

        Args:
            tensor

        Returns:
            float

        """
        return -self.penalty * be.tsum(be.log(tensor[self.slice_tuple]))

    def grad(self, tensor):
        """
        The value of the gradient of the penalty function on a tensor.

        Args:
            tensor

        Returns:
            tensor

        """
        retval = be.zeros_like(tensor)
        retval[self.slice_tuple] = -self.penalty / tensor[self.slice_tuple]
        return retval


class logdet_penalty(Penalty):
    """
    Penalty acts on the logarithm of the determinant of a matrix.
    Discourages the matrix from becoming singular.

    """
    def __init__(self, penalty, slice_tuple=(slice(None,None,None),)):
        """
        Create a logdet_penalty.

        Args:
            penalty (float): strength of the penalty
            slice_tuple (slice): list of slices that define the parts of the
                tensor to which the penalty will be applied

        Returns:
            logdet_penalty

        """
        super().__init__(penalty, slice_tuple)
        assert len(slice_tuple) < 2, "logdet doesn't work with multiindex slices"

    def value(self, tensor):
        """
        The value of the penalty function on a tensor.

        Args:
            tensor

        Returns:
            float

        """
        J = be.dot(be.transpose(tensor[self.slice_tuple]), tensor[self.slice_tuple])
        return -self.penalty * be.logdet(J)

    def grad(self, tensor):
        """
        The value of the gradient of the penalty function on a tensor.

        Args:
            tensor

        Returns:
            tensor

        """
        retval = be.ones_like(tensor)
        retval[self.slice_tuple] = -2 * self.penalty * be.transpose(be.pinv(tensor[self.slice_tuple]))
        return retval


class log_norm(Penalty):
    """
    An RBM with n visible units and m hidden units has an (n, m) weight matrix.
    The log norm penalty discourages any of the m columns of the weight matrix
    from having zero norm.

    """
    def __init__(self, penalty, slice_tuple=(slice(None,None,None),)):
        """
        Create an log_norm.

        Args:
            penalty (float): strength of the penalty
            slice_tuple (slice): list of slices that define the parts of the
                tensor to which the penalty will be applied

        Returns:
            log_norm

        """
        super().__init__(penalty, slice_tuple)

    def value(self, tensor):
        """
        The value of the penalty function on a tensor.

        Args:
            tensor

        Returns:
            float

        """
        return -self.penalty * 0.5*be.tsum(be.log(be.square(be.norm(tensor[self.slice_tuple], axis=0))))

    def grad(self, tensor):
        """
        The value of the gradient of the penalty function on a tensor.

        Args:
            tensor

        Returns:
            tensor

        """
        tmp = be.square(be.norm(tensor[self.slice_tuple], axis=0))
        retval = be.zeros_like(tensor)
        retval[self.slice_tuple] = -self.penalty * be.divide(tmp, tensor[self.slice_tuple])
        return retval


class l2_norm(Penalty):
    """
    An RBM with n visible units and m hidden units has an (n, m) weight matrix.
    The l2 norm penalty encourages the columns of the weight matrix to have
    norms that are close to the target value.

    "On Training Deep Boltzmann Machines"
    by Guillaume Desjardins, Aaron Courville, Yoshua Bengio
    http://arxiv.org/pdf/1203.4416.pdf

    """
    def __init__(self, penalty, slice_tuple=(slice(None,None,None),), target=1):
        """
        Create an l2_norm penalty.

        Args:
            penalty (float): strength of the penalty
            slice_tuple (slice): list of slices that define the parts of the
                tensor to which the penalty will be applied
            target (optional; float): the shrinkage target

        Returns:
            l2_norm

        """
        super().__init__(penalty, slice_tuple)
        self.target = target

    def value(self, tensor):
        """
        The value of the penalty function on a tensor.

        Args:
            tensor

        Returns:
            float

        """
        norms = be.norm(tensor[self.slice_tuple], axis=0)
        return 0.5 * self.penalty * be.tsum(be.square(norms - self.target))

    def grad(self, tensor):
        """
        The value of the gradient of the penalty function on a tensor.

        Args:
            tensor

        Returns:
            tensor

        """
        norms = be.EPSILON + be.norm(tensor[self.slice_tuple], axis=0)
        retval = be.zeros_like(tensor)
        retval[self.slice_tuple] = \
            self.penalty * (self.target- 1 / norms) * tensor[self.slice_tuple]
        return retval


# ----- FUNCTIONS ----- #

def from_config(config):
    """
    Builds an instance from a config.

    Args:
        config (List or dict)

    Returns:
        Penalty

    """
    empty_slice = (slice(None,None,None),)
    configs = be.force_list(config)
    pens = [getattr(sys.modules[__name__], c["name"]) for c in configs]
    penalties = \
        [pens[i](configs[i]["penalty"], be.maybe_key(configs[i],"slice", empty_slice))
         for i in range(len(pens))]
    if len(penalties) == 1:
        return penalties[0]
    return PenaltySum(penalties)


# ----- ALIASES ----- #

ridge = l2_penalty
lasso = l1_penalty
