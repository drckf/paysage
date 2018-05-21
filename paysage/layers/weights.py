import os, sys
from collections import OrderedDict, namedtuple
import pandas

from .. import penalties
from .. import constraints
from .. import backends as be


def weights_from_config(config):
    """
    Construct a layer from a configuration.

    Args:
        A dictionary configuration of the layer metadata.

    Returns:
        An object which is a subclass of 'Weights'.

    """
    layer_obj = getattr(sys.modules[__name__], config["layer_type"])
    return layer_obj.from_config(config)


ParamsWeights = namedtuple("ParamsWeights", ["matrix"])

class Weights(object):
    """
    Layer class for weights.

    """
    def __init__(self, shape):
        """
        Create a weight layer.

        Notes:
            The shape is regarded as a dimensionality of
            the target and domain units for the layer,
            as `shape = (target, domain)`.

        Args:
            shape (tuple): shape of the weight tensor (int, int)

        Returns:
            weights layer

        """
        # these attributes are immutable (their keys don't change)
        self.shape = shape
        self.params = ParamsWeights(be.zeros(shape))

        # these attributes are mutable (their keys do change)
        self.penalties = OrderedDict()
        self.constraints = OrderedDict()
        self.fixed_params = []

    def set_fixed_params(self, fixed_params):
        """
        Set the params that are not trainable.

        Notes:
            Modifies the layer.fixed_params attribute in place.

        Args:
            fixed_params (List[str]): a list of the fixed parameter names

        Returns:
            None

        """
        self.fixed_params = fixed_params

    def get_fixed_params(self):
        """
        Get the params that are not trainable.

        Args:
            None

        Returns:
            fixed_params (List[str]): a list of the fixed parameter names

        """
        return self.fixed_params

    def _get_trainable_indices(self):
        """
        Get the indices of the trainable params.

        Args:
            None

        Returns:
            trainable param indices (List[int])

        """
        fields = list(self.params._fields)
        trainable_fields = [f for f in fields if f not in self.fixed_params]
        return [fields.index(f) for f in trainable_fields]

    def set_params(self, new_params):
        """
        Set the value of the layer params attribute.

        Notes:
            Modifies layer.params in place.

        Args:
            new_params (namedtuple)

        Returns:
            None

        """
        for i in self._get_trainable_indices():
            self.params[i][:] = new_params[i]

    def get_param_names(self):
        """
        Return the field names of the params attribute.

        Args:
            None

        Returns:
            field names (List[str])

        """
        return list(self.params._fields)

    def get_config(self):
        """
        Get the configuration dictionary of the weights layer.

        Args:
            None:

        Returns:
            configuration (dict):

        """
        base_config = {
            "layer_type"  : self.__class__.__name__,
            "parameters"  : list(self.params._fields),
            "penalties"   : {pk: self.penalties[pk].get_config()
                             for pk in self.penalties},
            "constraints" : {ck: self.constraints[ck].__name__
                             for ck in self.constraints},
            "shape": self.shape
        }
        return base_config

    @classmethod
    def from_config(cls, config):
        """
        Create a weights layer from a configuration dictionary.

        Args:
            config (dict)

        Returns:
            layer (Weights)

        """
        weights = cls(config["shape"])
        for k, v in config["penalties"].items():
            weights.add_penalty({k: penalties.from_config(v)})
        for k, v in config["constraints"].items():
            weights.add_constraint({k: getattr(constraints, v)})
        return weights

    def save_params(self, store, key):
        """
        Save the parameters to a HDFStore.

        Notes:
            Performs an IO operation.

        Args:
            store (pandas.HDFStore): the writeable stream for the params.
            key (str): the path for the layer params.

        Returns:
            None

        """
        for i, ip in enumerate(self.params):
            df_params = pandas.DataFrame(be.to_numpy_array(ip))
            store.put(os.path.join(key, 'parameters', 'key'+str(i)), df_params)

    def load_params(self, store, key):
        """
        Load the parameters from an HDFStore.

        Notes:
            Performs an IO operation.

        Args:
            store (pandas.HDFStore): the readable stream for the params.
            key (str): the path for the layer params.

        Returns:
            None

        """
        params = []
        for i, _ in enumerate(self.params):
            params.append(be.float_tensor(
                store.get(os.path.join(key, 'parameters', 'key'+str(i))).as_matrix()
            )) # collapse trivial dimensions to a vector
        self.params = self.params.__class__(*params)

    def add_constraint(self, constraint):
        """
        Add a parameter constraint to the layer.

        Notes:
            Modifies the layer.contraints attribute in place.

        Args:
            constraint (dict): {param_name: constraint (paysage.constraints)}

        Returns:
            None

        """
        self.constraints.update(constraint)

    def enforce_constraints(self):
        """
        Apply the contraints to the layer parameters.

        Note:
            Modifies the parameters of the layer in place.

        Args:
            None

        Returns:
            None

        """
        for param_name in self.constraints:
            self.constraints[param_name](getattr(self.params, param_name))

    def add_penalty(self, penalty):
        """
        Add a penalty to the layer.

        Note:
            Modfies the layer.penalties attribute in place.

        Args:
            penalty (dict): {param_name: penalty (paysage.penalties)}

        Returns:
            None

        """
        self.penalties.update(penalty)

    def get_penalties(self):
        """
        Get the value of the penalties:

        E.g., L2 penalty = (1/2) * penalty * \sum_i parameter_i ** 2

        Args:
            None

        Returns:
            dict (float): the values of the penalty functions

        """
        pen = {param_name:
               self.penalties[param_name].value(
                   getattr(self.params, param_name)
               )
               for param_name in self.penalties}
        return pen

    def get_penalty_grad(self, deriv, param_name):
        """
        Get the gradient of the penalties on a parameter.

        E.g., L2 penalty gradient = penalty * parameter_i

        Args:
            deriv (tensor): derivative of the parameter
            param_name: name of the parameter

        Returns:
            tensor: derivative including penalty

        """
        if param_name not in self.penalties:
            return deriv
        return deriv + self.penalties[param_name].grad(
                getattr(self.params, param_name))

    def parameter_step(self, deltas):
        """
        Update the values of the parameters:

        layer.params.name -= deltas.name

        Notes:
            Modifies the elements of the layer.params attribute in place.

        Args:
            deltas (List[namedtuple]): List[param_name: tensor] (update)

        Returns:
            None

        """
        self.set_params(be.mapzip(be.subtract, deltas[0], self.params))
        self.enforce_constraints()

    def W(self, trans=False):
        """
        Get the weight matrix.

        A convenience method for accessing layer.params.matrix
        with a shorter syntax.

        Args:
            trans (optional; bool): transpose the matrix if true

        Returns:
            tensor: weight matrix

        """
        if not trans:
            return self.params.matrix
        return be.transpose(self.params.matrix)

    def derivatives(self, units_target, units_domain,
                    penalize=True, weighting_function=be.do_nothing):
        """
        Compute the derivative of the weights layer.

        dW_{ij} = - \frac{1}{num_samples} * \sum_{k} v_{ki} h_{kj}

        Args:
            units_target (tensor (num_samples, num_visible)): Rescaled target units.
            units_domain (tensor (num_samples, num_visible)): Rescaled domain units.
            penalize (bool): whether to add a penalty term.
            weighting_function (function): a weighting function to apply
                to units when computing the gradient.

        Returns:
            derivs (List[namedtuple]): List['matrix': tensor] (contains gradient)

        """
        tmp = -be.batch_outer(weighting_function(units_target),
                              units_domain) / len(units_target)
        if penalize:
            tmp = self.get_penalty_grad(tmp, "matrix")
        return [ParamsWeights(tmp)]

    def zero_derivatives(self):
        """
        Return an object like the derivatives that is filled with zeros.

        Args:
            None

        Returns:
            derivs (List[namedtuple]): List['matrix': tensor] (contains gradient)

        """
        return [be.apply(be.zeros_like, self.params)]

    def random_derivatives(self):
        """
        Return an object like the derivatives that is filled with random floats.

        Args:
            None

        Returns:
            derivs (List[namedtuple]): List['matrix': tensor] (contains gradient)

        """
        return [be.apply(be.rand_like, self.params)]

    def energy(self, target_units, domain_units):
        """
        Compute the contribution of the weight layer to the model energy.

        For sample k:
        E_k = -\sum_{ij} W_{ij} v_{ki} h_{kj}

        Args:
            target_units (tensor (num_samples, num_visible)): Rescaled target units.
            domain_units (tensor (num_samples, num_visible)): Rescaled domain units.

        Returns:
            tensor (num_samples,): energy per sample

        """
        return -be.batch_quadratic(target_units, self.W(), domain_units)

    def GFE_derivatives(self, rescaled_target_cumulants, rescaled_domain_cumulants):
        """
        Gradient of the Gibbs free energy associated with this layer

        Args:
            rescaled_target_cumulants (CumulantsTAP): rescaled magnetization of
             the shallower layer linked to w
            rescaled_domain_cumulants (CumulantsTAP): rescaled magnetization of
             the deeper layer linked to w

        Returns:
            derivs (namedtuple): 'matrix': tensor (contains gradient)

        """
        tmp = be.zeros_like(self.params.matrix)
        tmp = -be.outer(rescaled_target_cumulants.mean, rescaled_domain_cumulants.mean) - \
               be.multiply(self.params.matrix,
                be.outer(rescaled_target_cumulants.variance, rescaled_domain_cumulants.variance))

        return [ParamsWeights(tmp)]
