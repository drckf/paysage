import os
from collections import OrderedDict, namedtuple
import pandas

from .. import penalties
from .. import constraints
from .. import backends as be
from .. import math_utils as mu

# CumulantsTAP type is common to all layers
CumulantsTAP = namedtuple("CumulantsTAP", ["mean", "variance"])
CumulantsTAP.__doc__ += \
"\nNote: the expectation thoughout the TAP codebase is that both \
mean and variance are tensors of shape (num_samples>1, num_units) \
or (num_units) in which num_samples is some sampling multiplicity \
used in the tap calculations, not the SGD batch size."

# Params type must be redefined for all Layers
ParamsLayer = namedtuple("Params", [])

class Layer(object):
    """
    A general layer class with common functionality.

    """
    def __init__(self, num_units, center=False, *args, **kwargs):
        """
        Basic layer initialization method.

        Args:
            num_units (int): number of units in the layer
            center (bool): whether to center the layer
            *args: any arguments
            **kwargs: any keyword arguments

        Returns:
            layer

        """
        # these attributes are immutable (their keys don't change)
        self.params = ParamsLayer()
        # these attributes are mutable (their keys do change)
        self.penalties = OrderedDict()
        self.constraints = OrderedDict()
        self.len = num_units
        self.fixed_params = []
        self.moments = mu.MeanVarianceArrayCalculator()
        self.center = center
        self.centering_vec = None

    def get_center(self):
        """
        Get the vector used for centering:

        Args:
            None

        Returns:
            tensor ~ (num_units,)

        """
        if self.centering_vec is not None:
            return self.centering_vec
        return self.moments.mean

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

    def get_params(self):
        """
        Get the value of the layer params attribute.

        Args:
            None

        Returns:
            params (list[namedtuple]): length=1 list

        """
        return [self.params]

    def set_params(self, new_params):
        """
        Set the value of the layer params attribute.

        Notes:
            Modifies layer.params in place.
            Note: expects a length=1 list

        Args:
            new_params (list[namedtuple])

        Returns:
            None

        """
        for i in self._get_trainable_indices():
            self.params[i][:] = new_params[0][i]

    def get_param_names(self):
        """
        Return the field names of the params attribute.

        Args:
            None

        Returns:
            field names (List[str])

        """
        return list(self.params._fields)

    def get_base_config(self):
        """
        Get a base configuration for the layer.

        Notes:
            Encodes metadata for the layer.
            Includes the base layer data.

        Args:
            None

        Returns:
            A dictionary configuration for the layer.

        """
        return {
            "layer_type"  : self.__class__.__name__,
            "num_units"   : self.len,
            "center"      : self.center,
            "parameters"  : list(self.params._fields),
            "penalties"   : {pk: self.penalties[pk].get_config()
                             for pk in self.penalties},
            "constraints" : {ck: self.constraints[ck].__name__
                             for ck in self.constraints},
            "fixed_params": self.fixed_params
        }

    def get_config(self):
        """
        Get a full configuration for the layer.

        Notes:
            Encodes metadata on the layer.
            Weights are separately retrieved.
            Builds the base configuration.

        Args:
            None

        Returns:
            A dictionary configuration for the layer.

        """
        return self.get_base_config()

    @classmethod
    def from_config(cls, config):
        """
        Create a layer from a configuration dictionary.

        Args:
            config (dict)

        Returns:
            layer

        """
        # separate out the layer arguments from other parameters
        layer_config = dict(config)
        penalties_config = layer_config.pop("penalties", {})
        constraints_config = layer_config.pop("constraints", {})
        param_names = layer_config.pop("parameters", [])
        fixed_params = layer_config.pop("fixed_params", [])
        layer = cls(**layer_config)
        for k, v in penalties_config.items():
            layer.add_penalty({k: penalties.from_config(v)})
        for k, v in constraints_config.items():
            layer.add_constraint({k: getattr(constraints, v)})
        layer.set_fixed_params(fixed_params)
        return layer

    def save_params(self, store, key):
        """
        Save the parameters to a HDFStore.  Includes the moments for the layer.

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
            store.put(os.path.join(key, 'parameters', 'params'+str(i)), df_params)
            store.put(os.path.join(key, 'parameters', 'moments'+str(i)),
                      self.moments.to_dataframe())

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
                store.get(os.path.join(key, 'parameters', 'params'+str(i))).as_matrix()
            ).squeeze()) # collapse trivial dimensions to a vector

        # load parameters, but first unset fixed_params to load, then re-set
        fixed_params_cache = self.fixed_params
        self.fixed_params = []
        self.set_params([self.params.__class__(*params)])
        self.fixed_params = fixed_params_cache

        self.moments = mu.MeanVarianceArrayCalculator.from_dataframe(
                store.get(os.path.join(key, 'parameters', 'moments'+str(i))))

    def num_parameters(self):
        return self.len * len(self.params)

    def update_moments(self, units):
        """
        Set a reference mean and variance of the layer
            (used for centering and sampling).

        Notes:
            Modifies layer.reference_mean attribute in place.

        Args:
            units (tensor (batch_size, self.len)

        Returns:
            None

        """
        self.moments.update(units, axis=0)

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
        else:
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
        self.set_params([be.mapzip(be.subtract, deltas[0], self.params)])
        self.enforce_constraints()
