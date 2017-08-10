import os
from collections import OrderedDict, namedtuple
import pandas

from .. import penalties
from .. import constraints
from .. import backends as be

# CumulantsTAP type is common to all layers
CumulantsTAP = namedtuple("CumulantsTAP", ["mean", "variance"])

# Params type must be redefined for all Layers
ParamsLayer = namedtuple("Params", [])

class Layer(object):
    """
    A general layer class with common functionality.

    """
    def __init__(self, num_units, dropout_p, *args, **kwargs):
        """
        Basic layer initialization method.

        Args:
            num_units (int): number of units in the layer
            dropout_p (float): likelihood each unit is dropped out in training
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
        self.dropout_p = dropout_p
        self.len = num_units
        self.fixed_params = []
        self.moments = None

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

    def use_dropout(self):
        """
        Indicate if the layer has dropout.

        Args:
            None

        Returns:
            true of false

        """
        return self.dropout_p > 0

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
            "dropout"     : self.dropout_p,
            "parameters"  : list(self.params._fields),
            "penalties"   : {pk: self.penalties[pk].get_config()
                             for pk in self.penalties},
            "constraints" : {ck: self.constraints[ck].__name__
                             for ck in self.constraints}
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
        layer = cls(config["num_units"], config["dropout"])
        for k, v in config["penalties"].items():
            layer.add_penalty({k: penalties.from_config(v)})
        for k, v in config["constraints"].items():
            layer.add_constraint({k: getattr(constraints, v)})
        return layer

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
        for i, ip in enumerate(self.params):
            params.append(be.float_tensor(
                store.get(os.path.join(key, 'parameters', 'key'+str(i))).as_matrix()
            ).squeeze()) # collapse trivial dimensions to a vector
        self.set_params(self.params.__class__(*params))

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
        self.set_params(be.mapzip(be.subtract, deltas[0], self.params))
        self.enforce_constraints()

    def get_dropout_mask(self, batch_size=1):
        """
        Return a binary mask

        Args:
            batch_size (int): number of masks to generate

        Returns:
            mask (tensor (batch_size, self.len): binary mask
        """
        return be.float_tensor(be.rand((batch_size, self.len)) > self.dropout_p)

    def get_dropout_scale(self, batch_size=1):
        """
        Return a vector representing the probability that each unit is on
            with respect to dropout

        Args:
            batch_size (int): number of copies to return

        Returns:
            scale (tensor (batch_size, self.len)): vector of scales for each unit
        """
        return (1.0-self.dropout_p) * be.ones((batch_size, self.len))