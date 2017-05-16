import os
import copy
import pandas
from typing import List
from collections import namedtuple
from collections import OrderedDict

from .. import layers
from .. import backends as be
from ..models.initialize import init_model as init
from . import gradient_util as gu

class State(object):
    """
    A State is a list of tensors that contains the states of the units
    described by a model.

    For a model with L hidden layers, the tensors have shapes

    shapes = [
    (num_samples, num_visible),
    (num_samples, num_hidden_1),
                .
                .
                .
    (num_samples, num_hidden_L)
    ]

    """
    def __init__(self, tensors):
        """
        Create a State object.

        Args:
            tensors: a list of tensors

        Returns:
            state object

        """
        self.units = tensors
        self.shapes = [be.shape(t) for t in self.units]

    @classmethod
    def from_model(cls, batch_size, model):
        """
        Create a State object.

        Args:
            batch_size (int): the number of samples per layer
            model (Model): a model object

        Returns:
            state object

        """
        shapes = [(batch_size, l.len) for l in model.layers]
        units = [model.layers[i].random(shapes[i]) for i in range(model.num_layers)]
        return cls(units)

    @classmethod
    def from_visible(cls, vis, model):
        """
        Create a state object with given visible unit values.

        Args:
            vis (tensor (num_samples, num_visible)): visible unit values.
            model (Model): a model object

        Returns:
            state object

        """
        batch_size = be.shape(vis)[0]
        state = cls.from_model(batch_size, model)
        state.units[0] = vis
        return state

    @classmethod
    def from_state(cls, state):
        """
        Create a State object from an existing State.

        Args:
            state (State): a State instance

        Returns:
            state object

        """
        return copy.deepcopy(state)


class LayerConnection(object):
    """
    Container to manage how a layer is used in a model.
    Holds attributes and which layers it is connected to.

    """
    def __init__(self):
        """
        Constructor

        Args:
            left_connected_layers: a list of ConnectedLayer tuples
                for shallower connected layers.
            right_connected_layers: a list of ConnectedLayer tuples
                for deeper connected layers.
            sampling_clamped: whether this layer's sampling is fixed.
            gradient_clamped: whether this layer's gradient is fixed.
            excluded: whether this layer is excluded from the model.

        Returns:
            None

        """
        self.left_connected_layers = []
        self.right_connected_layers = []
        self.left_connected_weights = []
        self.right_connected_weights = []
        self.sampling_clamped = False
        self.gradient_clamped = False
        self.excluded = False


class WeightConnection(object):
    """
    Container to manage how Weights layers are used in a model.
    Holds attributes and which layers are connected to it.

    """
    def __init__(self,
                 left_layer=None,
                 right_layer=None,
                 gradient_clamped=False):
        """
        Constructor

        Args:
            left_layer: the shallower connected layer.
            right_layer: the deeper connected layer.
            gradient_clamped: whether this layer's gradient is fixed.

        """
        self.left_layer = left_layer
        self.right_layer = right_layer
        self.gradient_clamped = gradient_clamped


class ComputationGraph(object):
    """
    Manages the connections between layers in a model.
    Main layers or weight layers can have various properties set:
        - Clamped sampling, main layers only: the layer is not sampled
            (state is unchanged)
        - Clamped gradient: the gradient is not updated
        - Excluded layers: the layer is removed from the compute graph,
            and the visible-side weight layers are removed.
            Only valid if the subsequent layer is the same size.

    The computation graph is defined by
        - weight_connections: a list of WeightConnection objects.

    Used to set the connections for primary layers and for weight layers.
    Also used to modify connections, for example in layerwise training.

    """
    def __init__(self, num_layers, weight_connections=[]):
        """
        Instantiates with default connections, unless connections are specified.

        Args:
            num_layers: the number of layers
            weight_connections: a list of WeightConnection (default None)

        Returns:
            None

        """
        self.num_layers = num_layers
        self.weight_connections = weight_connections
        self.layer_connections = None

        # set the default connections if none given
        if not self.weight_connections:
            self.weight_connections = self.default_weight_connections()
        # the layer connections are inferred
        self.set_layer_connections()

    def default_weight_connections(self):
        """
        Creates the default weight connections.
        Assumes the layers are linearly connected, with the first being visible.

        Args:
            None

        Returns:
            weight_connections: a list of WeightConnection objects

        """
        return [WeightConnection(i, i+1) for i in range(self.num_layers-1)]

    def set_layer_connections(self):
        """
        Creates layer connections from `weight_connections`.
        Sets the value to the attribute `layer_connections`.

        Args:
            None

        Returns:
            None

        """
        layer_connections = []
        for i in range(self.num_layers):
            lc = LayerConnection()
            # loop over all weights
            for iw, w in enumerate(self.weight_connections):
                # if the layer is connected on the shallower side
                if w.left_layer == i:
                    lc.right_connected_layers.append(w.right_layer)
                    lc.right_connected_weights.append(iw)
                # or if it's connected on the deeper side
                elif w.right_layer == i:
                    lc.left_connected_layers.append(w.left_layer)
                    lc.left_connected_weights.append(iw)
            layer_connections.append(lc)
        self.layer_connections = layer_connections

    def set_clamped_sampling(self, clamped_layers):
        """
        Convenience function to set the layers which have sampling clamped.
        Sets exactly the given layers as clamped (e.g. unclamps any others).

        Args:
            clamped_layers (List): the exact set of layers to clamp

        Returns:
            None

        """
        for i in range(self.num_layers):
            self.layer_connections[i].sampling_clamped = (i in clamped_layers)


class Model(object):
    """
    General model class.
    (i.e., Restricted Boltzmann Machines).

    Example usage:
    '''
    vis = BernoulliLayer(nvis)
    hid = BernoulliLayer(nhid)
    rbm = Model([vis, hid])
    '''

    """
    def __init__(self, layer_list):
        """
        Create a model.

        Args:
            layer_list: A list of layers objects.

        Returns:
            model: A model.

        """
        # the layers are stored in a list with the visible units
        # as the zeroth element
        self.layers = layer_list
        self.num_layers = len(self.layers)
        self.graph = ComputationGraph(self.num_layers)

        # set the weights
        self.weights = [layers.Weights(
                            (self.layers[w.left_layer].len,
                             self.layers[w.right_layer].len
                            )
                        ) for w in self.graph.weight_connections]
        self.num_weights = len(self.weights)

    def get_config(self) -> dict:
        """
        Get a configuration for the model.

        Notes:
            Includes metadata on the layers.

        Args:
            None

        Returns:
            A dictionary configuration for the model.

        """
        config = {
            "model type": "RBM",
            "layers": [ly.get_config() for ly in self.layers],
        }
        return config

    @classmethod
    def from_config(cls, config: dict):
        """
        Build a model from the configuration.

        Args:
            A dictionary configuration of the model metadata.

        Returns:
            An instance of the model.

        """
        layer_list = []
        for ly in config["layers"]:
            layer_list.append(layers.Layer.from_config(ly))
        return cls(layer_list)

    def initialize(self, data, method: str='hinton') -> None:
        """
        Initialize the parameters of the model.

        Args:
            data: A Batch object.
            method (optional): The initialization method.

        Returns:
            None

        """
        try:
            func = getattr(init, method)
        except AttributeError:
            print(method + ' is not a valid initialization method for latent models')
        func(data, self)
        for l in self.layers:
            l.enforce_constraints()
        for w in self.weights:
            w.enforce_constraints()

    def random(self, vis):
        """
        Generate a random sample with the same shape,
        and of the same type, as the visible units.

        Args:
            vis: The visible units.

        Returns:
            tensor: Random sample with same shape as vis.

        """
        return self.layers[0].random(vis)

    def _connected_rescaled_units(self, i, state):
        """
        Helper function to retrieve the rescaled units connected to layer i.

        Args:
            i (int): the index of the layer of interest
            state (State): the current state of the units

        Returns:
            list[tensor]: the rescaled values of the connected units

        """
        connected_layers = self.graph.layer_connections[i].left_connected_layers \
                        + self.graph.layer_connections[i].right_connected_layers
        return [self.layers[j].rescale(state.units[j]) for j in connected_layers]

    def _connected_weights(self, i):
        """
        Helper function to retrieve the values of the weights connecting
        layer i to its neighbors.

        Args:
            i (int): the index of the layer of interest

        Returns:
            list[tensor]: the weights connecting layer i to its neighbros

        """
        left_weights = self.graph.layer_connections[i].left_connected_weights
        right_weights = self.graph.layer_connections[i].right_connected_weights
        return [self.weights[j].W() for j in left_weights] + [self.weights[j].W_T() for j in right_weights]

    def _alternating_update(self, func_name, state, beta=None):
        """
        Performs a single Gibbs sampling update in alternating layers.
        state -> new state

        Args:
            func_name (str, function name): layer function name to apply to the units to sample
            state (State object): the current state of each layer
            beta (optional, tensor (batch_size, 1)): Inverse temperatures

        Returns:
            new state

        """
        updated_state = State.from_state(state)

        # update the odd then the even layers
        for layer_set in [range(1, self.num_layers, 2),
                          range(0, self.num_layers, 2)]:
            for i in layer_set:
                if not self.graph.layer_connections[i].sampling_clamped:
                    func = getattr(self.layers[i], func_name)
                    updated_state.units[i] = func(
                        self._connected_rescaled_units(i, updated_state),
                        self._connected_weights(i),
                        beta)

        return updated_state

    def _cyclic_update(self, func_name, state, beta=None):
        """
        Performs a single Gibbs sampling update, cycling through the layers.
        Updates 1 -> n -> 0.
        state -> new state

        Args:
            func_name (str, function name): layer function name to apply to the units to sample
            state (State object): the current state of each layer
            beta (optional, tensor (batch_size, 1)): Inverse temperatures

        Returns:
            new state

        """
        updated_state = State.from_state(state)

        # update in sequence
        update_sequence = list(range(1,self.num_layers)) + list(range(self.num_layers-2,-1,-1))
        for i in update_sequence:
            if not self.graph.layer_connections[i].sampling_clamped:
                func = getattr(self.layers[i], func_name)
                updated_state.units[i] = func(
                    self._connected_rescaled_units(i, updated_state),
                    self._connected_weights(i),
                    beta)

        return updated_state

    def markov_chain(self, n, state, beta=None) -> State:
        """
        Perform multiple Gibbs sampling steps in alternating layers.
        state -> new state

        Notes:
            Samples layers according to the conditional probability
            on adjacent layers,
            x_i ~ P(x_i | x_(i-1), x_(i+1) )

        Args:
            n (int): number of steps.
            state (State object): the current state of each layer
            beta (optional, tensor (batch_size, 1)): Inverse temperatures

        Returns:
            new state

        """
        new_state = State.from_state(state)
        for _ in range(n):
            new_state = self._alternating_update('conditional_sample',
                                                 new_state,
                                                 beta)
        return new_state

    def mean_field_iteration(self, n, state, beta=None):
        """
        Perform multiple mean-field updates in alternating layers
        states -> new state

        Notes:
            Returns the expectation of layer units
            conditioned on adjacent layers,
            x_i = E[x_i | x_(i-1), x_(i+1) ]

        Args:
            n (int): number of steps.
            state (State object): the current state of each layer
            beta (optional, tensor (batch_size, 1)): Inverse temperatures

        Returns:
            new state

        """
        new_state = State.from_state(state)
        for _ in range(n):
            new_state = self._alternating_update('conditional_mean',
                                                 new_state,
                                                 beta)
        return new_state

    def deterministic_iteration(self, n, state, beta=None):
        """
        Perform multiple deterministic (maximum probability) updates
        in alternating layers.
        state -> new state

        Notes:
            Returns the layer units that maximize the probability
            conditioned on adjacent layers,
            x_i = argmax P(x_i | x_(i-1), x_(i+1))

        Args:
            n (int): number of steps.
            state (State object): the current state of each layer
            beta (optional, tensor (batch_size, 1)): Inverse temperatures

        Returns:
            new state

        """
        new_state = State.from_state(state)
        for _ in range(n):
            new_state = self._alternating_update('conditional_mode',
                                                 new_state,
                                                 beta)
        return new_state

    def gradient(self, data_state, model_state):
        """
        Compute the gradient of the model parameters.
        Scales the units in the state and computes the gradient.

        Args:
            data_state (State object): The observed visible units and sampled hidden units.
            model_state (State objects): The visible and hidden units sampled from the model.

        Returns:
            dict: Gradients of the model parameters.

        """
        grad = gu.Gradient(
            [None for l in self.layers],
            [None for w in self.weights]
        )

        update_layers = [i for i in range(self.num_layers) \
            if not self.graph.layer_connections[i].gradient_clamped]
        update_weights = [i for i in range(self.num_weights) \
            if not self.graph.weight_connections[i].gradient_clamped]

        # POSITIVE PHASE (using observed)

        # compute the postive phase of the gradients of the layer parameters
        for i in range(self.num_layers):
            if i in update_layers:
                grad.layers[i] = self.layers[i].derivatives(
                    data_state.units[i],
                    self._connected_rescaled_units(i, data_state),
                    self._connected_weights(i)
                )

        # compute the positive phase of the gradients of the weights
        for i in range(self.num_weights):
            if i in update_weights:
                iL = self.graph.weight_connections[i].left_layer
                iR = self.graph.weight_connections[i].right_layer
                grad.weights[i] = self.weights[i].derivatives(
                    self.layers[iL].rescale(data_state.units[iL]),
                    self.layers[iR].rescale(data_state.units[iR]),
                )

        # NEGATIVE PHASE (using sampled)

        # update the gradients of the layer parameters with the negative phase
        for i in range(self.num_layers):
            if i in update_layers:
                grad.layers[i] = be.mapzip(be.subtract,
                    self.layers[i].derivatives(
                        model_state.units[i],
                        self._connected_rescaled_units(i, model_state),
                        self._connected_weights(i)
                    ),
                grad.layers[i])
            else:
                grad.layers[i] = self.layers[i].get_null_params()

        # update the gradients of the weight parameters with the negative phase
        for i in range(self.num_weights):
            if i in update_weights:
                iL = self.graph.weight_connections[i].left_layer
                iR = self.graph.weight_connections[i].right_layer
                grad.weights[i] = be.mapzip(be.subtract,
                    self.weights[i].derivatives(
                        self.layers[iL].rescale(model_state.units[iL]),
                        self.layers[iR].rescale(model_state.units[iR]),
                    ),
                grad.weights[i])
            else:
                grad.weights[i] = self.weights[i].get_null_params()

        return grad

    def parameter_update(self, deltas):
        """
        Update the model parameters.

        Notes:
            Modifies the model parameters in place.

        Args:
            deltas (Gradient)

        Returns:
            None

        """
        for i in range(self.num_layers):
            self.layers[i].parameter_step(deltas.layers[i])
        for i in range(self.num_weights):
            self.weights[i].parameter_step(deltas.weights[i])

    def joint_energy(self, data):
        """
        Compute the joint energy of the model based on a state.

        Args:
            data (State object): the current state of each layer

        Returns:
            tensor (num_samples,): Joint energies.

        """
        energy = 0
        for i in range(self.num_layers):
            energy += self.layers[i].energy(data.units[i])
        for i in range(self.num_weights):
            iL = self.graph.weight_connections[i].left_layer
            iR = self.graph.weight_connections[i].right_layer
            energy += self.weights[i].energy(data.units[iL], data.units[iR])
        return energy

    def save(self, store: pandas.HDFStore) -> None:
        """
        Save a model to an open HDFStore.

        Notes:
            Performs an IO operation.

        Args:
            store (pandas.HDFStore)

        Returns:
            None

        """
        # save the config as an attribute
        config = self.get_config()
        store.put('model', pandas.DataFrame())
        store.get_storer('model').attrs.config = config
        # save the weights
        for i in range(self.num_weights):
            key = os.path.join('weights', 'weights'+str(i))
            self.weights[i].save_params(store, key)
        for i in range(self.num_layers):
            key = os.path.join('layers', 'layers'+str(i))
            self.layers[i].save_params(store, key)

    @classmethod
    def from_saved(cls, store: pandas.HDFStore) -> None:
        """
        Build a model by reading from an open HDFStore.

        Notes:
            Performs an IO operation.

        Args:
            store (pandas.HDFStore)

        Returns:
            None

        """
        # create the model from the config
        config = store.get_storer('model').attrs.config
        model = cls.from_config(config)
        # load the weights
        for i in range(len(model.weights)):
            key = os.path.join('weights', 'weights'+str(i))
            model.weights[i].load_params(store, key)
        # load the layer parameters
        for i in range(len(model.layers)):
            key = os.path.join('layers', 'layers'+str(i))
            model.layers[i].load_params(store, key)
        return model
