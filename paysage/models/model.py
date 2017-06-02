import os
import pandas

from .. import layers
from .. import backends as be
from .initialize import init_model as init
from . import gradient_util as gu
from . import model_utils as mu


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
        self.graph = mu.ComputationGraph(self.num_layers)

        # set the weights
        self.weights = [layers.Weights(
                            (self.layers[weight_index[0]].len,
                             self.layers[weight_index[1]].len
                            )
                        ) for weight_index in self.graph.weight_connections]
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
        connections = self.graph.layer_connections[i]
        return [self.layers[conn.layer].rescale(state.units[conn.layer]) \
                for conn in connections]

    def _connected_weights(self, i):
        """
        Helper function to retrieve the values of the weights connecting
        layer i to its neighbors.

        Args:
            i (int): the index of the layer of interest

        Returns:
            list[tensor]: the weights connecting layer i to its neighbros

        """
        connections = self.graph.layer_connections[i]
        return [self.weights[conn.weight].W_T() if conn.is_forward \
                else self.weights[conn.weight].W() \
                for conn in connections]

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
        updated_state = mu.State.from_state(state)

        # find layers that can be sampled
        sampled_layers = self.graph.get_sampled()

        # define the two sampling sets to alternate between
        layer_set_A = [i for i in range(1, self.num_layers, 2) if i in sampled_layers]
        layer_set_B = [i for i in range(0, self.num_layers, 2) if i in sampled_layers]

        # update the odd then the even layers
        for layer_set in [layer_set_A, layer_set_B]:
            for i in layer_set:
                func = getattr(self.layers[i], func_name)
                updated_state.units[i] = func(
                    self._connected_rescaled_units(i, updated_state),
                    self._connected_weights(i),
                    beta)

        return updated_state

    def markov_chain(self, n, state, beta=None) -> mu.State:
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
        new_state = mu.State.from_state(state)
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
        new_state = mu.State.from_state(state)
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
        new_state = mu.State.from_state(state)
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

        # POSITIVE PHASE (using observed)

        # compute the postive phase of the gradients of the layer parameters
        for i in range(self.num_layers):
            grad.layers[i] = self.layers[i].derivatives(
                data_state.units[i],
                self._connected_rescaled_units(i, data_state),
                self._connected_weights(i)
            )

        # compute the positive phase of the gradients of the weights
        for i in range(self.num_weights):
            iL = self.graph.weight_connections[i][0]
            iR = self.graph.weight_connections[i][1]
            grad.weights[i] = self.weights[i].derivatives(
                self.layers[iL].rescale(data_state.units[iL]),
                self.layers[iR].rescale(data_state.units[iR]),
            )

        # NEGATIVE PHASE (using sampled)

        # update the gradients of the layer parameters with the negative phase
        for i in range(self.num_layers):
            grad.layers[i] = be.mapzip(be.subtract,
                self.layers[i].derivatives(
                    model_state.units[i],
                    self._connected_rescaled_units(i, model_state),
                    self._connected_weights(i)
                ),
            grad.layers[i])

        # update the gradients of the weight parameters with the negative phase
        for i in range(self.num_weights):
            iL = self.graph.weight_connections[i][0]
            iR = self.graph.weight_connections[i][1]
            grad.weights[i] = be.mapzip(be.subtract,
                self.weights[i].derivatives(
                    self.layers[iL].rescale(model_state.units[iL]),
                    self.layers[iR].rescale(model_state.units[iR]),
                ),
            grad.weights[i])

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
        for layer_index in range(self.num_layers):
            if layer_index in self.graph.trainable_layers:
                self.layers[layer_index].parameter_step(deltas.layers[layer_index])
        for weight_index in range(self.num_weights):
            if weight_index in self.graph.trainable_weights:
                self.weights[weight_index].parameter_step(deltas.weights[weight_index])

    def joint_energy(self, data):
        """
        Compute the joint energy of the model based on a state.

        Args:
            data (State object): the current state of each layer

        Returns:
            tensor (num_samples,): Joint energies.

        """
        energy = 0
        for layer_index in range(self.num_layers):
            energy += self.layers[layer_index].energy(data.units[layer_index])
        for weight_index in range(self.num_weights):
            iL = self.graph.weight_connections[weight_index][0]
            iR = self.graph.weight_connections[weight_index][1]
            energy += self.weights[weight_index].energy(data.units[iL], data.units[iR])
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
        # save the parameters
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
