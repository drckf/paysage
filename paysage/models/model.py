import os
import copy
import pandas

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


class Model(object):
    """
    General model class.
    Currently only supports models with 2 layers,
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

        Notes:
            Only 2-layer models currently supported.

        Args:
            layer_list: A list of layers objects.

        Returns:
            model: A model.

        """
        # the layers are stored in a list with the visible units
        # as the zeroth element
        self.layers = layer_list
        self.num_layers = len(self.layers)
        self.layer_connections = self._layer_connections()
        self.weight_connections = self._weight_connections()

        assert self.num_layers == 2,\
        "Only models with 2 layers are currently supported"

        # adjacent layers are connected by weights
        # therefore, if there are len(layers) = n then len(weights) = n - 1
        self.weights = [
            layers.Weights((self.layers[i].len, self.layers[i+1].len))
        for i in range(self.num_layers - 1)
        ]

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

    def initialize(self, data, method: str='hinton'):
        """
        Inialize the parameters of the model.

        Args:
            data: A Batch object.
            method (optional): The initalization method.

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

    def _layer_connections(self):
        """
        Helper function to enumerate the connections between layers.
        List of list of indices of each layer connected to the layer.
        e.g. for a 4-layer model the connections are [[1], [0, 2], [1, 3], [2]].

        Args:
            None

        Returns:
            list: Indices of connecting layers.

        """
        return [[j for j in [i-1,i+1] if 0<=j<self.num_layers]
                   for i in range(self.num_layers)]

    def _weight_connections(self):
        """
        Helper function to enumerate the connections between weights and layers.
        List of list of indices of each weight layer connected to the layer.
        e.g. for a 4-layer model the connections are [[0], [0, 1], [1, 2], [2]].

        Args:
            None

        Returns:
            list: Indices of connecting weight layers.

        """
        return [[j for j in [i-1,i] if 0<=j<self.num_layers-1]
                   for i in range(self.num_layers)]

    def _alternating_update(self, func_name, state, beta=None, clamped=[]):
        """
        Performs a single Gibbs sampling update in alternating layers.
        state -> new state

        Args:
            func_name (str, function name): layer function name to apply to the units to sample
            state (State object): the current state of each layer
            beta (optional, tensor (batch_size, 1)): Inverse temperatures
            clamped (list): list of layer indices to clamp (no update)

        Returns:
            new state

        """
        updated_state = State.from_state(state)

        # update the odd then the even layers
        for ll in [range(1, self.num_layers, 2), range(0, self.num_layers, 2)]:
            for i in ll:
                if i in clamped:
                    continue
                else:
                    self.layers[i].update(
                        [self.layers[j].rescale(updated_state.units[j])
                            for j in self.layer_connections[i]],
                        [self.weights[j].W() if j < i else self.weights[j].W_T()
                            for j in self.weight_connections[i]],
                        beta)

                    updated_state.units[i] = getattr(self.layers[i], func_name)()

        return updated_state

    def markov_chain(self, n, state, beta=None, clamped=[]):
        """
        Perform multiple Gibbs sampling steps in alternating layers.
        state -> new state

        Args:
            n (int): number of steps.
            state (State object): the current state of each layer
            beta (optional, tensor (batch_size, 1)): Inverse temperatures
            clamped (list): list of layer indices to clamp

        Returns:
            new state

        """
        new_state = State.from_state(state)
        for _ in range(n):
            new_state = self._alternating_update('sample_state',
                                                 new_state,
                                                 beta,
                                                 clamped)
        return new_state

    def mean_field_iteration(self, n, state, beta=None, clamped=[]):
        """
        Perform multiple mean-field updates in alternating layers
        states -> new state

        Args:
            n (int): number of steps.
            state (State object): the current state of each layer
            beta (optional, tensor (batch_size, 1)): Inverse temperatures
            clamped (list): list of layer indices to clamp

        Returns:
            new state

        """
        new_state = State.from_state(state)
        for _ in range(n):
            new_state = self._alternating_update('mean',
                                                 new_state,
                                                 beta,
                                                 clamped)
        return new_state

    def deterministic_iteration(self, n, state, beta=None, clamped=[]):
        """
        Perform multiple deterministic (maximum probability) updates
        in alternating layers.
        state -> new state

        Args:
            n (int): number of steps.
            state (State object): the current state of each layer
            beta (optional, tensor (batch_size, 1)): Inverse temperatures
            clamped (list): list of layer indices to clamp

        Returns:
            new state

        """
        new_state = State.from_state(state)
        for _ in range(n):
            new_state = self._alternating_update('mode',
                                                 new_state,
                                                 beta,
                                                 clamped)
        return new_state

    def gradient(self, data_state, model_state):
        """
        Compute the gradient of the model parameters.
        Updates the states for the positive and negative phases,
        and computes the gradient from the unit values.

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

        # update the hidden layers
        new_data_state = self.mean_field_iteration(1, data_state, clamped=[0])
        # compute the gradients
        for i in range(self.num_layers):
            grad.layers[i] = self.layers[i].derivatives(
                new_data_state.units[i],
                [self.layers[j].rescale(new_data_state.units[j])
                    for j in self.layer_connections[i]],
                [self.weights[j].W() if j < i else self.weights[j].W_T()
                    for j in self.weight_connections[i]],
            )
        for i in range(self.num_layers - 1):
            grad.weights[i] = self.weights[i].derivatives(
                self.layers[i].rescale(new_data_state.units[i]),
                self.layers[i+1].rescale(new_data_state.units[i+1]),
            )

        # NEGATIVE PHASE (using sampled)

        # update the hidden layers
        new_model_state = self.mean_field_iteration(1, model_state, clamped=[0])
        # compute the gradients
        for i in range(self.num_layers):
            grad.layers[i] = be.mapzip(be.subtract,
                self.layers[i].derivatives(
                    new_model_state.units[i],
                    [self.layers[j].rescale(new_model_state.units[j])
                        for j in self.layer_connections[i]],
                    [self.weights[j].W() if j < i else self.weights[j].W_T()
                        for j in self.weight_connections[i]],
                ),
            grad.layers[i])
        for i in range(self.num_layers - 1):
            grad.weights[i] = be.mapzip(be.subtract,
                self.weights[i].derivatives(
                    self.layers[i].rescale(new_model_state.units[i]),
                    self.layers[i+1].rescale(new_model_state.units[i+1]),
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
        for i in range(self.num_layers):
            self.layers[i].parameter_step(deltas.layers[i])
        for i in range(self.num_layers - 1):
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
        for i in range(self.num_layers - 1):
            energy += self.layers[i].energy(data.units[i])
            energy += self.layers[i+1].energy(data.units[i+1])
            energy += self.weights[i].energy(data.units[i], data.units[i+1])
        return energy

    def marginal_free_energy(self, data):
        """
        Compute the marginal free energy of the model.

        If the energy is:
        E(v, h) = -\sum_i a_i(v_i) - \sum_j b_j(h_j) - \sum_{ij} W_{ij} v_i h_j
        Then the marginal free energy is:
        F(v) =  -\sum_i a_i(v_i) - \sum_j \log \int dh_j \exp(b_j(h_j) - \sum_i W_{ij} v_i)
        This can be extended to a deep model by a sum over all hidden states

        Args:
            data (State object): The current state of each layer.

        Returns:
            tensor (batch_size, ): Marginal free energies.

        """
        assert self.num_layers == 2 # supported for 2-layer models only
        i = 0
        phi = be.dot(data.units[i], self.weights[i].W())
        log_Z_hidden = self.layers[i+1].log_partition_function(phi)
        energy = 0
        energy += self.layers[i].energy(data.units[i])
        energy -= be.tsum(log_Z_hidden, axis=1)
        return energy

    def save(self, store):
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
        for i in range(self.num_layers - 1):
            key = os.path.join('weights', 'weights'+str(i))
            self.weights[i].save_params(store, key)
        for i in range(len(self.layers)):
            key = os.path.join('layers', 'layers'+str(i))
            self.layers[i].save_params(store, key)

    @classmethod
    def from_saved(cls, store):
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
