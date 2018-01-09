import copy
import numpy as np
from collections import namedtuple
from copy import deepcopy

from .. import backends as be


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
    def dropout_mask(cls, model, batch_size=1):
        """
        Create a list of layer masks for performing dropout

        Args:
            model (Model): a model object
            batch_size (int): number of masks per layer

        Returns:
            state: stores a mask tensor for each layer

        """
        if model.use_dropout():
            units = [lay.get_dropout_mask(batch_size) for lay in model.layers]
            return cls(units)
        else:
            return None

    @classmethod
    def dropout_rescale(cls, model, batch_size=1):
        """
        Create a list of tensors used for rescaling units according to dropout percentages

        Args:
            model (Model): a model object

        Returns:
            state: (1-layer.dropout_p) for each unit

        """
        if model.use_dropout():
            units = [lay.get_dropout_scale(batch_size) for lay in model.layers]
            return cls(units)
        else:
            return None

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
        # randomly initialize the state
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
        return cls([be.copy_tensor(t) for t in state.units])


class StateTAP(object):
    """
    A StateTAP is a list of CumulantsTAP objects for each layer in the model.

    """
    def __init__(self, cumulants):
        """
        Create a StateTAP.

        Args:
            cumulants: list of CumulantsTAP objects

        Returns:
            StateTAP

        """
        self.cumulants = cumulants
        self.len = len(self.cumulants)

    @classmethod
    def from_state(cls, state):
        """
        Create a StateTAP object from an existing StateTAP.

        Args:
            state (StateTAP): a StateTAP instance

        Returns:
            StateTAP object

        """
        return deepcopy(state)

    @classmethod
    def from_model(cls, model):
        """
        Create a StateTAP object from a model.

        Args:
            model (Model): a Model instance

        Returns:
            StateTAP object

        """
        return cls([layer.get_zero_magnetization() for layer in model.layers])

    @classmethod
    def from_model_rand(cls, model):
        """
        Create a StateTAP object from a model.

        Args:
            model (Model): a Model instance

        Returns:
            StateTAP object

        """
        return cls([layer.get_random_magnetization() for layer in model.layers])


def dropout_state(state: State, dropout_mask: State) -> State:
    """
    Apply a dropout mask to a state.

    Args:
        state (State): the state of the units
        dropout_mask (State): the dropout mask

    Returns:
        new state

    """
    if dropout_mask is not None:
        return State(be.mapzip(be.multiply, dropout_mask.units, state.units))
    else:
        return state

def dropout_state_(state: State, dropout_mask: State) -> None:
    """
    Apply a dropout mask to a state.

    Notes:
        Changes state in place!

    Args:
        state (State): the state of the units
        dropout_mask (State): the dropout mask

    Returns:
        None

    """
    if dropout_mask is not None:
        return be.mapzip_(be.multiply_, dropout_mask.units, state.units)


class Graph(object):
    """
    Container for the contents of a graph.
    Allows operations to extract useful objects,
        such as the adjacency matrix, adjacency list, and edge list.

    """
    def __init__(self, incidence_matrix):
        self.incidence_matrix = incidence_matrix
        self.num_vertices = incidence_matrix.shape[0]
        self.num_edges = incidence_matrix.shape[1]

        self.edge_list = self._edge_list()
        self.adjacency_matrix = self._adjacency_matrix()
        self.adjacency_list = self._adjacency_list()

    def _edge_list(self):
        """
        Get the edge list.
        Needs a current incidence matrix.
        Assumes there are exactly 2 vertices per edge.

        Args:
            None

        Returns:
            edge_list (List): the list of edges in the graph.

        """
        return np.array([np.nonzero(col)[0] for col in self.incidence_matrix.T])

    def _adjacency_matrix(self):
        """
        Get the adjacency matrix.
        Needs a current incidence matrix and edge list.

        Args:
            None

        Returns:
            adjacency_matrix (numpy array): the adjacency matrix

        """
        adj = np.zeros((self.num_vertices, self.num_vertices))
        for edge in self.edge_list:
            adj[edge, edge[::-1]] = 1
        return adj

    def _adjacency_list(self):
        """
        Get the adjacency list.
        Needs a current adjacency matrix.

        Args:
            None

        Returns:
            adjacency_list (List): the adjacency list
        """
        return [np.nonzero(row)[0] for row in self.adjacency_matrix]


"""
A tuple parameterizing the connection from one layer to another.
The `is_forward` attribute is True if the `layer` attribute (the connected layer)
    has a higher index than the index of this layer.
"""
LayerConnection = namedtuple("LayerConnection", ["layer", "weight", "is_forward"])


class ComputationGraph(object):
    """
    Manages the connections between layers in a model.
    Layers can have various properties set:
        - Clamped sampling: the layer is not sampled (state unchanged)
        - Trainable: the layer's parameters can be changed
    Weight layers (connecting two other layers) can have
        the trainable property set.

    The computation graph is defined by an Graph object, where
    layers are vertices and weight layers are edges.
    The incidence matrix defines the connections between layers
    and the corresponding edges.

    """
    def __init__(self, num_layers):
        """
        Builds the default incidence matrix.

        Args:
            num_layers: the number of layers in the model

        Returns:
            None

        """
        # build the incidence matrix
        incidence_matrix = self.default_incidence_matrix(num_layers)
        self.num_layers = num_layers
        self.num_weights = incidence_matrix.shape[1]

        # set the connections between layers
        self.connections = Graph(incidence_matrix)
        self.layer_connections = self.get_layer_connections()
        self.weight_connections = self.connections.edge_list

        # the default properties
        # all layers can be sampled, trained, all weights can be trained
        self.clamped_sampling = []
        self.trainable_layers = range(self.num_layers)
        self.trainable_weights = range(self.num_weights)

    def default_incidence_matrix(self, num_layers):
        """
        Builds the default incidence matrix.

        Args:
            None

        Returns:
            incidence_matrix: a matrix specifying the connections in the model.

        """
        incidence_matrix = np.zeros((num_layers, num_layers-1))
        for i in range(num_layers-1):
            incidence_matrix[i, i] = 1
            incidence_matrix[i+1, i] = 1
        return incidence_matrix

    def get_layer_connections(self):
        """
        Returns a list over layers.
        Each element of the list is a list of LayerConnection tuples,
        which are the connections to that layer.

        Args:
            None

        Returns:
            layer_connections (List): a list over layers, where each entry is
                a list of LayerConnection tuples of the connections to that layer.

        """
        layer_connections = []
        for layer_index in range(self.num_layers):
            layer_conn = []
            # find all connected weights, associate to layers
            connected_weights = np.nonzero(self.connections.incidence_matrix[layer_index])[0]
            for weight in connected_weights:
                edge_conn = self.connections.edge_list[weight]
                # if the connecting layer has a higher index,
                # set LayerConnection.is_forward = True
                if edge_conn[0] == layer_index:
                    layer_conn.append(LayerConnection(edge_conn[1], weight, True))
                else:
                    layer_conn.append(LayerConnection(edge_conn[0], weight, False))
            layer_connections.append(layer_conn)
        return layer_connections

    def set_clamped_sampling(self, clamped_sampling):
        """
        Convenience function to set the layers for which sampling is clamped.
        Sets exactly the given layers to have sampling clamped.

        Args:
            clamped_sampling (List): the exact set of layers which are have sampling clamped.

        Returns:
            None

        """
        self.clamped_sampling = list(clamped_sampling)

    def get_sampled(self):
        """
        Convenience function that returns the layers for which sampling is
        not clamped.
        Complement of the `clamped_sampling` attribute.

        Args:
            None

        Returns:
            unclamped_sampling (List): layers for which sampling is not clamped.

        """
        return [i for i in range(self.num_layers) if i not in self.clamped_sampling]

    def set_trainable_layers(self, trainable_layers):
        """
        Convenience function to set the layers which are trainable
        Sets exactly the given layers as trainable.
        Sets weights untrainable if the higher index layer is untrainable.

        Args:
            trainable_layers (List): the exact set of layers which are trainable

        Returns:
            None

        """
        self.trainable_layers = list(trainable_layers)
        # set weights where an untrainable layer is a higher index to untrainable
        untrainable_layers = [i for i in range(self.num_layers) if i not in self.trainable_layers]
        untrainable_weights = np.where(np.in1d(self.connections.edge_list.T[1], untrainable_layers))[0]
        self.trainable_weights = [i for i in range(self.num_weights) if i not in untrainable_weights]
