import copy
import numpy as np

from .. import backends as be


"""
Comments for drckf:

A deep boltzmann machine typically has a structure like:

L_0 : W_0 : L_1 : W_1 : L_2 ~ layer : weight : layer : weight : layer

the connectivity can be represented by a graph where the layers are "verticies"
and the weights are "edges". below, i put quotation marks around the usual
graph theory terms

the "adjacency matrix" for the layers is:
     L_0 L_1 L_2
L_0: 0   1   0
L_1: 1   0   1
L_2: 0   1   0

so that the layer connections (i.e., the "adjacency list") are:
0: [1]
1: [0, 2]
2: [1]

notice that if the adjacency matrix is A then the connections of layer i
are given by A[i].nonzero() (works for both numpy and torch tensors -- though,
they return slightly different objects so need a backend function to standardize
the result).

the "unoriented incidence matrix" of the graph is:
     W_0 W_1
L_0: 1   0
L_1: 1   1
L_2: 0   1

the weight connections (i.e., the "edge list") are:
0: [0, 1]
1: [1, 2]

Thoughts:

"Gradient Clampling" seems like a natural thing to put in the layer itself,
because it can be handled easily in the `layer.parameter_step` method.

"""

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


class IncidenceMatrix(object):
    """
    Holds the incidence matrix for a graph.
    Allows operations to extract the adjacency matrix, adjacency list, and edge list.

    """
    def __init__(self, incidence_matrix):
        self.incidence_matrix = None
        self.edge_list = None
        self.adjacency_matrix = None
        self.adjacency_list = None

        # update the other graph attributes
        self.update(incidence_matrix)

    def update(self, incidence_matrix):
        """
        Update the incidence_matrix, edge list,
            adjacency matrix, adjacency list, and edge list attributes.

        Args:
            None

        Returns:
            None

        """
        self.incidence_matrix = incidence_matrix
        self.edge_list = self._edge_list()
        self.adjacency_matrix = self._adjacency_matrix()
        self.adjacency_list = self._adjacency_list()

    def _edge_list(self):
        """
        Get the edge list.
        Needs a current incidence matrix.

        Args:
            None

        Returns:
            edge_list (List): the list of edges in the graph.

        """
        return [np.nonzero(col)[0] for col in self.incidence_matrix.T]

    def _adjacency_matrix(self):
        """
        Get the adjacency matrix.
        Needs a current incidence matrix and edge list.

        Args:
            None

        Returns:
            adjacency_matrix (numpy array): the adjacency matrix

        """
        adj = np.zeros(2*[len(self.incidence_matrix)])
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


class ComputationGraph(object):
    """
    Manages the connections between layers in a model.
    Layers can have various properties set:
        - Clamped sampling: the layer is not sampled (state unchanged)
        - Trainable: the layer's parameters can be changed
    Weight layers (connecting two other layers) can have
        the trainable property set.

    The computation graph is defined by an IncidenceMatrix, where
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
        self.num_layers = num_layers

        # build the incidence matrix
        incidence_matrix = self.default_incidence_matrix()

        # set the connections between layers
        self.update_connections(incidence_matrix)

        # the default properties
        # all layers can be sampled, trained, all weights can be trained
        self.clamped_sampling = []
        self.trainable_layers = range(len(self.layer_connections))
        self.trainable_weights = range(len(self.weight_connections))

    def default_incidence_matrix(self):
        """
        Builds the default incidence matrix.

        Args:
            None

        Returns:
            incidence_matrix: a matrix specifying the connections in the model.

        """
        incidence_matrix = np.zeros((self.num_layers, self.num_layers-1))
        for i in range(self.num_layers-1):
            incidence_matrix[i, i] = 1
            incidence_matrix[i+1, i] = 1
        return incidence_matrix

    def update_connections(self, incidence_matrix):
        """
        Update the connections in the model.
        Modifies the connections attributes.

        Args:
            incidence_matrix: a matrix specifying the connections in the model.

        Returns:
            None

        """
        self.connections = IncidenceMatrix(incidence_matrix)

        # define the connections between layers
        self.layer_connections = self.connections.adjacency_list
        self.weight_connections = self.connections.edge_list

    def set_clamped_sampling(self, clamped_sampling):
        """
        Convenience function to set the layers for which sampling is clamped.
        Sets exactly the given layers to have sampling clamped.

        Args:
            clamped_sampling (List): the exact set of layers which are have sampling clamped.

        Returns:
            None

        """
        self.clamped_sampling = clamped_sampling

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
        self.trainable_layers = trainable_layers
        # set weights where an untrainable layer is a higher index to untrainable
        untrainable_layers = list(set(range(len(self.layer_connections))) - set(trainable_layers))
        untrainable_weights = []
        for weight_index, weight_con in enumerate(self.weight_connections):
            if weight_con[1] in untrainable_layers:
                untrainable_weights.append(weight_index)
        self.trainable_weights = sorted(list(set(range(len(self.weight_connections)))
                                            - set(untrainable_weights)))
