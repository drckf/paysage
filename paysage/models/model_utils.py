import copy
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

class LayerConnection(object):
    """
    Container to manage how a layer is used in a model.
    Holds attributes and which layers it is connected to.

    """
    def __init__(self,
                 left_connected_layers = None,
                 right_connected_layers = None,
                 left_connected_weights = None,
                 right_connected_weights = None,
                 sampling_clamped = False,
                 trainable = True,
                 excluded = False):
        """
        Constructor

        Args:
            left_connected_layers: a list of ConnectedLayer tuples
                for shallower connected layers.
            right_connected_layers: a list of ConnectedLayer tuples
                for deeper connected layers.
            sampling_clamped: whether this layer's sampling is fixed.
            trainable: whether this layer's paramaters can be updated.
            excluded: whether this layer is excluded from the model.

        Returns:
            None

        """
        self.left_connected_layers = left_connected_layers \
            if left_connected_layers else []
        self.right_connected_layers = right_connected_layers \
            if right_connected_layers else []
        self.left_connected_weights = left_connected_weights \
            if left_connected_weights else []
        self.right_connected_weights = right_connected_weights \
            if right_connected_weights else []
        self.sampling_clamped = sampling_clamped
        self.trainable = trainable
        self.excluded = excluded


class WeightConnection(object):
    """
    Container to manage how Weights layers are used in a model.
    Holds attributes and which layers are connected to it.

    """
    def __init__(self,
                 left_layer=None,
                 right_layer=None,
                 trainable=True):
        """
        Constructor

        Args:
            left_layer: the shallower connected layer.
            right_layer: the deeper connected layer.
            trainable: whether this layer's parameters can be updated.

        """
        self.left_layer = left_layer
        self.right_layer = right_layer
        self.trainable = trainable


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
    def __init__(self, num_layers):
        """
        Instantiates with default connections, unless connections are specified.

        Args:
            num_layers: the number of layers
            weight_connections (List): a list of WeightConnection (default None)

        Returns:
            None

        """
        self.num_layers = num_layers
        self.weight_connections = self.build_weight_connections()
        # the layer connections are built from the weight_connections attribute
        self.layer_connections = self.build_layer_connections()

    def build_weight_connections(self, excluded=[]):
        """
        Creates the weight connections.
        Assumes the layers are linearly connected, with the first being visible.

        Args:
            excluded (List): a list of layer indices excluded from the model

        Returns:
            weight_connections (List): a list of WeightConnection objects

        """
        weight_connections = []
        # set the starting layer
        left_layer_index = 0
        while left_layer_index in excluded:
            left_layer_index += 1
        # create weight connections until done
        while True:
            # find the first available connection
            right_layer_index = left_layer_index + 1
            while right_layer_index in excluded:
                right_layer_index += 1
            # break if we've exceed the available layers
            if right_layer_index >= self.num_layers:
                break

            # otherwise, create the weights layer
            weight_connections.append(WeightConnection(left_layer_index, right_layer_index))
            left_layer_index = right_layer_index
        return weight_connections

    def build_layer_connections(self):
        """
        Creates layer connections from the `weight_connections` attribute.

        Args:
            None

        Returns:
            layer_connections: a list of LayerConnection objects

        """
        layer_connections = []
        for layer_index in range(self.num_layers):
            lc = LayerConnection()
            # loop over all weights
            for weight_index, weight in enumerate(self.weight_connections):
                # if the layer is connected on the shallower side
                if weight.left_layer == layer_index:
                    lc.right_connected_layers.append(weight.right_layer)
                    lc.right_connected_weights.append(weight_index)
                # or if it's connected on the deeper side
                elif weight.right_layer == layer_index:
                    lc.left_connected_layers.append(weight.left_layer)
                    lc.left_connected_weights.append(weight_index)
            layer_connections.append(lc)
        return layer_connections

    def set_clamped_sampling(self, clamped_layers):
        """
        Convenience function to set the layers which have sampling clamped.
        Sets exactly the given layers as clamped (e.g. unclamps any others).

        Args:
            clamped_layers (List): the exact set of layers to clamp

        Returns:
            None

        """
        for layer_index in range(self.num_layers):
            self.layer_connections[layer_index].sampling_clamped = (layer_index in clamped_layers)

    def set_trainable_layers(self, trainable_layers):
        """
        Convenience function to set the layers which are trainable
        Sets exactly the given layers as trainable.
        Sets the left-connected weights as untrainable.

        Args:
            trainable_layers (List): the exact set of layers which are trainable

        Returns:
            None

        """
        for layer_index in range(self.num_layers):
            self.layer_connections[layer_index].trainable = (layer_index in trainable_layers)
            # if the layer is not trainable, set the weights as untrainable
            if not self.layer_connections[layer_index].trainable:
                for weight_index in self.layer_connections[layer_index].left_connected_weights:
                    self.weight_connections[weight_index].trainable = False

    def set_excluded_layers(self, excluded_layers):
        """
        Convenience function to set the excluded layers.
        Sets exactly the given layers as excluded (e.g. unexcludes any others).

        Args:
            excluded_layers (List): the exact set of layers to exclude

        Returns:
            None

        """
        # set the exclusions
        for layer_index in range(self.num_layers):
            self.layer_connections[layer_index].excluded = (layer_index in excluded_layers)

        # build the connections
        self.weight_connections = self.build_weight_connections(excluded=excluded_layers)
        self.layer_connections = self.build_layer_connections()
