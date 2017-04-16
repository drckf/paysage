import os
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
            vis (tensor (num_samples, num_visible))
            model (Model): a model object

        Returns:
            state object

        """
        batch_size = be.shape(vis)[0]
        state = cls.from_model(batch_size, model)
        state.units[0] = vis
        return state



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

    # it might be more clear to rename skip_layers to clamped
    def _alternating_update(self, func_name, state, beta=None, skip_layers=[]):
        """
        Performs a single Gibbs sampling update in alternating layers.
        state -> new state

        Args:
            func_name (str, function name): layer function name to apply to the units to sample
            state (State object): the current state of each layer
            beta (optional, tensor (batch_size, 1)): Inverse temperatures
            skip_layers (list): list of layer indices to skip updates

        Returns:
            new state

        """

        #
        # i'm not sure this function is correct
        # it looks like the states corresponding to a layer in skip_layers
        # would still be modified by this function (i checked this, they are)
        # only the connections to layers in skip_layers are ignored
        # so it is as if these layers are invisible to their neighors
        #
        # consider what we want when we do a positive phase update
        # we an inital state object with units [v, h] where v are the observed data points
        # we want to sample a new state with units [v, h']
        # where h -> h' but v stayes the same
        # thus, we want to sample h' ~ p(h | v) = Z^{-1} exp( b(h) + W v h )
        # it is important that the effective field on h includes the state of v
        # so that we are sampling from the conditional distribution
        # some more comments on this below
        #

        layer_ix = [[x for x in x if x not in skip_layers]
                       for x in self.layer_connections]
        weight_ix = [[x for x in x if x not in skip_layers]
                        for x in self.weight_connections]

        # update the odd then the even layers
        for ll in [range(1, self.num_layers, 2), range(0, self.num_layers, 2)]:
            for i in ll:

                #
                # here i is the index of a layer to be sampled from
                # the functions below modify the state of the units corresponding to that layer
                # we don't want to modify the units for the layers in skip_layers
                # here is one approach:
                #
                # if i in skip_layers:
                #   continue
                # else:
                #   update the extrinsic parameters of layer i
                #   draw a random sample layer from i
                #

                self.layers[i].update(
                    [self.layers[j].rescale(state.units[j]) for j in layer_ix[i]],
                    [self.weights[j].W() if j < i else self.weights[j].W_T()
                        for j in weight_ix[i]],
                    beta)

                #
                # THIS MODIFIES THE INPUT OBJECT DIRECTLY
                # EITHER THIS SHOULD BE AN EXPLICIT IN-PLACE UPDATE
                # CLEARLY ANNOTATED IN THE FUNCTION DOCS
                # OR NEED TO FIX SO FUNCTION DOESN'T HAVE SIDE EFFECTS
                #

                state.units[i] = getattr(self.layers[i], func_name)()

        return state


    def mcstep(self, state, beta=None, skip_layers=[]):
        """
        Perform a single Gibbs sampling update in alternating layers.
        state -> new state

        Args:
            state (State object): the current state of each layer
            beta (optional, tensor (batch_size, 1)): Inverse temperatures
            skip_layers (list): list of layer indices to skip updates

        Returns:
            new state

        """
        return self._alternating_update('sample_state', state, beta, skip_layers)

    def markov_chain(self, n, state, beta=None, skip_layers=[]):
        """
        Perform multiple Gibbs sampling steps in alternating layers.
        state -> new state

        Args:
            n (int): number of steps.
            state (State object): the current state of each layer
            beta (optional, tensor (batch_size, 1)): Inverse temperatures
            skip_layers (list): list of layer indices to skip updates

        Returns:
            new state

        """
        for _ in range(n):
            state = self.mcstep(state, beta, skip_layers)
        return state

    def mean_field_step(self, state, beta=None, skip_layers=[]):
        """
        Perform a single mean-field update in alternating layers.
        state -> new state

        Args:
            state (State object): the current state of each layer
            beta (optional, tensor (batch_size, 1)): Inverse temperatures
            skip_layers (list): list of layer indices to skip updates

        Returns:
            new state

        """
        return self._alternating_update('mean', state, beta, skip_layers)

    def mean_field_iteration(self, n, state, beta=None, skip_layer=[]):
        """
        Perform multiple mean-field updates in alternating layers
        states -> new state

        Args:
            n (int): number of steps.
            state (State object): the current state of each layer
            beta (optional, tensor (batch_size, 1)): Inverse temperatures
            skip_layers (list): list of layer indices to skip updates

        Returns:
            new state

        """
        for _ in range(n):
            state = self.mean_field_step(state, beta, skip_layers)
        return state

    def deterministic_step(self, state, beta=None, skip_layers=[]):
        """
        Perform a single deterministic (maximum probability) update
        in alternating layers.
        state -> new state

        Args:
            state (State object): the current state of each layer
            beta (optional, tensor (batch_size, 1)): Inverse temperatures
            skip_layers (list): list of layer indices to skip updates

        Returns:
            new state

        """
        return self._alternating_update('mode', state, beta, skip_layers)

    def deterministic_iteration(self, n, state, beta=None, skip_layers=[]):
        """
        Perform multiple deterministic (maximum probability) updates
        in alternating layers.
        state -> new state

        Args:
            n (int): number of steps.
            state (State object): the current state of each layer
            beta (optional, tensor (batch_size, 1)): Inverse temperatures
            skip_layers (list): list of layer indices to skip updates

        Returns:
            new state

        """
        for _ in range(n):
            state = self.deterministic_step(state, beta, skip_layers)
        return state

    #TODO: use State
    # currently, gradients are computed using the mean of the hidden units
    # conditioned on the value of the visible units
    # this will not work for deep models, because we cannot compute
    # the means for models with more than 1 hidden layer
    # therefore, the gradients need to be computed from samples
    # of all of the visible and hidden layer units (i.e., States)
    #
    # Args should be:
    # data (State): observed visible units and sampled hidden units
    # model (State): visible and hidden units sampled from the model
    def gradient(self, data_state, model_state):
        """
        Compute the gradient of the model parameters.

        For vis \in {vdata, vmodel}, we:

        1. Scale the visible data.
        vis_scaled = self.layers[i].rescale(vis)

        2. Update the hidden layer.
        self.layers[i+1].update(vis_scaled, self.weights[i].W())

        3. Compute the mean of the hidden layer.
        hid = self.layers[i].mean()

        4. Scale the mean of the hidden layer.
        hid_scaled = self.layers[i+1].rescale(hid)

        5. Compute the derivatives.
        vis_derivs = self.layers[i].derivatives(vis, hid_scaled,
                                                self.weights[i].W())
        hid_derivs = self.layers[i+1].derivatives(hid, vis_scaled,
                                      be.transpose(self.weights[i+1].W())
        weight_derivs = self.weights[i].derivatives(vis_scaled, hid_scaled)

        The gradient is obtained by subtracting the vmodel contribution
        from the vdata contribution.

        Args:
            data_state (State object): The observed visible units and sampled hidden units.
            model_state (State objects): The visible and hidden units sampled from the model.

        Returns:
            dict: Gradients of the model parameters.

        """
        i = 0

        grad = gu.Gradient(
            [None for l in self.layers],
            [None for w in self.weights]
        )

        # POSITIVE PHASE (using observed)

        # 1. Scale vdata
        vdata_scaled = self.layers[i].rescale(data_state.units[i])

        # 2. Update the hidden layer
        self.layers[i+1].update(
            [vdata_scaled],
            [self.weights[0].W()]
        )

        # 3. Compute the mean of the hidden layer
        data_state.units[i+1] = self.layers[i+1].mean()

        # 4. Scale the hidden mean
        hid_scaled = self.layers[i+1].rescale(data_state.units[i+1])

        # 5. Compute the gradients
        grad.layers[i] = self.layers[i].derivatives(data_state.units[i],
                                                    [hid_scaled],
                                                    [self.weights[0].W()]
        )

        grad.layers[i+1] = self.layers[i+1].derivatives(data_state.units[i+1],
                                                        [vdata_scaled],
                                                        [self.weights[0].W_T()]
        )

        grad.weights[i] = self.weights[i].derivatives(vdata_scaled,
                                                      hid_scaled)

        # NEGATIVE PHASE (using sampled)

        # 1. Scale vdata
        vmodel_scaled = self.layers[i].rescale(model_state.units[i])

        # 2. Update the hidden layer
        self.layers[i+1].update(
            [vmodel_scaled],
            [self.weights[0].W()]
        )

        # 3. Compute the mean of the hidden layer
        model_state.units[i+1] = self.layers[i+1].mean()

        # 4. Scale hidden mean
        hid_scaled = self.layers[i+1].rescale(model_state.units[i+1])

        # 5. Compute the gradients
        grad.layers[i] = be.mapzip(be.subtract,
                                   self.layers[i].derivatives(
                                       model_state.units[i],
                                       [hid_scaled],
                                       [self.weights[0].W()]
                                   ),
                                   grad.layers[i])

        grad.layers[i+1] = be.mapzip(be.subtract,
                                     self.layers[i+1].derivatives(
                                         model_state.units[i+1],
                                         [vmodel_scaled],
                                         [self.weights[0].W_T()]
                                     ),
                                     grad.layers[i+1])

        grad.weights[i] = be.mapzip(be.subtract,
                                    self.weights[i].derivatives(
                                        vmodel_scaled,
                                        hid_scaled),
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
