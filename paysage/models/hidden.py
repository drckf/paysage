import os
import pandas
from collections import namedtuple
from cytoolz import compose
from math import sqrt

from .. import layers
from .. import backends as be
from ..models.initialize import init_hidden as init

# ----- CLASSES ----- #

Gradient = namedtuple("Gradient", [
    "layers",
    "weights"
])



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
        units = [layers[i].random(shapes[i]) for i in range(model.num_layers)]
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

        assert self.num_layers == 2,\
        "Only models with 2 layers are currently supported"

        # adjacent layers are connected by weights
        # therefore, if there are len(layers) = n then len(weights) = n - 1
        self.weights = [
        layers.Weights((self.layers[i].len, self.layers[i+1].len))
        for i in range(self.num_layers - 1)
        ]

    def get_config(self):
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
        "layer_types": ["visible", "hidden"],
        }
        return config

    @classmethod
    def from_config(cls, config):
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

    def initialize(self, data, method='hinton'):
        """
        Inialize the parameters of the model.

        Args:
            data: A batch object.
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

    #TODO: use State
    # currently, this method takes in a single tensor (vis)
    # and outputs a single tensor (new vis)
    # the hidden units are only treated implicitly
    #
    # this method should take in a State (with the units of all of the layers)
    # and output a new State (with the updated units of all of the layers)
    #
    # this could be done in with the following steps:
    # 1) update the extrinsic parameters of the odd layers
    # 2) sample new configurations for the odd layers
    # 3) update the extrinsic parameters of the even layers
    # 4) sample new configurations for the even layers
    #
    # either, this could return a new State object (which involves a copy)
    # or, it could mutate the values of the State tensors in place
    def mcstep(self, state, beta=None, update_vis=True):
        """
        Perform a single Gibbs sampling update.
        v -> update h distribution ~ h -> update v distribution ~ v'

        Args:
            state (State object): the current state of each layer
            beta (optional, (batch_size, 1)): Inverse temperatures
            update_vis (bool): update state of layer 0 if True

        Returns:
            new state

        """
        # update the odd layers
        for i in range(1, self.num_layers, 2):
            pass


        # update the even layers
        for i in range(0, self.num_layers, 2):
            pass

        i = 0

        self.layers[i+1].update(
        [self.layers[i].rescale(vis)],
        [self.weights[i].W()],
        beta)

        hid = self.layers[i+1].sample_state()

        self.layers[i].update(
        [self.layers[i+1].rescale(hid)],
        [self.weights[i].W_T()],
        beta)

        return self.layers[i].sample_state()

    # TODO: use State
    # this function is just a repeated application of mcstep
    # so it should be changed to operate on State objects too
    def markov_chain(self, vis, n, beta=None):
        """
        Perform multiple Gibbs sampling steps.
        v ~ h ~ v_1 ~ h_1 ~ ... ~ v_n

        Args:
            vis (batch_size, num_visible): Observed visible units.
            n: Number of steps.
            beta (optional, (batch_size, 1)): Inverse temperatures.

        Returns:
            tensor: New visible units (v').

        """
        new_vis = be.float_tensor(vis)
        for t in range(n):
            new_vis = self.mcstep(new_vis, beta)
        return new_vis

    #TODO: use State
    # currently, this method takes in a single tensor (vis)
    # and outputs a single tensor (new vis)
    # the hidden units are only treated implicitly
    #
    # this method should take in a State (with the units of all of the layers)
    # and output a new State (with the updated units of all of the layers)
    #
    # this could be done in with the following steps:
    # 1) update the extrinsic parameters of the odd layers
    # 2) compute the mean of the odd layers
    # 3) update the extrinsic parameters of the even layers
    # 4) compute the mean of the even layers
    #
    # either, this could return a new State object (which involves a copy)
    # or, it could mutate the values of the State tensors in place
    def mean_field_step(self, vis, beta=None):
        """
        Perform a single mean-field update.
        v -> update h distribution -> h -> update v distribution -> v'

        Args:
            vis (batch_size, num_visible): Observed visible units.
            beta (optional, (batch_size, 1)): Inverse temperatures.

        Returns:
            tensor: New visible units (v').

        """
        i = 0

        self.layers[i+1].update(
        [self.layers[i].rescale(vis)],
        [self.weights[i].W()],
        beta)

        hid = self.layers[i+1].mean()

        self.layers[i].update(
        [self.layers[i+1].rescale(hid)],
        [self.weights[i].W_T()],
        beta)

        return self.layers[i].mean()

    # TODO: use State
    # this function is just a repeated application of mean_field_step
    # so it should be changed to operate on State objects too
    def mean_field_iteration(self, vis, n, beta=None):
        """
        Perform multiple mean-field updates.
        v -> h -> v_1 -> h_1 -> ... -> v_n

        Args:
            vis (batch_size, num_visible): Observed visible units.
            n: Number of steps.
            beta (optional, (batch_size, 1)): Inverse temperatures.

        Returns:
            tensor: New visible units (v').

        """
        new_vis = be.float_tensor(vis)
        for t in range(n):
            new_vis = self.mean_field_step(new_vis, beta)
        return new_vis

    #TODO: use State
    # currently, this method takes in a single tensor (vis)
    # and outputs a single tensor (new vis)
    # the hidden units are only treated implicitly
    #
    # this method should take in a State (with the units of all of the layers)
    # and output a new State (with the updated units of all of the layers)
    #
    # this could be done in with the following steps:
    # 1) update the extrinsic parameters of the odd layers
    # 2) compute the mode of the odd layers
    # 3) update the extrinsic parameters of the even layers
    # 4) compute the mode of the even layers
    #
    # either, this could return a new State object (which involves a copy)
    # or, it could mutate the values of the State tensors in place
    def deterministic_step(self, vis, beta=None):
        """
        Perform a single deterministic (maximum probability) update.
        v -> update h distribution -> h -> update v distribution -> v'

        Args:
            vis (batch_size, num_visible): Observed visible units.
            beta (optional, (batch_size, 1)): Inverse temperatures.

        Returns:
            tensor: New visible units (v').

        """
        i = 0

        self.layers[i+1].update(
        [self.layers[i].rescale(vis)],
        [self.weights[i].W()],
        beta)

        hid = self.layers[i+1].mode()

        self.layers[i].update(
        [self.layers[i+1].rescale(hid)],
        [self.weights[i].W_T()],
        beta)

        return self.layers[i].mode()

    # TODO: use State
    # this function is just a repeated application of deterministic_step
    # so it should be changed to operate on State objects too
    def deterministic_iteration(self, vis, n, beta=None):
        """
        Perform multiple deterministic (maximum probability) updates.
        v -> h -> v_1 -> h_1 -> ... -> v_n

        Args:
            vis (batch_size, num_visible): Observed visible units.
            n: Number of steps.
            beta (optional, (batch_size, 1)): Inverse temperatures.

        Returns:
            tensor: New visible units (v').

        """
        new_vis = be.float_tensor(vis)
        for _ in range(n):
            new_vis = self.deterministic_step(new_vis, beta)
        return new_vis

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
    def gradient(self, vdata, vmodel):
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
            vdata: The observed visible units.
            vmodel: The sampled visible units.

        Returns:
            dict: Gradients of the model parameters.

        """
        i = 0

        grad = Gradient(
        [None for l in self.layers],
        [None for w in self.weights]
        )

        # POSITIVE PHASE (using observed)

        # 1. Scale vdata
        vdata_scaled = self.layers[i].rescale(vdata)

        # 2. Update the hidden layer
        self.layers[i+1].update(
        [vdata_scaled],
        [self.weights[0].W()]
        )

        # 3. Compute the mean of the hidden layer
        hid = self.layers[i+1].mean()

        # 4. Scale the hidden mean
        hid_scaled = self.layers[i+1].rescale(hid)

        # 5. Compute the gradients
        grad.layers[i] = self.layers[i].derivatives(vdata,
                                           [hid_scaled],
                                           [self.weights[0].W()]
                                           )

        grad.layers[i+1] = self.layers[i+1].derivatives(hid,
                                               [vdata_scaled],
                                               [self.weights[0].W_T()]
                                               )

        grad.weights[i] = self.weights[i].derivatives(vdata_scaled,
                                                         hid_scaled)

        # NEGATIVE PHASE (using sampled)

        # 1. Scale vdata
        vmodel_scaled = self.layers[i].rescale(vmodel)

        # 2. Update the hidden layer
        self.layers[i+1].update(
        [vmodel_scaled],
        [self.weights[0].W()]
        )

        # 3. Compute the mean of the hidden layer
        hid = self.layers[i+1].mean()

        # 4. Scale hidden mean
        hid_scaled = self.layers[i+1].rescale(hid)

        # 5. Compute the gradients
        grad.layers[i] = be.mapzip(be.subtract,
                                  self.layers[i].derivatives(
                                                 vmodel,
                                                 [hid_scaled],
                                                 [self.weights[0].W()]
                                                 ),
                                  grad.layers[i])

        grad.layers[i+1] = be.mapzip(be.subtract,
                                  self.layers[i+1].derivatives(
                                                   hid,
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

    # TODO: use State
    # Args should be:
    # data (state): values of all the units
    # this should be the easiest function to update
    # also, it isn't really used anywhere right now
    def joint_energy(self, vis, hid):
        """
        Compute the joint energy of the model.

        Args:
            vis (batch_size, num_visible): Observed visible units.
            hid (batch_size, num_hidden): Sampled hidden units:

        Returns:
            tensor (batch_size, ): Joint energies.

        """
        energy = 0
        for i in range(len(self.weights)):
            energy += self.layers[i].energy(vis)
            energy += self.layers[i+1].energy(vis)
            energy += self.weights[i].energy(vis, hid)
        return energy

    # TODO: not sure what to do about this function for deep models
    # i think it should be implemented only for models with 1 hidden layer
    # could still take in a State object
    # but should assert self.num_layers == 2
    def marginal_free_energy(self, vis):
        """
        Compute the marginal free energy of the model.

        If the energy is:
        E(v, h) = -\sum_i a_i(v_i) - \sum_j b_j(h_j) - \sum_{ij} W_{ij} v_i h_j
        Then the marginal free energy is:
        F(v) =  -\sum_i a_i(v_i) - \sum_j \log \int dh_j \exp(b_j(h_j) - \sum_i W_{ij} v_i)

        Args:
            vis (batch_size, num_visible): Observed visible units.

        Returns:
            tensor (batch_size, ): Marginal free energies.

        """
        i = 0
        phi = be.dot(vis, self.weights[i].W())
        log_Z_hidden = self.layers[i+1].log_partition_function(phi)
        energy = 0
        energy += self.layers[i].energy(vis)
        energy -= be.tsum(log_Z_hidden, axis=1)
        return energy

    def save(self, store):
        """
        Save a model to an open HDFStore.

        Note:
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
            df_weights = pandas.DataFrame(
                            be.to_numpy_array(self.weights[i].W())
                         )
            store.put('weights/weights_'+str(i), df_weights)
        for i in range(len(self.layers)):
            layer_type = config["layer_types"][i]
            layer = config["layers"][i]
            layer_key = os.path.join('layers', layer_type)
            # intrinsic params
            intrinsics = layer["intrinsic"]
            for ip in intrinsics:
                df_params = pandas.DataFrame(
                            be.to_numpy_array(self.layers[i].int_params[ip])
                         )
                store.put(os.path.join(layer_key,'intrinsic', ip), df_params)
            # extrinsic params
            extrinsics = layer["extrinsic"]
            for ep in extrinsics:
                df_params = pandas.DataFrame(
                            be.to_numpy_array(self.layers[i].ext_params[ep])
                         )
                store.put(os.path.join(layer_key,'extrinsic', ep), df_params)

# ----- FUNCTIONS ----- #

def grad_fold(func, grad):
    """
    Apply a function entrywise over a Gradient objet,
    combining the result.

    Args:
        func (callable): function with two arguments
        grad (Gradient)

    returns:
        float

    """
    result = 0
    for ly in grad.layers:
        result = be.fold(func, ly)
    for w in grad.weights:
        result = be.fold(func, w)
    return result

def grad_accumulate(func, grad):
    """
    Apply a funciton entrywise over a Gradient object,
    accumulating the result.

    Args:
        func (callable): function with one argument
        grad (Gradient)

    returns:
        float

    """
    result = 0
    for ly in grad.layers:
        result = be.accumulate(func, ly)
    for w in grad.weights:
        result = be.accumulate(func, w)
    return result

def grad_apply(func, grad):
    """
    Apply a function entrywise over a Gradient object.

    Args:
        func (callable)
        grad (Gradient)

    Returns:
        Gradient

    """
    return Gradient(
    [be.apply(func, ly) for ly in grad.layers],
    [be.apply(func, w) for w in grad.weights]
    )

def grad_apply_(func_, grad):
    """
    Apply a function entrywise over a Gradient object.

    Notes:
        Modifies elements of grad in place.

    Args:
        func_ (callable, in place operation)
        grad (Gradient)

    Returns:
        None

    """
    for ly in grad.layers:
        be.apply_(func_, ly)
    for w in grad.weights:
        be.apply_(func_, w)

def grad_mapzip(func, grad1, grad2):
    """
    Apply a function entrywise over the zip of two Gradient objects.

    Args:
        func_ (callable, in place operation)
        grad (Gradient)

    Returns:
        Gradient

    """
    n = len(grad1.layers)
    m = len(grad1.weights)
    return Gradient(
    [be.mapzip(func, grad1.layers[i], grad2.layers[i]) for i in range(n)],
    [be.mapzip(func, grad1.weights[i], grad2.weights[i]) for i in range(m)]
    )

def grad_mapzip_(func_, grad1, grad2):
    """
    Apply an in place function entrywise over the zip of two Gradient objects.

    Notes:
        Modifies elements of grad1 in place.

    Args:
        func_ (callable, in place operation)
        grad1 (Gradient)
        grad2 (Gradient)

    Returns:
        None

    """
    n = len(grad1.layers)
    m = len(grad1.weights)
    for i in range(n):
        be.mapzip_(func_, grad1.layers[i], grad2.layers[i])
    for j in range(m):
        be.mapzip_(func_, grad1.weights[j], grad2.weights[j])

def grad_magnitude(grad):
    """
    Compute the root-mean-square of the gradient.

    Args:
        grad (Gradient)

    Returns:
        magnitude (float)

    """
    n = len(grad.layers) + len(grad.weights)
    tensor_mean_square = compose(be.mean, be.square)
    return sqrt(grad_accumulate(tensor_mean_square, grad) / n)
