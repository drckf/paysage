import os
import copy
import pandas
from cytoolz import partial
from copy import deepcopy
from typing import List

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
        self.len = len (self.shapes)

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

class StateTAP(object):
    """A TAPState is a list of CumulantsTAP objects for each layer in the model."""
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
        return copy.deepcopy(state)

    @classmethod
    def from_model(cls, model):
        """
        Create a StateTAP object from an existing StateTAP.

        Args:
            state (StateTAP): a StateTAP instance

        Returns:
            StateTAP object

        """
        return cls([layer.get_zero_magnetization() for layer in model.layers])

    @classmethod
    def from_model_rand(cls, model):
        """
        Create a StateTAP object from an existing StateTAP.

        Args:
            state (StateTAP): a StateTAP instance

        Returns:
            StateTAP object

        """
        return cls([layer.get_random_magnetization() for layer in model.layers])    



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
        self.layer_connections = self._layer_connections()
        self.weight_connections = self._weight_connections()

        # adjacent layers are connected by weights
        # therefore, if there are len(layers) = n then len(weights) = n - 1
        self.weights = [
            layers.Weights((self.layers[i].len, self.layers[i+1].len))
        for i in range(self.num_layers - 1)
        ]

    #
    # Methods for saving and loading models. 
    #

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
        for i in range(self.num_layers - 1):
            key = os.path.join('weights', 'weights'+str(i))
            self.weights[i].save_params(store, key)
        for i in range(len(self.layers)):
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

    #
    # Methods that define topology
    #

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

    def _connected_rescaled_units(self, i, state):
        """
        Helper function to retrieve the rescaled units connected to layer i.

        Args:
            i (int): the index of the layer of interest
            state (State): the current state of the units

        Returns:
            list[tensor]: the rescaled values of the connected units

        """
        return [self.layers[j].rescale(state.units[j])
                            for j in self.layer_connections[i]]

    def _connected_weights(self, i):
        """
        Helper function to retrieve the values of the weights connecting
        layer i to its neighbors.

        Args:
            i (int): the index of the layer of interest

        Returns:
            list[tensor]: the weights connecting layer i to its neighbros

        """
        return [self.weights[j].W() if j < i else self.weights[j].W_T()
                            for j in self.weight_connections[i]]

    #
    # Methods for sampling and sample based training 
    # 

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
        for layer_set in [range(1, self.num_layers, 2),
                          range(0, self.num_layers, 2)]:
            for i in layer_set:
                if i not in clamped:
                    func = getattr(self.layers[i], func_name)
                    updated_state.units[i] = func(
                        self._connected_rescaled_units(i, updated_state),
                        self._connected_weights(i),
                        beta)

        return updated_state

    def markov_chain(self, n, state, beta=None, clamped: List[int]=[]) -> State:
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
            clamped (list): list of layer indices to clamp

        Returns:
            new state

        """
        new_state = State.from_state(state)
        for _ in range(n):
            new_state = self._alternating_update('conditional_sample',
                                                 new_state,
                                                 beta,
                                                 clamped)
        return new_state

    def mean_field_iteration(self, n, state, beta=None, clamped=[]):
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
            clamped (list): list of layer indices to clamp

        Returns:
            new state

        """
        new_state = State.from_state(state)
        for _ in range(n):
            new_state = self._alternating_update('conditional_mean',
                                                 new_state,
                                                 beta,
                                                 clamped)
        return new_state

    def deterministic_iteration(self, n, state, beta=None, clamped=[]):
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
            clamped (list): list of layer indices to clamp

        Returns:
            new state

        """
        new_state = State.from_state(state)
        for _ in range(n):
            new_state = self._alternating_update('conditional_mode',
                                                 new_state,
                                                 beta,
                                                 clamped)
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
        grad = gu.null_grad(self)

        # POSITIVE PHASE (using observed)

        # compute the postive phase of the gradients of the layer parameters
        for i in range(self.num_layers):
            grad.layers[i] = self.layers[i].derivatives(
                data_state.units[i],
                [self.layers[j].rescale(data_state.units[j])
                    for j in self.layer_connections[i]],
                [self.weights[j].W() if j < i else self.weights[j].W_T()
                    for j in self.weight_connections[i]],
            )

        # compute the positive phase of the gradients of the weights
        for i in range(self.num_layers - 1):
            grad.weights[i] = self.weights[i].derivatives(
                self.layers[i].rescale(data_state.units[i]),
                self.layers[i+1].rescale(data_state.units[i+1]),
            )

        # NEGATIVE PHASE (using sampled)

        # update the gradients of the layer parameters with the negative phase
        for i in range(self.num_layers):
            grad.layers[i] = be.mapzip(be.subtract,
                self.layers[i].derivatives(
                    model_state.units[i],
                    [self.layers[j].rescale(model_state.units[j])
                        for j in self.layer_connections[i]],
                    [self.weights[j].W() if j < i else self.weights[j].W_T()
                        for j in self.weight_connections[i]],
                ),
            grad.layers[i])

        # update the gradients of the weight parameters with the negative phase
        for i in range(self.num_layers - 1):
            grad.weights[i] = be.mapzip(be.subtract,
                self.weights[i].derivatives(
                    self.layers[i].rescale(model_state.units[i]),
                    self.layers[i+1].rescale(model_state.units[i+1]),
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

    #
    # Methods for training with the TAP approximation
    #

    def gibbs_free_energy(self, state):
        """
        Gibbs Free Energy (GFE) according to TAP2 appoximation

        Args:
            state (StateTAP): cumulants of the layers

        Returns:
            float: Gibbs free energy
        """
        total = 0

        lagrange = [self.layers[l].lagrange_multiplers(state.cumulants[l]) 
                    for l in range(self.num_layers)] 

        for index in range(self.num_layers):
            lay = self.layers[index]
            total += lay.TAP_entropy(lagrange[index], state.cumulants[index])

        for index in range(self.num_layers-1):
            w = self.weights[index].W_T()
            total -= be.quadratic(state.cumulants[index].mean, w, state.cumulants[index+1].mean)
            total -= 0.25 * be.quadratic(state.cumulants[index].variance, be.square(w), state.cumulants[index+1].variance)

        return total

    def compute_StateTAP(self, init_lr=0.1, tol=1e-7, max_iters=50):
        """
        Compute the state of the layers by minimizing the second order TAP 
        approximation to the Helmholtz free energy.

        If the energy is,
        '''
        E(v, h) := -\langle a,v \rangle - \langle b,h \rangle - \langle v,W \cdot h \rangle,
        '''
        with Boltzmann probability distribution,
        '''
        P(v,h) := Z^{-1} \exp{-E(v,h)},
        '''
        and the marginal,
        '''
        P(v) := \sum_{h} P(v,h),
        '''
        then the Helmholtz free energy is,
        '''
        F(v) := - log Z = -log \sum_{v,h} \exp{-E(v,h)}.
        '''
        We add an auxiliary local field q, and introduce the inverse temperature
         variable \beta to define
        '''
        \beta F(v;q) := -log\sum_{v,h} \exp{-\beta E(v,h) + \beta \langle q, v \rangle}
        '''
        Let \Gamma(m) be the Legendre transform of F(v;q) as a function of q,
         the Gibbs free energy.
        The TAP formula is Taylor series of \Gamma in \beta, around \beta=0.
        Setting \beta=1 and regarding the first two terms of the series as an
         approximation of \Gamma[m],
        we can minimize \Gamma in m to obtain an approximation of F(v;q=0) = F(v)

        This implementation uses gradient descent from a random starting location
         to minimize the function

        Args:
            init_lr float: initial learning rate
            tol float: tolerance for quitting minimization.
            max_iters: maximum gradient decsent steps.

        Returns:
            state of the layers (StateTAP)

        """
        decrease = be.float_scalar(0.5)

        # generate random sample in domain to use as a starting location for gradient descent
        state = StateTAP.from_model_rand(self)

        free_energy = self.gibbs_free_energy(state)
        lr = init_lr
        lr_ = partial(be.tmul_, be.float_scalar(lr))

        for _ in range(max_iters):
            # compute the gradient of the Gibbs Free Energy
            grad = self._TAP_magnetization_grad(state)
            for g in grad:
                be.apply_(lr_, g)

            # take a gradient step to compute a new state
            new_state = StateTAP([
            self.layers[l].clip_magnetization(be.mapzip(be.subtract, grad[l], state.cumulants[l])) 
            for l in range(self.num_layers)])
            # compute the new free energy and perform an update
            new_free_energy = self.gibbs_free_energy(new_state)
            if (free_energy - new_free_energy < 0):
                # the step was too large, halve the learning rate
                lr *= decrease
                lr_ = partial(be.tmul_, be.float_scalar(lr))
                if (lr < 1e-10):
                    break
            elif (free_energy - new_free_energy < tol):
                break
            else:
                state = new_state
                free_energy = new_free_energy

        return state

    def _TAP_magnetization_grad(self, state):
        """
        Gradient of the Gibbs free energy with respect to the magnetization parameters

        Args:
            state (StateTAP): magnetizations at which to compute the deriviates

        Returns:
            list (list of gradient magnetization objects for each layer)
            
        """
        grad = [None for lay in self.layers]
        for i in range(self.num_layers):
            grad[i] = self.layers[i].TAP_magnetization_grad(
                    state.cumulants[i],
                    [state.cumulants[j] for j in self.layer_connections[i]],
                    [self.weights[j].W() if j < i else self.weights[j].W_T() 
                    for j in self.weight_connections[i]]
                    )
        return grad

    def _grad_gibbs_free_energy(self, state):
        """
        Gradient of the Gibbs free energy with respect to the model parameters

        Args:
            state (StateTAP):
              magnetizations at which to compute the derivatives

        Returns:
            namedtuple (Gradient)
        """
        grad_GFE = gu.Gradient(
            [self.layers[l].GFE_derivatives(state.cumulants[l]) for l in range(self.num_layers)],
            [self.weights[w].GFE_derivatives(state.cumulants[w], state.cumulants[w+1]) 
            for w in range(self.num_layers-1)]
            )

        return grad_GFE

    def grad_TAP_free_energy(self, init_lr_EMF, tolerance_EMF, max_iters_EMF):
        """
        Compute the gradient of the Helmholtz free engergy of the model according 
        to the TAP expansion around infinite temperature.

        This function will use the class members which specify the parameters for 
        the Gibbs FE minimization.
        The gradients are taken as the average over the gradients computed at 
        each of the minimial magnetizations for the Gibbs FE.

        Args:
            init_lr float: initial learning rate which is halved whenever necessary 
            to enforce descent.
            tol float: tolerance for quitting minimization.
            max_iters int: maximum gradient decsent steps

        Returns:
            namedtuple: (Gradient): containing gradients of the model parameters.

        """
        state = self.compute_StateTAP(init_lr_EMF, tolerance_EMF, max_iters_EMF)
        return self._grad_gibbs_free_energy(state)

    def TAP_gradient(self, data_state, init_lr, tolerance, max_iters):
        """
        Gradient of -\ln P(v) with respect to the model parameters

        Args:
            data_state (State object): The observed visible units and sampled
             hidden units.
            init_lr float: initial learning rate which is halved whenever necessary
             to enforce descent.
            tol float: tolerance for quitting minimization.
            max_iters int: maximum gradient decsent steps

        Returns:
            Gradient (namedtuple): containing gradients of the model parameters.

        """
        # compute average grad_F_marginal over the minibatch
        pos_phase = gu.null_grad(self)

        # compute the postive phase of the gradients of the layer parameters
        for i in range(self.num_layers):
            pos_phase.layers[i] = self.layers[i].derivatives(
                data_state.units[i],
                [self.layers[j].rescale(data_state.units[j])
                    for j in self.layer_connections[i]],
                [self.weights[j].W() if j < i else self.weights[j].W_T()
                    for j in self.weight_connections[i]],
            )

        # compute the positive phase of the gradients of the weights
        for i in range(self.num_layers - 1):
            pos_phase.weights[i] = self.weights[i].derivatives(
                self.layers[i].rescale(data_state.units[i]),
                self.layers[i+1].rescale(data_state.units[i+1]),
            )

        # compute the gradient of the Helmholtz FE via TAP_gradient
        neg_phase = self.grad_TAP_free_energy(init_lr, tolerance, max_iters)

        return gu.grad_mapzip(be.subtract, pos_phase, neg_phase)
