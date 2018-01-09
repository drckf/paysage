import os
import pandas
from cytoolz import partial
from typing import List

from .. import layers
from .. import backends as be
from . import init_model as init
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
    def __init__(self, layer_list: List, weight_list: List = None):
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
        self.multipliers = be.ones((self.num_layers))

        # set the weights
        if weight_list is not None:
            self.weights = weight_list
        else:
            self.weights = [layers.Weights(
                    (self.layers[weight_index[0]].len,
                     self.layers[weight_index[1]].len
                     )
                    ) for weight_index in self.graph.weight_connections]
        self.num_weights = len(self.weights)

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
            "layers": [layer.get_config() for layer in self.layers],
            "weights": [weight.get_config() for weight in self.weights]
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
        for layer in config["layers"]:
            layer_list.append(layers.layer_from_config(layer))
        tmp = cls(layer_list)
        for i in range(len(config["weights"])):
            tmp.weights[i] = layers.weights_from_config(config["weights"][i])
        return tmp

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

    #
    # Methods that define topology
    #

    def use_dropout(self):
        """
        Indicate if the model has dropout.

        Args:
            None

        Returns:
            true of false

        """
        return any(layer.use_dropout() for layer in self.layers)

    def _connected_rescaled_units(self, i: int, state: mu.State,
                                  dropout_mask: mu.State = None) -> List:
        """
        Helper function to retrieve the rescaled units connected to layer i.

        Args:
            i (int): the index of the layer of interest
            state (State): the current state of the units
            dropout_mask (State): mask on
                model units for dropout, 1: on 0: dropped-out

        Returns:
            list[tensor]: the rescaled values of the connected units

        """
        connections = self.graph.layer_connections[i]
        if dropout_mask is not None:
            return [self.multipliers[conn.layer] * self.layers[conn.layer].rescale(
                    be.multiply(dropout_mask.units[conn.layer], state.units[conn.layer]))
                    for conn in connections]
        else:
            return [self.multipliers[conn.layer] * self.layers[conn.layer].rescale(
                    state.units[conn.layer]) for conn in connections]

    def _connected_weights(self, i:int) -> List:
        """
        Helper function to retrieve the values of the weights connecting
        layer i to its neighbors.

        Args:
            i (int): the index of the layer of interest

        Returns:
            list[tensor]: the weights connecting layer i to its neighbors

        """
        connections = self.graph.layer_connections[i]
        return [self.weights[conn.weight].W_T() if conn.is_forward \
                else self.weights[conn.weight].W() \
                for conn in connections]

    def _connected_cumulants(self, i: int, state: mu.State) -> List:
        """
        Helper function to retrieve the cumulants,
        CumulantsTAP attributes of StateTAP objects.

        Args:
            i (int): the index of the layer of interest
            state (StateTAP): the current TAP state of the units

        Returns:
            list[tensor]: the cumulants of connected layers to the layer.

        """
        connections = self.graph.layer_connections[i]
        return [state.cumulants[conn.layer] for conn in connections]

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

    def _alternating_update_(self, func_name: str, state: mu.State,
                            dropout_mask: mu.State = None, beta=None) -> None:
        """
        Performs a single Gibbs sampling update in alternating layers.

        Notes:
            Changes state in place.

        Args:
            func_name (str, function name): layer function name to apply to the units to sample
            state (State object): the current state of each layer
            dropout_mask (State object): mask on model units
                for dropout, 1: on 0: dropped-out
            beta (optional, tensor (batch_size, 1)): Inverse temperatures

        Returns:
            new state

        """
        # define even and odd sampling sets to alternate between, including
        # only layers that can be sampled
        (odd_layers, even_layers) = (range(1, self.num_layers, 2),
                                     range(0, self.num_layers, 2))
        layer_order = [i for i in list(odd_layers) + list(even_layers)
                       if i in self.graph.get_sampled()]

        # update the odd then the even layers
        for i in layer_order:
            func = getattr(self.layers[i], func_name)
            state.units[i] = func(
                self._connected_rescaled_units(i, state, dropout_mask),
                self._connected_weights(i),
                beta=beta)

    def markov_chain(self, n: int, state: mu.State, dropout_mask: mu.State = None,
                     beta=None) -> mu.State:
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
            dropout_mask (State object):
                mask on model units for dropout, 1: on 0: dropped-out
            beta (optional, tensor (batch_size, 1)): Inverse temperatures

        Returns:
            new state

        """
        new_state = mu.State.from_state(state)
        for _ in range(n):
            self._alternating_update_('conditional_sample', new_state,
                                      dropout_mask = dropout_mask, beta = beta)
        return new_state

    def mean_field_iteration(self, n: int, state: mu.State, dropout_mask: mu.State = None,
                             beta=None) -> mu.State:
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
            dropout_mask (State object):
                mask on model units for dropout, 1: on 0: dropped-out
            beta (optional, tensor (batch_size, 1)): Inverse temperatures

        Returns:
            new state

        """
        new_state = mu.State.from_state(state)
        for _ in range(n):
            self._alternating_update_('conditional_mean', new_state,
                                      dropout_mask=dropout_mask, beta=beta)
        return new_state

    def deterministic_iteration(self, n: int, state: mu.State, dropout_mask: mu.State = None,
                                beta=None) -> mu.State:
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
            dropout_mask (State object):
                mask on model units for dropout, 1: on 0: dropped-out
            beta (optional, tensor (batch_size, 1)): Inverse temperatures

        Returns:
            new state

        """
        new_state = mu.State.from_state(state)
        for _ in range(n):
            self._alternating_update_('conditional_mode', new_state,
                                      dropout_mask=dropout_mask, beta=beta)
        return new_state

    def gradient(self, data_state, model_state, positive_dropout=None, negative_dropout=None):
        """
        Compute the gradient of the model parameters.
        Scales the units in the state and computes the gradient.

        Args:
            data_state (State object): The observed visible units and sampled hidden units.
            model_state (State object): The visible and hidden units sampled from the model.
            positive_dropout (State object): mask on model units
                for positive phase dropout, 1: on 0: dropped-out
            negative_dropout (State object): mask on model units
                for negative phase dropout, 1: on 0: dropped-out

        Returns:
            dict: Gradients of the model parameters.

        """
        grad = gu.null_grad(self)

        # POSITIVE PHASE (using observed)
        data_state_dropout = mu.dropout_state(data_state, positive_dropout)

        # compute the postive phase of the gradients of the layer parameters
        for i in range(self.num_layers):
            grad.layers[i] = self.layers[i].derivatives(
                data_state_dropout.units[i],
                self._connected_rescaled_units(i, data_state),
                self._connected_weights(i),
                penalize=True
                )

        # compute the positive phase of the gradients of the weights
        for i in range(self.num_weights):
            iL = self.graph.weight_connections[i][0]
            iR = self.graph.weight_connections[i][1]
            grad.weights[i] = self.weights[i].derivatives(
                self.layers[iL].rescale(data_state.units[iL]),
                self.layers[iR].rescale(data_state.units[iR]),
                penalize=True
                )

        # NEGATIVE PHASE (using sampled)
        model_state_dropout = mu.dropout_state(model_state, negative_dropout)

        # update the gradients of the layer parameters with the negative phase
        for i in range(self.num_layers):
            deriv = self.layers[i].derivatives(
                    model_state_dropout.units[i],
                    self._connected_rescaled_units(i, model_state),
                    self._connected_weights(i),
                    penalize=False
                    )
            grad.layers[i] = [be.mapzip(be.subtract, z[0], z[1])
            for z in zip(deriv, grad.layers[i])]

        # update the gradients of the weight parameters with the negative phase
        for i in range(self.num_weights):
            iL = self.graph.weight_connections[i][0]
            iR = self.graph.weight_connections[i][1]
            deriv = self.weights[i].derivatives(
                    self.layers[iL].rescale(model_state.units[iL]),
                    self.layers[iR].rescale(model_state.units[iR]),
                    penalize=False
                    )
            grad.weights[i] = [be.mapzip(be.subtract, z[0], z[1])
            for z in zip(deriv, grad.weights[i])]

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
        rescaled_units = mu.dropout_state(data, mu.State.dropout_rescale(self))
        energy = 0
        for layer_index in range(self.num_layers):
            energy += self.layers[layer_index].energy(rescaled_units.units[layer_index])
        for weight_index in range(self.num_weights):
            iL = self.graph.weight_connections[weight_index][0]
            iR = self.graph.weight_connections[weight_index][1]
            energy += self.weights[weight_index].energy(rescaled_units.units[iL],
                                  rescaled_units.units[iR])
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

        for index in range(self.num_weights):
            w = self.weights[index].W()
            total -= be.quadratic(state.cumulants[index].mean, state.cumulants[index+1].mean, w)
            total -= 0.5 * be.quadratic(state.cumulants[index].variance,
                           state.cumulants[index+1].variance, be.square(w))

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
        state = mu.StateTAP.from_model_rand(self)

        free_energy = self.gibbs_free_energy(state)
        lr = init_lr
        lr_ = partial(be.tmul_, be.float_scalar(lr))

        for _ in range(max_iters):
            # compute the gradient of the Gibbs Free Energy
            grad = self._TAP_magnetization_grad(state)
            for g in grad:
                be.apply_(lr_, g)

            # take a gradient step to compute a new state
            new_state = mu.StateTAP([
                self.layers[l].clip_magnetization(
                    be.mapzip(be.subtract, grad[l], state.cumulants[l])
                )
                for l in range(self.num_layers)])
            # compute the new free energy and perform an update
            new_free_energy = self.gibbs_free_energy(new_state)
            if free_energy - new_free_energy < 0:
                # the step was too large, halve the learning rate
                lr *= decrease
                lr_ = partial(be.tmul_, be.float_scalar(lr))
                if lr < 1e-10:
                    break
            elif free_energy - new_free_energy < tol:
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
                self._connected_cumulants(i, state),
                self._connected_weights(i)
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
            [self.layers[i].GFE_derivatives(state.cumulants[i]) for i in range(self.num_layers)],
            [self.weights[i].GFE_derivatives(
                state.cumulants[self.graph.weight_connections[i][0]],
                state.cumulants[self.graph.weight_connections[i][1]],
                )
            for i in range(self.num_weights)]
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

    def TAP_gradient(self, data_state, init_lr, tolerance, max_iters, positive_dropout=None):
        """
        Gradient of -\ln P(v) with respect to the model parameters

        Args:
            data_state (State object): The observed visible units and sampled
             hidden units.
            init_lr float: initial learning rate which is halved whenever necessary
             to enforce descent.
            tol float: tolerance for quitting minimization.
            max_iters (int): maximum gradient decsent steps
            positive_dropout (State object): mask on model units for positive phase dropout
             1: on 0: dropped-out

        Returns:
            Gradient (namedtuple): containing gradients of the model parameters.

        """
        # compute average grad_F_marginal over the minibatch
        pos_phase = gu.null_grad(self)

        if positive_dropout is not None:
            data_state.units = be.mapzip(be.multiply, positive_dropout.units, data_state.units)

        # compute the postive phase of the gradients of the layer parameters
        for i in range(self.num_layers):
            pos_phase.layers[i] = self.layers[i].derivatives(
                data_state.units[i],
                self._connected_rescaled_units(i, data_state),
                self._connected_weights(i)
            )

        # compute the positive phase of the gradients of the weights
        for i in range(self.num_weights):
            iL = self.graph.weight_connections[i][0]
            iR = self.graph.weight_connections[i][1]
            pos_phase.weights[i] = self.weights[i].derivatives(
                self.layers[iL].rescale(data_state.units[iL]),
                self.layers[iR].rescale(data_state.units[iR]),
            )

        # compute the gradient of the Helmholtz FE via TAP_gradient
        neg_phase = self.grad_TAP_free_energy(init_lr, tolerance, max_iters)

        return gu.grad_mapzip(be.subtract, pos_phase, neg_phase)
