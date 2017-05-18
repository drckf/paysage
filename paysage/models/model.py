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


    def TAP_gradient(self, data_state, num_r, num_p, persistent_samples,
                     init_lr_EMF, tolerance_EMF, max_iters_EMF):
        """
        Gradient of -\ln P(v) with respect to the model parameters

        Args:
            data_state (State object): The observed visible units and sampled hidden units.
            num_r: (int>=0) number of random seeds to use for Gibbs FE minimization
            num_p: (int>=0) number of persistent seeds to use for Gibbs FE minimization
            persistent_samples list of magnetizations: persistent magnetization parameters
                to keep as seeds for Gibbs free energy estimation.
            init_lr float: initial learning rate which is halved whenever necessary to enforce descent.
            tol float: tolerance for quitting minimization.
            max_iters int: maximum gradient decsent steps

        Returns:
            namedtuple: Gradient: containing gradients of the model parameters.

        """
        # compute average grad_F_marginal over the minibatch
        grad_MFE = gu.null_grad(self)

        # compute the postive phase of the gradients of the layer parameters
        for i in range(self.num_layers):
            grad_MFE.layers[i] = self.layers[i].derivatives(
                data_state.units[i],
                [self.layers[j].rescale(data_state.units[j])
                    for j in self.layer_connections[i]],
                [self.weights[j].W() if j < i else self.weights[j].W_T()
                    for j in self.weight_connections[i]],
            )

        # compute the positive phase of the gradients of the weights
        for i in range(self.num_layers - 1):
            grad_MFE.weights[i] = self.weights[i].derivatives(
                self.layers[i].rescale(data_state.units[i]),
                self.layers[i+1].rescale(data_state.units[i+1]),
            )

        # compute the gradient of the Helmholtz FE via TAP_gradient
        grad_HFE = self.grad_TAP_free_energy(num_r, num_p, persistent_samples,
                     init_lr_EMF, tolerance_EMF, max_iters_EMF)

        return gu.grad_mapzip(be.subtract, grad_MFE, grad_HFE)

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

    def gibbs_free_energy(self, mag):
        """
        Gibbs FE according to TAP2 appoximation

        Args:
            mag (list of magnetizations of layers):
              magnetizations at which to compute the free energy

        Returns:
            float: Gibbs free energy
        """
        total = 0
        B = [self.layers[l]._gibbs_lagrange_multipliers_expectation(mag[l]) for l in range(self.num_layers)]
        A = [self.layers[l]._gibbs_lagrange_multipliers_variance(mag[l]) for l in range(self.num_layers)]

        for l in range(self.num_layers):
            lay = self.layers[l]
            total += lay._gibbs_free_energy_entropy_term(B[l], A[l], mag[l])

        for w in range(self.num_layers-1):
            way = self.weights[w]
            total -= be.dot(mag[w].expectation(), be.dot(way.params.matrix, mag[w+1].expectation()))
            total -= 0.5 * be.dot(mag[w].variance(), \
                     be.dot(be.square(way.params.matrix), mag[w+1].variance()))

        return total

    def TAP_free_energy(self, seed=None, init_lr=0.1, tol=1e-7, max_iters=50, method='gd'):
        """
        Compute the Helmholtz free energy of the model according to the TAP
        expansion around infinite temperature to second order.

        If the energy is,
        '''
            E(v, h) := -\langle a,v \rangle - \langle b,h \rangle - \langle v,W \cdot h \rangle,
        '''
        with Boltzmann probability distribution,
        '''
            P(v,h)  := 1/\sum_{v,h} \exp{-E(v,h)} * \exp{-E(v,h)},
        '''
        and the marginal,
        '''
            P(v)    := \sum_{h} P(v,h),
        '''
        then the Helmholtz free energy is,
        '''
            F(v) := -log\sum_{v,h} \exp{-E(v,h)}.
        '''
        We add an auxiliary local field q, and introduce the inverse temperature variable \beta to define
        '''
            \beta F(v;q) := -log\sum_{v,h} \exp{-\beta E(v,h) + \beta \langle q, v \rangle}
        '''
        Let \Gamma(m) be the Legendre transform of F(v;q) as a function of q, the Gibbs free energy.
        The TAP formula is Taylor series of \Gamma in \beta, around \beta=0.
        Setting \beta=1 and regarding the first two terms of the series as an approximation of \Gamma[m],
        we can minimize \Gamma in m to obtain an approximation of F(v;q=0) = F(v)

        This implementation uses gradient descent from a random starting location to minimize the function

        Args:
            seed 'None' or Magnetization: initial seed for the minimization routine.
                                          Chosing 'None' will result in a random seed
            init_lr float: initial learning rate which is halved whenever necessary to enforce descent.
            tol float: tolerance for quitting minimization.
            max_iters: maximum gradient decsent steps.
            method: one of 'gd' or 'constraint' picking which Gibbs FE minimization method to use.

        Returns:
            tuple (magnetization, TAP-approximated Helmholtz free energy)
                  (Magnetization, float)

        """
        # TODO: re-implement support for constraint satisfaction method
        if method not in ['gd', 'constraint']:
            raise ValueError("Must specify a valid method for minimizing the Gibbs free energy")

        def minimize_gibbs_free_energy_GD(m, init_lr=0.01, tol=1e-6, max_iters=1):
            """
            Simple gradient descent routine to minimize Gibbs free energy

            Note: The fact that this method is a closure suggests that it might be moved to a
                   utility class later

            Args:
                m (list of magnetizations of layers): seed for gradient descent
                init_lr float: initial learning rate which is halved whenever necessary
                               to enforce descent.
                tol float: tolerance for quitting minimization.
                max_iters int: maximum gradient decsent steps

            Returns:
                tuple (list of magnetizations, minimal GFE value)

            """
            mag = deepcopy(m)
            eps = 1e-6
            its = 0

            gam = self.gibbs_free_energy(mag)
            lr = init_lr
            clip_ = partial(be.clip_inplace, a_min=eps, a_max=1.0-eps)
            lr_ = partial(be.tmul_, be.float_scalar(lr))
            #print(gam)
            while (its < max_iters):
                its += 1
                grad = self._grad_magnetization_GFE(mag)
                for g in grad:
                    be.apply_(lr_, g)
                m_provisional = [be.mapzip(be.subtract, grad[l], mag[l]) for l in range(self.num_layers)]

                # Warning: in general a lot of clipping gets done here
                for m_l in m_provisional:
                    be.apply_(clip_, m_l)

                gam_provisional = self.gibbs_free_energy(m_provisional)
                if (gam - gam_provisional < 0):
                    lr *= 0.5
                    lr_ = partial(be.tmul_, be.float_scalar(lr))
                    #print("decreased lr" + str(its))
                    if (lr < 1e-10):
                        #print("tol reached on iter" + str(its))
                        break
                elif (gam - gam_provisional < tol):
                    break
                else:
                    #print(gam - gam_provisional)
                    mag = m_provisional
                    gam = gam_provisional

            return (mag, gam)

        # generate random sample in domain to use as a starting location for gradient descent
        if seed==None :
            seed = [lay.get_random_magnetization() for lay in self.layers]
            clip_ = partial(be.clip_inplace, a_min=0.005, a_max=0.995)
            for m in seed:
                be.apply_(clip_, m)

        if method == 'gd':
            return minimize_gibbs_free_energy_GD(seed, init_lr, tol, max_iters)
        elif method == 'constraint':
            assert False, \
                   "Constraint satisfaction is not currently supported"
            return minimize_gibbs_free_energy_GD(seed, init_lr, tol, max_iters)

    def _grad_magnetization_GFE(self, mag):
        """
        Gradient of the Gibbs free energy with respect to the magnetization parameters

        Args:
            mag (list of magnetizations of layers):
              magnetizations at which to compute the deriviates

        Returns:
            list (list of gradient magnetization objects for each layer)
        """
        grad = [None for lay in self.layers]
        for l in range(self.num_layers):
            grad[l] = self.layers[l]._grad_magnetization_GFE(mag[l])

        for k in range(self.num_layers - 1):
            way = self.weights[k]
            w = way.params.matrix
            ww = be.square(w)
            grad[k+1].grad_GFE_update_down(mag[k], mag[k+1], w, ww)
            grad[k].grad_GFE_update_up(mag[k], mag[k+1], w, ww)

        return grad

    def _grad_gibbs_free_energy(self, mag):
        """
        Gradient of the Gibbs free energy with respect to the model parameters

        Args:
            mag (list of magnetizations of layers):
              magnetizations at which to compute the deriviates

        Returns:
            namedtuple (Gradient)
        """
        grad_GFE = gu.Gradient(
            [self.layers[l].GFE_derivatives(mag[l]) for l in range(self.num_layers)],
            [self.weights[w].GFE_derivatives(mag[w], mag[w+1])
                for w in range(self.num_layers-1)]
            )
        return grad_GFE

    def grad_TAP_free_energy(self, num_r, num_p, persistent_samples,
                             init_lr_EMF, tolerance_EMF, max_iters_EMF):
        """
        Compute the gradient of the Helmholtz free engergy of the model according
        to the TAP expansion around infinite temperature.

        This function will use the class members which specify the parameters for
        the Gibbs FE minimization.
        The gradients are taken as the average over the gradients computed at
        each of the minimial magnetizations for the Gibbs FE.

        Args:
            num_r: (int>=0) number of random seeds to use for Gibbs FE minimization
            num_p: (int>=0) number of persistent seeds to use for Gibbs FE minimization
            persistent_samples list of magnetizations: persistent magnetization parameters
                to keep as seeds for Gibbs free energy estimation.
            init_lr float: initial learning rate which is halved whenever necessary
            to enforce descent.
            tol float: tolerance for quitting minimization.
            max_iters int: maximum gradient decsent steps

        Returns:
            namedtuple: (Gradient): containing gradients of the model parameters.

        """

        # compute the TAP approximation to the Helmholtz free energy:
        grad_EMF = gu.zero_grad(self)

        # compute minimizing magnetizations from random initializations
        for s in range(num_r):
            mag, EMF = self.TAP_free_energy(None,
                                            init_lr_EMF,
                                            tolerance_EMF,
                                            max_iters_EMF)
            # Compute the gradients at this minimizing magnetization
            grad_gfe = self._grad_gibbs_free_energy(mag)
            gu.grad_mapzip_(be.add_, grad_gfe, grad_EMF)

        # compute minimizing magnetizations from seeded initializations
        for s in range(num_p): # persistent seeds
            self.persistent_samples[s], EMF = \
             self.TAP_free_energy(persistent_samples[s],
                                  init_lr_EMF,
                                  tolerance_EMF,
                                  max_iters_EMF)
            # Compute the gradients at this minimizing magnetization
            grad_gfe = self._grad_gibbs_free_energy(persistent_samples[s])
            gu.grad_mapzip_(be.add_, grad_gfe, grad_EMF)

        # average
        scale = partial(be.tmul_, be.float_scalar(1/(num_p + num_r)))
        gu.grad_apply_(scale, grad_EMF)

        return grad_EMF

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
