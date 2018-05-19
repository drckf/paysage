import os, pandas, operator
from cytoolz import partial
from typing import List

from .. import layers
from .. import backends as be
from . import initialize as init
from . import gradient_util as gu
from . import graph as mg
from . import state as ms

class BoltzmannMachine(object):
    """
    General model class.
    (i.e., Restricted Boltzmann Machines).

    Example usage:
    '''
    vis = BernoulliLayer(nvis)
    hid = BernoulliLayer(nhid)
    rbm = BoltzmannMachine([vis, hid])
    '''


    """
    def __init__(self, layer_list: List, conn_list: List = None):
        """
        Create a model.

        Args:
            layer_list (List[layer])
            conn_list (optional; List[Connection])

        Returns:
            BoltzmannMachine

        """
        # layers are stored in a list with the visible units as the 0'th element
        self.layers = layer_list
        self.num_layers = len(self.layers)
        self.clamped_sampling = []
        self.multipliers = [None for _ in range(self.num_layers)]

        # set the weights
        self.connections = conn_list if conn_list is not None else self._default_connections()
        self.count_connections()

        # set up the moments of the layers to ensure model envelope is defined
        for layer in self.layers:
            layer.update_moments(
                layer.conditional_mean([be.zeros((1,1))], [be.zeros((1, layer.len))]))

    def count_connections(self):
        """
        Set the num_connections attribute.

        Notes:
            Modifies the num_connections attribute in place!

        Args:
            None

        Returns:
            None

        """
        self.num_connections = len(self.connections)

    def _default_connections(self):
        """
        Sets the default connections assuming a linear model graph.

        Args:
            None

        Returns:
            List[Connection]

        """
        conns = []
        for i in range(self.num_layers - 1):
            w = layers.Weights((self.layers[i].len, self.layers[i+1].len))
            conns.append(mg.Connection(i, i+1, w))
        return conns

    def num_parameters(self):
        """
        Return the number of parameters in the model

        Args:
            None

        Returns:
            number of parameters
        """
        c = 0
        for l in self.layers:
            c += l.num_parameters()
        for conn in self.connections:
            c += conn.weights.shape[0]*conn.weights.shape[1]
        return c

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
            "type": "BoltzmannMachine",
            "layers": [layer.get_config() for layer in self.layers],
            "connections": [conn.get_config() for conn in self.connections]
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
        layer_list = [layers.layer_from_config(l) for l in config["layers"]]
        conn_list = None
        if "connections" in config:
            conn_list = [mg.Connection.from_config(c) for c in config["connections"]]
        return cls(layer_list, conn_list)

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
        config = self.get_config()
        store.put('model', pandas.DataFrame())
        store.get_storer('model').attrs.config = config
        for i in range(self.num_layers):
            key = os.path.join('layers', 'layers_'+str(i))
            self.layers[i].save_params(store, key)
        for i in range(self.num_connections):
            key = os.path.join('connections', 'weights_'+str(i))
            self.connections[i].weights.save_params(store, key)

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
        # load the layer parameters
        for i in range(len(model.layers)):
            key = os.path.join('layers', 'layers_'+str(i))
            model.layers[i].load_params(store, key)
        # load the weights
        for i in range(len(model.connections)):
            key = os.path.join('connections', 'weights_'+str(i))
            model.connections[i].weights.load_params(store, key)
        return model

    def copy(self):
        """
        Copy a Boltzmann machine.

        Args:
            None

        Returns:
            BoltzmannMachine

        """
        model = BoltzmannMachine.from_config(self.get_config())
        # set the weights
        for i in range(model.num_connections):
            model.connections[i].weights.set_params(self.connections[i].weights.params)
        # set the layer parameters
        for i in range(model.num_layers):
            model.layers[i].set_params(self.layers[i].get_params())
        return model

    def copy_params(self, model):
        """
        Copy the params from a source model into self.

        Notes:
            Modifies attributes in place!

        Args:
            model (BoltzmannMachine)

        Returns:
            None

        """
        # set the weights
        for i in range(self.num_connections):
            self.connections[i].weights.set_params(model.connections[i].weights.params)
        # set the layer parameters
        for i in range(self.num_layers):
            self.layers[i].set_params(model.layers[i].get_params())

    #
    # Methods that define topology
    #

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

    def _connected_rescaled_units(self, i: int, state: ms.State) -> List:
        """
        Helper function to retrieve the rescaled units connected to layer i.

        Args:
            i (int): the index of the layer of interest
            state (State): the state of the units

        Returns:
            List[tensor]: the rescaled values of the connected units

        """

        units = []
        for conn in self.connections:
            if i == conn.target_index:
                units += [be.maybe_a(self.multipliers[conn.domain_index],
                                     self.layers[conn.domain_index].rescale(
                        state[conn.domain_index]), operator.mul)]
            elif i == conn.domain_index :
                units += [self.layers[conn.target_index].rescale(
                        state[conn.target_index])]
        return units

    def _connected_weights(self, i: int) -> List:
        """
        Helper function to retrieve the values of the weights connecting
        layer i to its neighbors.

        Args:
            i (int): the index of the layer of interest

        Returns:
            list[tensor]: the weights connecting layer i to its neighbors

        """
        weights = []
        for conn in self.connections:
            if i == conn.target_index:
                weights += [conn.weights.W(trans=True)]
            elif i == conn.domain_index:
                weights += [conn.weights.W(trans=False)]
        return weights

    #
    # Methods for sampling and sample based training
    #

    def initialize(self, batch, method: str='hinton', **kwargs) -> None:
        """
        Initialize the parameters of the model.

        Args:
            batch: A Batch object.
            method (optional): The initialization method.

        Returns:
            None

        """
        try:
            func = getattr(init, method)
        except AttributeError:
            print(method + ' is not a valid initialization method for latent models')
        func(batch, self, **kwargs)

        for l in self.layers:
            l.enforce_constraints()

        for l in range(1, len(self.layers)):
            lay = self.layers[l]
            n = lay.len
            lay.update_moments(
                lay.conditional_mean([be.zeros((1,1))], [be.zeros((1, n))]))

        for conn in self.connections:
            conn.weights.enforce_constraints()

    def _alternating_update_(self, func_name: str, state: ms.State, beta=None) -> None:
        """
        Performs a single Gibbs sampling update in alternating layers.

        Notes:
            Changes state in place.

        Args:
            func_name (str, function name): layer function name to apply to the
                units to sample
            state (State object): the state of each layer
            beta (optional, tensor (batch_size, 1)): Inverse temperatures

        Returns:
            None

        """
        # define even and odd sampling sets to alternate between, including
        # only layers that can be sampled
        (odd_layers, even_layers) = (range(1, self.num_layers, 2),
                                     range(0, self.num_layers, 2))
        layer_order = [i for i in list(odd_layers) + list(even_layers)
                       if i in self.get_sampled()]

        for i in layer_order:
            func = getattr(self.layers[i], func_name)
            state[i] = func(
                self._connected_rescaled_units(i, state),
                self._connected_weights(i),
                beta=beta)

    def markov_chain(self, n: int, state: ms.State, beta=None,
                     callbacks=None) -> ms.State:
        """
        Perform multiple Gibbs sampling steps in alternating layers.
        state -> new state

        Notes:
            Samples layers according to the conditional probability
            on adjacent layers,
            x_i ~ P(x_i | x_(i-1), x_(i+1) )

        Args:
            n (int): number of steps.
            state (State object): the state of each layer
            beta (optional, tensor (batch_size, 1)): Inverse temperatures
            callbacks(optional, List[callable]): list of functions to call
                at each step; signature func(State)

        Returns:
            new state

        """
        new_state = ms.State.from_state(state)
        for _ in range(n):
            self._alternating_update_('conditional_sample', new_state, beta = beta)
            if callbacks is not None:
                for func in callbacks:
                    func(new_state)
        return new_state

    def mean_field_iteration(self, n: int, state: ms.State, beta=None,
                             callbacks=None) -> ms.State:
        """
        Perform multiple mean-field updates in alternating layers
        state -> new state

        Notes:
            Returns the expectation of layer units
            conditioned on adjacent layers,
            x_i = E[x_i | x_(i-1), x_(i+1) ]

        Args:
            n (int): number of steps.
            state (State object): the state of each layer
            beta (optional, tensor (batch_size, 1)): Inverse temperatures
            callbacks (optional, List[callable]): list of functions to call
                at each step; signature func(State)

        Returns:
            new state

        """
        new_state = ms.State.from_state(state)
        for _ in range(n):
            self._alternating_update_('conditional_mean', new_state, beta=beta)
            if callbacks is not None:
                for func in callbacks:
                    func(new_state)
        return new_state

    def deterministic_iteration(self, n: int, state: ms.State, beta=None,
                                callbacks=None) -> ms.State:
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
            state (State object): the state of each layer
            beta (optional, tensor (batch_size, 1)): Inverse temperatures
            callbacks (optional, List[callable]): list of functions to call
                at each step; signature func(State)

        Returns:
            new state

        """
        new_state = ms.State.from_state(state)
        for _ in range(n):
            self._alternating_update_('conditional_mode', new_state, beta=beta)
            if callbacks is not None:
                for func in callbacks:
                    func(new_state)
        return new_state

    def compute_reconstructions(self, visible, method='markov_chain'):
        """
        Compute the reconstructions of a visible tensor.

        Args:
            visible (tensor (num_samples, num_units))
            method (str): ['markov_chain', 'mean_field_iteration', 'deterministic_iteration']

        Returns:
            reconstructions (tensor (num_samples, num_units))

        """
        data_state = ms.State.from_visible(visible, self)
        return getattr(self, method)(1, data_state)

    def exclusive_gradient_(self, grad, state, func, penalize=True,
                            weighting_function=be.do_nothing):
        """
        Compute the gradient of the model parameters using only a single phase.
        Scales the units in the state and computes the gradient.
        Includes a weight factor for the gradients.

        Notes:
            Modifies grad in place.

        Args:
            grad (Gradient): a gradient object
            state (State object): the state of the units
            func (Callable): a function like func(tensor, tensor) -> tensor
            penalize (bool): control on applying layer penalties
            weighting_function (function): a weighting function to apply
                to units when computing the gradient.

        Returns:
            dict: Gradients of the model parameters.

        """
        # note that derivatives are normalized with respect to batch size
        # compute the gradients of the layer parameters
        for i in range(self.num_layers):
            deriv = self.layers[i].derivatives(
                state[i],
                self._connected_rescaled_units(i, state),
                self._connected_weights(i),
                penalize=penalize,
                weighting_function=weighting_function
                )
            grad.layers[i] = [be.mapzip(func, z[0], z[1])
            for z in zip(deriv, grad.layers[i])]

        # compute the gradients of the weights
        for i in range(self.num_connections):
            target = self.connections[i].target_index
            domain = self.connections[i].domain_index
            deriv = self.connections[i].weights.derivatives(
                self.layers[target].rescale(state[target]),
                self.layers[domain].rescale(state[domain]),
                penalize=penalize,
                weighting_function=weighting_function
                )
            grad.weights[i] = [be.mapzip(func, z[0], z[1])
            for z in zip(deriv, grad.weights[i])]

        return grad

    def gradient(self, data_state, model_state, data_weighting_function=be.do_nothing,
                 model_weighting_function=be.do_nothing):
        """
        Compute the gradient of the model parameters.
        Scales the units in the state and computes the gradient.

        Args:
            data_state (State object): The observed visible units and
                sampled hidden units.
            model_state (State object): The visible and hidden units
                sampled from the model.

        Returns:
            dict: Gradients of the model parameters.

        """
        grad = gu.zero_grad(self)
        self.exclusive_gradient_(grad, data_state, be.add, penalize=True,
                                 weighting_function=data_weighting_function)
        self.exclusive_gradient_(grad, model_state, be.subtract, penalize=False,
                                 weighting_function=model_weighting_function)
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
            self.layers[layer_index].parameter_step(deltas.layers[layer_index])
        for conn_index in range(self.num_connections):
            self.connections[conn_index].weights.parameter_step(deltas.weights[conn_index])

    def joint_energy(self, state):
        """
        Compute the joint energy of the model based on a state.

        Args:
            state (State object): the current state of each layer

        Returns:
            tensor (num_samples,): Joint energies.

        """
        energy = 0
        for layer_index in range(self.num_layers):
            energy += self.layers[layer_index].energy(state[layer_index])
        for conn_index in range(self.num_connections):
            target = self.connections[conn_index].target_index
            domain = self.connections[conn_index].domain_index
            energy += self.connections[conn_index].weights.energy(
                    self.layers[target].rescale(state[target]),
                    self.layers[domain].rescale(state[domain]))
        return energy

    #
    # Methods for training with the TAP approximation
    #

    def _connected_elements(self, i: int, lst : List) -> List:
        """
        Helper function to retrieve a list of elements from a list
        whose elements correspond to the layers of the model

        Args:
            i (int): the index of the layer of interest
            lst (list *): a list of any object types

        Returns:
            list[*]: the sub-list of connected elements

        """
        connections = [lst[conn.target_index] for conn in self.connections
                       if i == conn.domain_index]
        connections += [lst[conn.domain_index] for conn in self.connections
                        if i == conn.target_index]
        return connections

    def _connecting_transforms(self, i: int, lst : List) -> List:
        """
        Helper function to retrieve a list of transforms from a list
        whose elements correspond to transforms connecting layers of the model.
        The input list must be indexed analogously to the weights of the model.

        Args:
            i (int): the index of the layer of interest
            lst (list [tensor]): a list of tensors mapping one layer to another

        Returns:
            list[*]: the sub-list of connected transforms

        """
        connections = [lst[j] for j in range(len(self.connections))
                       if i == self.connections[j].domain_index]
        connections += [be.transpose(lst[j]) for j in range(len(self.connections))
                        if i == self.connections[j].target_index]
        return connections

    def _get_rescaled_weights(self) -> List:
        """
        Helper function to retrieve a list of weights and a list of squared weights
        which have been rescaled according to their neighboring layers' rescaling
        coefficients.  I.e.,

        W_ij |-> W_ij/(s_i*s_j)

        Args:
            None

        Returns:
            (list[weight tensor], list[weight tensor])

        """
        rescaled_w = []
        for conn in self.connections:
            target_scale = self.layers[conn.target_index].reciprocal_scale()
            domain_scale = self.layers[conn.domain_index].reciprocal_scale()
            rescaled_w.append(be.multiply(be.multiply(
                    be.unsqueeze(target_scale, axis=1), conn.weights.W()),
                    be.unsqueeze(domain_scale, axis=0)))
        rescaled_w2 = [be.square(w) for w in rescaled_w]
        return (rescaled_w, rescaled_w2)

    def gibbs_free_energy(self, cumulants, rescaled_weight_cache=None):
        """
        Gibbs Free Energy (GFE) according to TAP2 appoximation

        Args:
            cumulants list(CumulantsTAP): cumulants of the layers
            rescaled_weight_cache tuple(list[tensor], list[tensor]):
             cached list of rescaled weight matrices and squares thereof

        Returns:
            float: Gibbs free energy
        """

        # cache rescaled weights for efficient computation of GFE
        if rescaled_weight_cache is None:
            rescaled_weight_cache = self._get_rescaled_weights()

        total = 0
        for index in range(self.num_layers):
            lay = self.layers[index]
            total += lay.TAP_entropy(cumulants[index])

        for index in range(self.num_connections):
            w = rescaled_weight_cache[0][index]
            w2 = rescaled_weight_cache[1][index]
            total -= be.quadratic(cumulants[index].mean, cumulants[index+1].mean, w)
            total -= 0.5 * be.quadratic(cumulants[index].variance,
                           cumulants[index+1].variance, w2)

        return total

    def compute_StateTAP(self, use_GD=True, init_lr=0.1, tol=1e-7, max_iters=50, ratchet=True,
                         decrease_on_neg=0.9, mean_weight=0.9, mean_square_weight=0.999,
                         seed=None, rescaled_weight_cache=None):
        """
        Compute the state of the layers by minimizing the second order TAP
        approximation to the Helmholtz free energy.  This function selects one of two
        possible implementations of this minimization procedure, gradient-descent or
        self-consistent iteration.

        Args:
            use_GD (bool): use gradient descent or use self_consistent iteration
            init_lr (float): initial learning rate for GD
            tol (float): tolerance for quitting minimization
            max_iters (int): maximum gradient decsent steps
            ratchet (bool): don't perform gradient update if not lowering GFE
            decrease_on_neg (float): factor to multiply lr by if the gradient step
                fails to lower the GFE
            mean_weight (float): mean weight parameter for ADAM
            mean_square_weight (float): mean square weight parameter for ADAM
                setting to 0.0 turns off adaptive weighting
            seed (CumulantsTAP): seed for the minimization
            rescaled_weight_cache tuple(list[tensor],list[tensor]): cache of
                rescaled weight and weight_square matrices

        Returns:
            tuple (StateTAP, float): TAP state of the layers and the GFE
        """
        if use_GD:
            return self._compute_StateTAP_GD(init_lr, tol, max_iters, ratchet,
                                             decrease_on_neg,
                                             mean_weight, mean_square_weight,
                                             rescaled_weight_cache=rescaled_weight_cache,
                                             seed=seed)
        else:
            return self._compute_StateTAP_self_consistent(tol=tol, max_iters=max_iters,
                                                          rescaled_weight_cache=rescaled_weight_cache,
                                                          seed=seed)

    def _compute_StateTAP_GD(self, init_lr=0.1, tol=1e-7, max_iters=50, ratchet=True,
                             decrease_on_neg=0.9, mean_weight=0.9, mean_square_weight=0.999,
                             seed=None, rescaled_weight_cache=None):
        """
        Compute the state of the layers by minimizing the second order TAP
        approximation to the Helmholtz free energy via gradient descent.

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

        This implementation uses ADAM gradient descent from a random starting location
         to minimize the function.

        Args:
            use_GD (bool): use gradient descent or use self_consistent iteration
            init_lr (float): initial learning rate for GD
            tol (float): tolerance for quitting minimization
            max_iters (int): maximum gradient decsent steps
            ratchet (bool): don't perform gradient update if not lowering GFE
            decrease_on_neg (float): factor to multiply lr by if the gradient step
                fails to lower the GFE
            mean_weight (float): mean weight parameter for ADAM
            mean_square_weight (float): mean square weight parameter for ADAM
                setting to 0.0 turns off adaptive weighting
            seed (CumulantsTAP): seed for the minimization
            rescaled_weight_cache tuple(list[tensor],list[tensor]): cache of
                rescaled weight and weight_square matrices

        Returns:
            tuple (StateTAP, float): TAP state of the layers and the GFE

        """

        # generate random sample in domain to use as a starting location for
        # gradient descent
        state = seed
        if seed is None:
            state = ms.StateTAP.from_model_rand(self)
        cumulants = state.cumulants

        # cache rescaled weights and squares for efficient computation of GFE
        if rescaled_weight_cache is None:
            rescaled_weight_cache = self._get_rescaled_weights()

        free_energy = self.gibbs_free_energy(cumulants, rescaled_weight_cache)

        lr = be.float_scalar(init_lr)
        lr_ = partial(be.tmul_, lr)
        beta_1 = be.float_scalar(mean_weight)
        beta_2 = be.float_scalar(mean_square_weight)
        bt_1_ = partial(be.mix_, beta_1)
        bt_2_ = partial(be.mix_, beta_2)
        comp_bt1 = partial(be.tmul, be.float_scalar(1.0/(1.0 - beta_1)))
        comp_bt2 = partial(be.tmul, be.float_scalar(1.0/(1.0 - beta_2)))

        mom = [lay.get_zero_magnetization() for lay in self.layers]
        var = [lay.get_zero_magnetization() for lay in self.layers]
        var_corr = [lay.get_zero_magnetization() for lay in self.layers]
        grad = [lay.get_zero_magnetization() for lay in self.layers]
        eps = [be.apply(be.ones_like, mag) for mag in grad]
        for mag in eps:
            be.apply_(partial(be.tmul_, be.float_scalar(1e-6)), mag)
        coeff = [lay.get_zero_magnetization() for lay in self.layers]
        depth = range(self.num_layers)

        for _ in range(max_iters):
            # compute the gradient of the Gibbs Free Energy
            new_grad = self._TAP_magnetization_grad(cumulants, rescaled_weight_cache)

            # compute momentum, and unbiased momentum.  Cache the latter in grad
            for i in depth:
                be.mapzip_(bt_1_, mom[i], new_grad[i])
                grad[i] = be.apply(comp_bt1, mom[i])

            # If we are using adaptive rescaling:
            if mean_square_weight > 1e-6:
                for i in depth:
                    be.mapzip_(bt_2_, var[i], be.apply(be.square, new_grad[i]))
                    var_corr[i] = be.apply(comp_bt2, var[i])
                for i in depth:
                    coeff[i] = be.apply(be.reciprocal,
                               be.mapzip(be.add, be.apply(be.sqrt, var_corr[i]), eps[i]))
                for c in coeff:
                    be.apply_(lr_,c)
                for i in depth:
                    grad[i] = be.mapzip(be.multiply, coeff[i], grad[i])
            else:
                for g in grad:
                    be.apply_(lr_,g)

            new_cumulants = [
                self.layers[l].clip_magnetization(
                    be.mapzip(be.subtract, grad[l], cumulants[l])
                )
                for l in range(self.num_layers)]
            new_free_energy = self.gibbs_free_energy(new_cumulants,
                                                     rescaled_weight_cache)

            neg = free_energy - new_free_energy < 0
            if abs(free_energy - new_free_energy) < tol:
                break

            if neg:
                # the step was too large, reduce the learning rate
                lr *= decrease_on_neg
                lr_ = partial(be.tmul_, be.float_scalar(lr))
                if lr < 1e-10:
                    break
                if ratchet == False:
                    cumulants = new_cumulants
                    free_energy = new_free_energy
            else:
                cumulants = new_cumulants
                free_energy = new_free_energy

        return ms.StateTAP(cumulants, self.lagrange_multipliers_analytic(cumulants)), \
               free_energy

    def _compute_StateTAP_self_consistent(self, tol=1e-7, max_iters=50,
                                          seed=None, rescaled_weight_cache=None):
        """
        Compute the state of the layers by minimizing the second order TAP
        approximation to the Helmholtz free energy by iterating the stationarity
        conditions on the magnetizations.

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
            tol (float): tolerance for quitting minimization.
            max_iters (int): maximum gradient decsent steps.
            seed (CumulantsTAP): seed for minimization
            rescaled_weight_cache tuple(list[tensor],list[tensor]): cache of
                rescaled weight and weight_square matrices

        Returns:
            tuple (StateTAP, float): TAP state of the layers and the GFE

        """

        # generate random sample in domain to use as a starting location
        # for gradient descent
        state = seed
        if seed is None:
            state = ms.StateTAP.from_model_rand(self)

        # cache rescaled weights for efficient computation of GFE
        if rescaled_weight_cache is None:
            rescaled_weight_cache = self._get_rescaled_weights()

        free_energy = self.gibbs_free_energy(state.cumulants, rescaled_weight_cache)

        for itr in range(max_iters):
            # Perform a self-consistent update to each layer
            for i in range(self.num_layers-1, -1, -1):
                self.layers[i].update_lagrange_multipliers_(
                    state.cumulants[i],
                    state.lagrange_multipliers[i],
                    self._connected_elements(i, state.cumulants),
                    self._connecting_transforms(i, rescaled_weight_cache[0]),
                    self._connecting_transforms(i, rescaled_weight_cache[1]))
                self.layers[i].self_consistent_update_(
                    state.cumulants[i],
                    state.lagrange_multipliers[i])

            # compute the new free energy and perform an update
            new_free_energy = self.gibbs_free_energy(state.cumulants, rescaled_weight_cache)

            if abs(free_energy - new_free_energy) < tol:
                break
            free_energy = new_free_energy

        return state, free_energy

    def lagrange_multipliers_analytic(self, cumulants):
        """
        Compute lagrange multipliers of each layer according to an analytic calculation
            of lagrange multipliers at beta=0.

        Args:
            cumulants (list[CumulantsTAP]): list of magnetizations
                of each layer

        Returns:
            lagrange_multipliers (list [CumulantsTAP])

        """
        return [self.layers[i].lagrange_multipliers_analytic(cumulants[i])
                for i in range(self.num_layers)]

    def _TAP_magnetization_grad(self, cumulants, rescaled_weight_cache=None):
        """
        Gradient of the Gibbs free energy with respect to the
            magnetization parameters

        Args:
            cumulants (list [CumulantsTAP]): magnetizations at which to compute the deriviates
            rescaled_weight_cache (list [tensor]): list of cached squares of weight
                matrices linking layers

        Returns:
            list (list [CumulantsTAP]): list of gradient magnetizations of each layer

        """
        if rescaled_weight_cache is None:
            rescaled_weight_cache = self._get_rescaled_weights()

        grad = [None for lay in self.layers]
        for i in range(self.num_layers):
            grad[i] = self.layers[i].TAP_magnetization_grad(
                cumulants[i],
                self._connected_elements(i, cumulants),
                self._connecting_transforms(i, rescaled_weight_cache[0]),
                self._connecting_transforms(i, rescaled_weight_cache[1]))
        return grad

    def _grad_gibbs_free_energy(self, state, rescaled_weight_cache=None):
        """
        Gradient of the Gibbs free energy with respect to the model parameters

        Args:
            state (StateTAP): magnetizations at which to compute the derivatives
            rescaled_weight_cache (list [tensor]): list of cached squares of weight
             matrices linking layers

        Returns:
            namedtuple (Gradient)
        """
        if rescaled_weight_cache is None:
            rescaled_weight_cache = self._get_rescaled_weights()

        #TODO move the higher-order TAP scale derivatives out of the layers
        grad_GFE = gu.Gradient(
            [self.layers[i].GFE_derivatives(state.cumulants[i],
                self._connected_elements(i, state.cumulants),
                self._connecting_transforms(i, rescaled_weight_cache[0]),
                self._connecting_transforms(i, rescaled_weight_cache[1]))
             for i in range(self.num_layers)]
            ,
            [conn.weights.GFE_derivatives(
                self.layers[conn.target_index].rescale_cumulants(
                    state.cumulants[conn.target_index]),
                self.layers[conn.domain_index].rescale_cumulants(
                    state.cumulants[conn.domain_index]),
                )
            for conn in self.connections]
            )

        return grad_GFE

    def grad_TAP_free_energy(self, use_GD=True, init_lr=0.1, tol=1e-7, max_iters=50, ratchet=True,
                             decrease_on_neg=0.9, mean_weight=0.9, mean_square_weight=0.999):
        """
        Compute the gradient of the Helmholtz free engergy of the model according
        to the TAP expansion around infinite temperature.

        This function will use the class members which specify the parameters for
        the Gibbs FE minimization.
        The gradients are taken as the average over the gradients computed at
        each of the minimial magnetizations for the Gibbs FE.

        Args:
            use_GD (bool): use gradient descent or use self_consistent iteration
            init_lr (float): initial learning rate for GD
            tol (float): tolerance for quitting minimization
            max_iters (int): maximum gradient decsent steps
            ratchet (bool): don't perform gradient update if not lowering GFE
            decrease_on_neg (float): factor to multiply lr by if the gradient step
                fails to lower the GFE
            mean_weight (float): mean weight parameter for ADAM
            mean_square_weight (float): mean square weight parameter for ADAM
                setting to 0.0 turns off adaptive weighting

        Returns:
            namedtuple: (Gradient): containing gradients of the model parameters.

        """
        rescaled_weight_cache = self._get_rescaled_weights()
        state,_ = self.compute_StateTAP(use_GD, init_lr, tol, max_iters, ratchet,
                                        decrease_on_neg, mean_weight, mean_square_weight,
                                        rescaled_weight_cache = rescaled_weight_cache)
        return self._grad_gibbs_free_energy(state,
                                            rescaled_weight_cache = rescaled_weight_cache)

    def TAP_gradient(self, data_state, use_GD=True, init_lr=0.1, tol=1e-7, max_iters=50,
                     ratchet=True, decrease_on_neg=0.9, mean_weight=0.9,
                     mean_square_weight=0.999):
        """
        Gradient of -\ln P(v) with respect to the model parameters

        Args:
            data_state (State object): The observed visible units and sampled
                hidden units.
            use_GD (bool): use gradient descent or use self_consistent iteration
            init_lr (float): initial learning rate for GD
            tol (float): tolerance for quitting minimization
            max_iters (int): maximum gradient decsent steps
            ratchet (bool): don't perform gradient update if not lowering GFE
            decrease_on_neg (float): factor to multiply lr by if the gradient step
                fails to lower the GFE
            mean_weight (float): mean weight parameter for ADAM
            mean_square_weight (float): mean square weight parameter for ADAM
                setting to 0.0 turns off adaptive weighting

        Returns:
            gradient (Gradient): containing gradients of the model parameters.

        """
        # compute average grad_F_marginal over the minibatch
        pos_phase = gu.null_grad(self)
        # compute the postive phase of the gradients of the layer parameters
        for i in range(self.num_layers):
            pos_phase.layers[i] = self.layers[i].derivatives(
                data_state[i],
                self._connected_rescaled_units(i, data_state),
                self._connected_weights(i),
                penalize=True)

        for i in range(self.num_connections):
            target = self.connections[i].target_index
            domain = self.connections[i].domain_index
            pos_phase.weights[i] = self.connections[i].weights.derivatives(
                self.layers[target].rescale(data_state[target]),
                self.layers[domain].rescale(data_state[domain]),
                penalize=True)

        # compute the gradient of the Helmholtz FE via TAP_gradient
        neg_phase = self.grad_TAP_free_energy(use_GD, init_lr, tol, max_iters,
                                              ratchet, decrease_on_neg,
                                              mean_weight, mean_square_weight)

        grad = gu.grad_mapzip(be.subtract, neg_phase, pos_phase)
        return grad
