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
        self.len = len(self.units)

    def batch_size(self):
        """
        Get the batch size of the state.

        Args:
            None

        Returns:
            batch size: int

        """
        return be.shape(self.units[0])[0]

    def number_of_units(self, layer):
        """
        Get the number of units in a layer of the state.

        Args:
            layer (int)

        Returns:
            number_of_units (int)
        """
        return be.shape(self.units[layer])[1]

    def number_of_layers(self):
        """
        Get the number of layers in the state.

        Args:
            None

        Returns:
            number_of_layers (int)
        """
        return self.len

    def __getitem__(self, i):
        """
        Get the i'th element of the units attribute.

        Call like:
            state[i]

        Args:
            i (int): index

        Returns:
            tensor

        """
        return self.units[i]

    def __setitem__(self, i, value):
        """
        Set the i'th element of the units attribute to value

        Call like:
            state[i] = value

        Args:
            i (int): index
            value (tensor)

        Returns:
            None

        """
        self.units[i] = value

    def __iter__(self):
        """
        Iterates over the elements of the units attribute.

        E.g.
        for tensor in state:
            do something

        """
        for i in range(self.len):
            yield self.units[i]

    def __len__(self):
        """
        Get the number of layers in the state using len(state).

        Args:
            None

        Returns:
            int

        """
        return self.len

    @classmethod
    def from_model(cls, batch_size, model):
        """
        Create a State object.

        Args:
            batch_size (int): the number of samples per layer
            model (BoltzmannMachine): a model object

        Returns:
            state object

        """
        shapes = [(batch_size, l.len) for l in model.layers]
        units = [model.layers[i].random(shapes[i]) for i in range(model.num_layers)]
        return cls(units)

    @classmethod
    def from_model_envelope(cls, batch_size, model):
        """
        Create a State object.

        Args:
            batch_size (int): the number of samples per layer
            model (BoltzmannMachine): a model object

        Returns:
            state object

        """
        shapes = [(batch_size, l.len) for l in model.layers]
        units = [model.layers[i].envelope_random(shapes[i]) for i in range(model.num_layers)]
        return cls(units)

    @classmethod
    def from_visible(cls, vis, model):
        """
        Create a state object with given visible unit values.

        Args:
            vis (tensor (num_samples, num_visible)]: visible unit values.
            model (BoltzmannMachine): a model object

        Returns:
            state object

        """

        # randomly initialize the state
        batch_size = be.shape(vis)[0]
        state = cls.from_model(batch_size, model)
        state[0] = vis
        return state

    @classmethod
    def from_state(cls, state, sample_indices=None):
        """
        Create a State object by copying all (or a subset of) an existing State.

        Args:
            state (State): a State instance
            sample_indices (optional; tensor): a tensor of sample indices

        Returns:
            state object

        """
        if sample_indices is None:
            return cls([be.copy_tensor(t) for t in state.units])
        return cls([be.index_select(t, sample_indices, 0) for t in state.units])

    @classmethod
    def separate_samples(cls, state, sample_indices):
        """
        Separate a single state into a list of states, each containing unit
        values from a single sample.

        Args:
            state (State)
            sample_indices (optional; tensor): a tensor of sample indices

        Returns:
            List[state]

        """
        return [State.from_state(state, be.long_tensor([i])) for i in sample_indices]

    def get_visible(self):
        """
        Extract the visible units

        Args:
            None

        Returns:
            vis (tensor (num_samples, num_visible)): visible unit values.

        """
        return self.units[0]


class StateTAP(object):
    """
    A StateTAP is a list of CumulantsTAP objects for each layer in the model.

    """
    def __init__(self, cumulants, lagrange_multipliers):
        """
        Create a StateTAP.

        Args:
            cumulants: list of CumulantsTAP objects
            lagrange_multipliers: list of CumulantsTAP objects

        Returns:
            StateTAP

        """
        self.cumulants = cumulants
        self.lagrange_multipliers = lagrange_multipliers
        self.len = len(self.cumulants)
        assert len(self.cumulants)==len(self.lagrange_multipliers), \
            "The list of cumulants and lagrange multipliers must be the same length"

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
            model (BoltzmannMachine): a BoltzmannMachine instance

        Returns:
            StateTAP object

        """
        cumulants = [layer.get_zero_magnetization() for layer in model.layers]
        lms = model.lagrange_multipliers_analytic(cumulants)
        return cls(cumulants,lms)

    @classmethod
    def from_model_rand(cls, model, num_samples=1):
        """
        Create a StateTAP object from a model.

        Args:
            model (BoltzmannMachine): a BoltzmannMachine instance
            num_samples (int): number of random samples to draw

        Returns:
            StateTAP object

        """
        cumulants = [layer.get_random_magnetization(num_samples)
                     for layer in model.layers]
        lms = model.lagrange_multipliers_analytic(cumulants)
        return cls(cumulants,lms)


def state_allclose(state1: State, state2: State,
                   rtol:float=1e-05, atol:float=1e-08) -> bool:
    """
    Check that for approximate equality between two states.

    Args:
        state1 (State): a state to compare.
        state2 (State): a state to compare.
        rtol (optional): Relative tolerance.
        atol (optional): Absolute tolerance.

    Returns:
        approximate equality (bool)

    """
    assert len(state1) == len(state2), "States are not the same length"
    return all(be.allclose(state1[l], state2[l], rtol, atol)
               for l in range(len(state1)))
