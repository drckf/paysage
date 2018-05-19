from math import sqrt

from .. import backends as be
from ..samplers import SequentialMC
from ..models.state import State

# ----- CLASSES ----- #

class ModelAssessment(object):

    def __init__(self, data, model, fantasy_steps=10,
                 num_fantasy_particles=None, beta_std=0):
        """
        Create a ModelAssessment object.

        Args:
            data (tensor ~ (num_samples, num_units))
            model (BoltzmannMachine)
            fantasy_steps (int; optional)
            num_fantasy_particles (int; optional)
            beta_std (float; optional)

        """
        self.model = model

        self.data_state = State.from_visible(data, model)

        # generate some fantasy particles from the model
        npart = self.data_state.batch_size() if num_fantasy_particles is None \
        else num_fantasy_particles
        self.model_state = SequentialMC.generate_fantasy_state(model,
                                                               npart,
                                                               fantasy_steps,
                                                               beta_std=beta_std)

        # compute reconstructions
        self.reconstructions = model.compute_reconstructions(data)

    def comparison(self, func, numpy=True):
        """
        Compare a function computed from the data and model states.

        Args:
            func (callable): func: State -> tensor
            numpy (optional; bool): return the arrays in numpy form if true

        Returns:
            data (tensor ~ 1D), model (tensor ~ 1D),
            correlation (float), root mean squared error (float)

        """
        data_tensor = be.flatten(func(self.data_state))
        model_tensor = be.flatten(func(self.model_state))
        corr = be.corr(data_tensor, model_tensor)[0]
        rms = sqrt(be.mean(be.square(be.subtract(data_tensor, model_tensor))))
        if numpy:
            return (be.to_numpy_array(data_tensor), be.to_numpy_array(model_tensor),
                corr, rms)
        return (data_tensor, model_tensor, corr, rms)

    def sample_data(self, sample_indices, layer=0, func=None):
        """
        Select a subset samples from the data state.

        Args:
            sample_indices (tensor): list of indices
            layer (optional; int): the layer to get from the state
            func (optional; callable): a function to apply to the units

        Returns:
            data: List[tensor]

        """
        state_list = State.separate_samples(self.data_state,
                                                sample_indices=sample_indices)
        tmp = [state.get_layer(layer) for state in state_list]
        if func is not None:
            return [be.apply(func, t) for t in tmp]
        return tmp

    def sample_model(self, sample_indices, layer=0, func=None):
        """
        Select a subset samples from the model state.

        Args:
            sample_indices (tensor): list of indices
            layer (optional; int): the layer to get from the state
            func (optional; callable): a function to apply to the units

        Returns:
            model: List[tensor]

        """
        state_list = State.separate_samples(self.model_state,
                                                sample_indices=sample_indices)
        tmp = [state.get_layer(layer) for state in state_list]
        if func is not None:
            return [be.apply(func, t) for t in tmp]
        return tmp

    def sample_reconstructions(self, sample_indices, layer=0, func=None):
        """
        Select a subset samples from the model state.

        Args:
            sample_indices (tensor): list of indices
            layer (optional; int): the layer to get from the state
            func (optional; callable): a function to apply to the units

        Returns:

            data: List[tensor], reconstructions: List[tensor]

        """
        state_list = State.separate_samples(self.reconstructions,
                                                sample_indices=sample_indices)
        recon = [state.get_layer(layer) for state in state_list]
        if func is not None:
            recon = [be.apply(func, t) for t in recon]
        data =  self.sample_data(sample_indices, layer=layer, func=func)
        return data, recon
