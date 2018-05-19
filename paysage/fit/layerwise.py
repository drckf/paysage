from .. import backends as be
from ..models.state import State
from .. import layers
from ..models import BoltzmannMachine
from .. models import Connection
from . import methods
from . import sgd
from .. import preprocess as pre


class LayerwisePretrain(object):
    """
    Pretrain a model in layerwise fashion using the method from:

    "Deep Boltzmann Machines" by Ruslan Salakhutdinov and Geoffrey Hinton

    """
    def __init__(self, model, batch):
        """
        Create a LayerwisePretrain object.

        Args:
            model: a model object
            batch: a batch object

        Returns:
            LayerwisePretrain

        """
        self.model = model
        self.batch = batch

    def _create_transform(self, model, basic_transform):
        """
        Closure that creates a transform function from a model.

        Args:
            model (BoltzmannMachine)

        Returns:
            transform (callable)

        """
        def transform(data):
            # create a state
            state = State.from_visible(basic_transform.compute(data), model)
            # cache the model clamping
            clamping = model.clamped_sampling
            # clamp the visible units
            model.set_clamped_sampling([0])
            # perform a mean field iteration
            state = model.mean_field_iteration(1, state)
            # reset the model clamping
            model.set_clamped_sampling(clamping)
            # return the units of the last layer
            return state[-1]
        return pre.Transformation(transform)

    def _copy_params_from_submodels(self, submodels):
        """
        Copy the parameters from a list of submodels into a single model.

        Notes:
            Modifies the parameters fo the model attribute in place.

        Args:
            submodels (List[BoltzmannMachine]): a stack of RBMs

        Returns:
            None

        """
        # copy the params from the zeroth layer
        self.model.layers[0].set_params(submodels[0].layers[0].get_params())
        # copy the rest of the layer and weight parameters
        for i in range(len(submodels)):
            self.model.layers[i+1].set_params(submodels[i].layers[1].get_params())
            if (i == 0) or (i == len(submodels)-1):
                # keep the weights of the zeroth layer and the last layer
                self.model.connections[i].weights.params.matrix[:] = \
                    submodels[i].connections[0].weights.W()
            else:
                # halve the weights of the other layers
                self.model.connections[i].weights.params.matrix[:] = \
                    0.5 * submodels[i].connections[0].weights.W()

    def train(self, optimizer, num_epochs, mcsteps=1, method=methods.pcd,
              beta_std=0.6, init_method="hinton", negative_phase_batch_size=None,
              verbose=True):
        """
        Train the model layerwise.

        Notes:
            Updates the model parameters in place.

        Args:
            optimizer: an optimizer object
            num_epochs (int): the number of epochs
            mcsteps (int; optional): the number of Monte Carlo steps per gradient
            method (fit.methods obj; optional): the method used to approximate the likelihood
                               gradient [cd, pcd, tap]
            beta_std (float; optional): the standard deviation of the inverse
                temperature of the SequentialMC sampler
            init_method (str; optional): submodel initialization method
            negative_phase_batch_size (int; optional): the batch size for the negative phase.
                If None, matches the positive_phase batch size.
            verbose (bool; optional): print output to stdout

        Returns:
            None

        """
        # create the submodels
        submodels = []
        for i in range(self.model.num_layers -1):
            layer_list = [layers.layer_from_config(self.model.layers[i].get_config()),
                          layers.layer_from_config(self.model.layers[i+1].get_config())]
            w = layers.weights_from_config(self.model.connections[i].weights.get_config())
            connection_list = [Connection(0, 1, w)]
            submodels += [BoltzmannMachine(layer_list, connection_list)]

        # set the multipliers to double the effect of the visible and the last
        # hidden layers
        submodels[0].multipliers = be.float_tensor([2,1])
        submodels[-1].multipliers = be.float_tensor([1,2])

        # cache the learning rate and the transform
        lr_schedule_cache = optimizer.stepsize.copy()
        transform_cache = self.batch.get_transforms()

        for i in range(len(submodels)):
            be.maybe_print('training model {}'.format(i), end="\n\n", verbose=verbose)

            # update the transform
            basic_transform = self.batch.get_transforms()
            if i > 0:
                self.batch.set_transforms({key:self._create_transform(submodels[i-1],
                                               basic_transform[key]) for key in basic_transform})
                # set the parameters of the zeroth layer using the
                # parameters of the first layer of the previous model
                submodels[i].layers[0].set_params(submodels[i-1].layers[1].get_params())
                submodels[i].layers[0].set_fixed_params(submodels[i-1].layers[1].get_param_names())

            # initialize the submodel
            submodels[i].initialize(self.batch, method=init_method)

            # reset the state of the optimizer
            optimizer.reset()
            optimizer.stepsize = lr_schedule_cache

            # set up a sampler
            trainer = sgd.StochasticGradientDescent(submodels[i], self.batch)
            trainer.train(optimizer, num_epochs, method=method, mcsteps=mcsteps,
                          beta_std=beta_std, verbose=verbose,
                          negative_phase_batch_size=negative_phase_batch_size)

        # reset the transform
        self.batch.set_transforms(transform_cache)

        # update the model
        self._copy_params_from_submodels(submodels)
