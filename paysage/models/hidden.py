import os
import pandas

from .. import layers
from .. import backends as be
from ..models.initialize import init_hidden as init


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

    def mcstep(self, vis, beta=None):
        """
        Perform a single Gibbs sampling update.
        v -> update h distribution ~ h -> update v distribution ~ v'

        Args:
            vis (batch_size, num_visible): Observed visible units.
            beta (optional, (batch_size, 1)): Inverse temperatures.

        Returns:
            tensor: New visible units (v').

        """
        i = 0
        self.layers[i+1].update(self.layers[i].rescale(vis),
                                self.weights[i].W(), beta)
        hid = self.layers[i+1].sample_state()
        self.layers[i].update(self.layers[i+1].rescale(hid),
                              be.transpose(self.weights[i].W()), beta)
        return self.layers[i].sample_state()

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
        self.layers[i+1].update(self.layers[i].rescale(vis),
                                self.weights[i].W(), beta)
        hid = self.layers[i+1].mean()
        self.layers[i].update(self.layers[i+1].rescale(hid),
                              be.transpose(self.weights[i].W()), beta)
        return self.layers[i].mean()

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
        self.layers[i+1].update(self.layers[i].rescale(vis),
                                  self.weights[i].W(), beta)
        hid = self.layers[i+1].mode()
        self.layers[i].update(self.layers[i+1].rescale(hid),
                              be.transpose(self.weights[i].W()), beta)
        return self.layers[i].mode()

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

        grad = {
        'layers': [None for l in self.layers],
        'weights': [None for w in self.weights]
        }

        W_transpose = be.transpose(self.weights[0].W())

        # POSITIVE PHASE (using observed)

        # 1. Scale vdata
        vdata_scaled = self.layers[i].rescale(vdata)

        # 2. Update the hidden layer
        self.layers[i+1].update(vdata_scaled, self.weights[0].W())

        # 3. Compute the mean of the hidden layer
        hid = self.layers[i+1].mean()

        # 4. Scale the hidden mean
        hid_scaled = self.layers[i+1].rescale(hid)

        # 5. Compute the gradients
        grad['layers'][i] = self.layers[i].derivatives(vdata, hid_scaled,
                                               self.weights[0].W())
        grad['layers'][i+1] = self.layers[i+1].derivatives(hid, vdata_scaled,
                                                           W_transpose)

        grad['weights'][i] = self.weights[i].derivatives(vdata_scaled,
                                                         hid_scaled)

        # NEGATIVE PHASE (using sampled)

        # 1. Scale vdata
        vmodel_scaled = self.layers[i].rescale(vmodel)

        # 2. Update the hidden layer
        self.layers[i+1].update(vmodel_scaled, self.weights[0].W())

        # 3. Compute the mean of the hidden layer
        hid = self.layers[i+1].mean()

        # 4. Scale hidden mean
        hid_scaled = self.layers[i+1].rescale(hid)

        # 5. Compute the gradients
        be.subtract_dicts_inplace(grad['layers'][i],
                                  self.layers[i].derivatives(
                                                 vmodel,
                                                 hid_scaled,
                                                 self.weights[0].W()))
        be.subtract_dicts_inplace(grad['layers'][i+1],
                                  self.layers[i+1].derivatives(
                                                   hid,
                                                   vmodel_scaled,
                                                   W_transpose))

        be.subtract_dicts_inplace(grad['weights'][i],
                                  self.weights[i].derivatives(
                                                  vmodel_scaled,
                                                  hid_scaled))
        return grad

    def parameter_update(self, deltas):
        """
        Update the model parameters.

        Notes:
            Modifies the model parameters in place.

        Args:
            deltas: A dictionary of parameter updates.

        Returns:
            None

        """
        for i in range(self.num_layers):
            self.layers[i].parameter_step(deltas['layers'][i])
        for i in range(self.num_layers - 1):
            self.weights[i].parameter_step(deltas['weights'][i])

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
        i = 0
        energy += self.layers[i].energy(vis)
        energy += self.layers[i+1].energy(vis)
        energy += self.weights[i].energy(vis, hid)
        return energy

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
