import pandas
from math import sqrt

from .. import backends as be
from .. import math_utils
from ..metrics.generator_metrics import KLDivergence, ReverseKLDivergence

class PCA(object):

    def __init__(self, num_components, stepsize=0.001):
        """
        Computes the principal components of a dataset using stochastic gradient
        descent.

        Arora, Raman, et al.
        "Stochastic optimization for PCA and PLS."
        Communication, Control, and Computing (Allerton), 2012
        50th Annual Allerton Conference on. IEEE, 2012.

        Args:
            num_components (int): The number of directions to extract.
            stepsize (optional): Learning rate schedule.

        Returns:
            PCA

        """
        self.num_units = None
        self.num_components = num_components
        self.W = None
        self.var = None
        self.mean = None
        self.stepsize = stepsize
        self.var_calc = math_utils.MeanVarianceArrayCalculator()
        self.mean_calc = math_utils.MeanArrayCalculator()

    def save(self, store: pandas.HDFStore, num_components_save: int = None) -> None:
        """
        Save the PCA transform in an HDFStore.
        Allows to save only the first num_components_save.

        Notes:
            Performs an IO operation.

        Args:
            store (pandas.HDFStore)
            num_components_save (int): the number of principal components to save.
                If None, all are saved.

        Returns:
            None

        """
        n = num_components_save if num_components_save is not None \
            else self.num_components
        assert n <= self.num_components

        # the config
        config = {'num_components': n,
                  'stepsize': self.stepsize}
        store.put('pca', pandas.DataFrame())
        store.get_storer('pca').attrs.config = config

        # the parameters
        store.put('pca/W', pandas.DataFrame(be.to_numpy_array(self.W[:,:n])))
        store.put('pca/var', pandas.DataFrame(be.to_numpy_array(self.var[:n])))
        # check if the mean exists before saving
        if self.mean is not None:
            store.put('pca/mean', pandas.DataFrame(be.to_numpy_array(self.mean)))
        var_calc_df = self.var_calc.to_dataframe()
        # if fit from SVD, there is no calculator used
        if var_calc_df is not None:
            store.put('pca/var_calc', var_calc_df.iloc[:n])

    @classmethod
    def from_saved(cls, store: pandas.HDFStore) -> None:
        """
        Create the PCA from its saved parameters.

        Notes:
            Performs an IO operation.

        Args:
            store (pandas.HDFStore)

        Returns:
            PCA

        """
        config = store.get_storer('pca').attrs.config
        pca = cls(config['num_components'], config['stepsize'])
        pca.W = be.float_tensor(store.get('pca/W').as_matrix())
        pca.var = be.float_tensor(store.get('pca/var').as_matrix()[:,0])
        # check the mean is present
        if 'pca/mean' in store.keys():
            pca.mean = be.float_tensor(store.get('pca/mean').as_matrix()[:,0])
        # if the saved PCA was fit from SVD, there is not calculator defined
        if 'pca/var_calc' in store.keys():
            pca.var_calc = math_utils.MeanVarianceArrayCalculator.from_dataframe(
                            store.get('pca/var_calc'))
        else:
            pca.var_calc = math_utils.MeanVarianceArrayCalculator()
        return pca

    def _try_to_initialize_W(self, tensor):
        """
        Initialize the principal components, if necessary.

        Notes:
            Modifes the PCA.W attribute in place!

        Args:
            tensor (num_samples, num_units)

        Returns:
            None

        """
        if self.W is None:
            self.num_units = be.shape(tensor)[1]
            self.W, _ = be.qr(be.randn((self.num_units, self.num_components)))

    def train_on_batch(self, tensor, grad_steps=1, orthogonalize=False):
        """
        Update the principal components using stochastic gradient descent.

        Notes:
            Modifies the PCA.W attribute in place!

        Args:
            tensor (num_samples, num_units): a batch of data
            grad_steps (int): the number of gradient steps to make

        Returns:
            None

        """
        self._try_to_initialize_W(tensor)

        # center the data
        centered = be.center(tensor)

        for _ in range(grad_steps):
            # compute the gradient
            grad = be.dot(be.transpose(centered),
                          be.dot(centered, self.W)) / len(centered)

            # perform a stochastic gradient update of the loadings
            self.W += self.stepsize * grad

        if orthogonalize:
            self.W, _ = be.qr(self.W)

    def update_variance_on_batch(self, tensor):
        """
        Update the variances along the principle directions.

        Notes:
            Modifies the PCA.var_cal attribute in place!

        Args:
            tensor (num_samples, num_units)

        Returns:
            None

        """
        # center the data
        centered = be.center(tensor)

        # perform a batch update of the variance
        self.var_calc.update(be.dot(centered, self.W))
        self.mean_calc.update(be.unsqueeze(be.mean(tensor, axis=0), axis=0))

    def compute_error_on_batch(self, tensor):
        """
        Compute the reconstruction error || X - X W W^T ||^2.

        Args:
            tensor (num_samples, num_units)

        Returns:
            float

        """
        # center the data
        centered = be.center(tensor)

        recon = be.dot(be.dot(centered, self.W), be.transpose(self.W))
        return be.norm(be.subtract(recon, centered)) ** 2

    def compute_validation_error(self, batch):
        """
        Compute the root-mean-squared reconstruction error from the
        validation set.

        Args:
            batch: a batch object

        Returns:
            float

        """
        num_samples = 0
        error = 0
        while True:
            try:
                v_data = batch.get(mode='validate')
            except StopIteration:
                break
            num_samples += len(v_data)
            error += self.compute_error_on_batch(v_data)
        return sqrt(error / num_samples)

    def sample_pca(self, n):
        """
        Sample from the multivariate Gaussian represented by the pca

        Args:
            n (int): number of samples

        Returns:
            samples (tensor (n, num_units))

        """
        r = be.multiply(be.randn((n, self.num_components)),
                        be.unsqueeze(be.sqrt(self.var), axis=0))
        return be.dot(r, be.transpose(self.W)) + self.mean

    def compute_validation_kld(self, batch):
        """
        Compute the KL divergence between the pca distribution and the
        distribution of the validation set.

        Args:
            batch: a batch object

        Returns:
            float

        """
        kld = 0.
        batches = 0.
        while True:
            try:
                v_data = batch.get(mode='validate')
            except StopIteration:
                break
            # generate PCA samples
            f_data = self.sample_pca(be.shape(v_data)[0])
            kld += KLDivergence.klpq(v_data, f_data, 5, True)
            batches += 1.
        return kld / batches

    def compute_validation_rkld(self, batch):
        """
        Compute the Reverse KL divergence between the pca distribution and the
        distribution of the validation set.

        Args:
            batch: a batch object

        Returns:
            float

        """
        rkld = 0.
        batches = 0.
        while True:
            try:
                v_data = batch.get(mode='validate')
            except StopIteration:
                break
            # generate PCA samples
            f_data = self.sample_pca(be.shape(v_data)[0])
            rkld += ReverseKLDivergence.klqp(v_data, f_data, 5, True)
            batches += 1.
        return rkld / batches

    def project(self, tensor):
        """
        Project a tensor onto the principal components.

        Args:
            tensor (num_samples, num_units)

        Returns:
            tensor (num_samples, num_components)

        """
        return be.dot(tensor, self.W)

    def transform(self, tensor):
        """
        Transform a tensor by removing the global mean and projecting.

        Args:
            tensor (num_samples, num_units)

        Returns:
            tensor (num_samples, num_components)

        """
        return self.project(tensor - self.mean)

    @classmethod
    def from_batch(cls, batch, num_components, epochs=1, grad_steps_per_minibatch=1,
                   stepsize=1e-3, minimum_stepsize=1e-8, convergence=1e-2,
                   orthogonalize_rate=5, verbose=True):

        """
        Computes the principal components of a dataset using stochastic gradient
        descent.

        Arora, Raman, et al.
        "Stochastic optimization for PCA and PLS."
        Communication, Control, and Computing (Allerton), 2012
        50th Annual Allerton Conference on. IEEE, 2012.

        Args:
            batch: A batch object.
            num_components (int): The number of directions to extract.
            epochs (int; optional): The number of epochs.
            grad_steps_per_minibatch(int; optional): Gradient steps.
            stepsize (optional; float): initial learning rate
            minimum_stepsize (optional; float): minimum stepsize before terminating
            convergence (optional; float): convergence error threshold
            orthogonalize_rate: (optional; int): how many minibatches to skip
                before orthogonalizing W
            verbose (optional; bool): whether or not to print to the screen

        Returns:
            PCA

        """
        pca = cls(num_components, stepsize)

        # compute the principal components
        error = None
        cache = None
        converged = False

        be.maybe_print("PCA with stochastic gradient descent.", verbose=verbose)

        for t in range(epochs):
            c = 0
            while True:
                try:
                    v_data = batch.get(mode='train')
                except StopIteration:
                    break
                c += 1
                ortho = (orthogonalize_rate > 0 and c % orthogonalize_rate==0)
                pca.train_on_batch(v_data, grad_steps=grad_steps_per_minibatch,
                                   orthogonalize=ortho)

            # orthogonalize W
            # note that this doesn't have to be done every gradient step (per Arora)
            # so we only do it once per epoch to save computation
            pca.W, _ = be.qr(pca.W)

            # compute the error on the validation set
            new_error = pca.compute_validation_error(batch)
            if error is not None:
                delta = new_error - error
                converged = delta < 0 and abs(delta) < convergence

            # adjust the stepsize using bold driver
            if error is None or new_error < error:
                error = new_error
                cache = be.copy_tensor(pca.W)
                pca.stepsize *= 1.1
            else:
                pca.W = be.copy_tensor(cache)
                pca.stepsize *= 0.5

            be.maybe_print("PCA epoch {}".format(1 + t), verbose=verbose)
            be.maybe_print("- Current stepsize: {0:.10f}".format(pca.stepsize),
                           verbose=verbose)
            be.maybe_print("- Reconstruction error: {0:.10f}".format(error),
                           verbose=verbose)

            if converged or pca.stepsize < minimum_stepsize:
                be.maybe_print("Reached PCA convergence criterion.",
                               verbose=verbose)
                break

        be.maybe_print("Computing variances along principal directions.",
                       verbose=verbose)
        while True:
            try:
                v_data = batch.get(mode='train')
            except StopIteration:
                break
            pca.update_variance_on_batch(v_data)

        # set mean
        pca.mean = pca.mean_calc.mean
        # reorder the principal components by variance
        pca.var = pca.var_calc.var
        order = be.argsort(-pca.var)
        pca.var = pca.var[order]
        pca.W = pca.W[:, order]

        kld = pca.compute_validation_kld(batch)
        be.maybe_print("- KLDivergence: {0:.10f}".format(kld), verbose=verbose)
        rkld = pca.compute_validation_rkld(batch)
        be.maybe_print("- ReverseKLDivergence: {0:.10f}".format(rkld), verbose=verbose)

        be.maybe_print("PCA done.\n", verbose=verbose)
        return pca

    @classmethod
    def from_svd(cls, tensor, num_components, verbose=True):
        """
        Fit PCA on a single tensor using SVD, rather than minibatch SGD.

        Args:
            tensor (num_samples, num_units): the complete data

        Returns:
            None

        """
        pca = cls(num_components)

        tensor_center, pca.mean = be.center(tensor), be.mean(tensor, axis=0)
        _, s, w = be.svd(tensor_center)
        pca.W = w[:, :num_components]
        pca.var = be.square(s[:num_components]) / (len(tensor) - 1)
        be.maybe_print("PCA done.\n", verbose=verbose)
        return pca
