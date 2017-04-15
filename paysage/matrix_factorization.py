import numpy as np

class KMeans(object):
    """Class which implements kmeans clustering with Lloyd's algorithm"""

    def __init__(self, k):
        """
        Specify the number of clusters, which corresponds to the number of 
        hidden layers if this is being used as a weight initialization.
        
        Args:
            k (int): number of clusters
    
        Returns:
            KMeans object
        """
        self.k = k
        
    def fit(self, data, max_it=50, num_attempts=3):
        """
        Main call to compute a kmeans clustering of the data. Uses the best of 
        multiple runs to achieve a better result. Cluster labels are stored in 
        the cluster_id attribute, while 

        Args:
            data (tensor (num_samples, num_dim)):
                Data matrix to cluster
            max_it (int, optional): 
                Maximum number of iterations alotted
            num_attempts (int, optional): 
                Number of times to run kmeans for finding better solution

        Returns:
            None
        """
        self.data = data
        self.N = self.data.shape[0]
        best_loss = np.inf 
        for i in range(num_attempts):
            self._initialize()
            res = self._compute_kmeans(max_it)
            if res:
                l2 = self._compute_loss()
                if l2 < best_loss:
                    final_centers = self.centers
                    best_loss = l2
        assert not np.isinf(best_loss), "never converged"
        self.centers = final_centers
        self._assign()       

    def _compute_kmeans(self, max_it):
        """
        A single run of the kmeans algorithm
        """
        i_it, converged = 0, False
        while not converged and i_it < max_it: 
            old_ids = self.cluster_id
            self._assign()
            if np.all(old_ids==self.cluster_id):
                converged = True
            self._update()        
            i_it += 1

        return converged

    def _initialize(self):
        """
        Initialization using the Forgy method
        """
        assert self.k < self.N, "Must have more samples than clusters"
        self.cluster_id = np.random.randint(0, self.k, (self.N,))
        center_ind = np.random.choice(self.N, self.k)
        self.centers = self.data[center_ind,:]

    def _assign(self):
        """
        Assign step, in which we find the closest center to each sample
        """
        diff = self.data[:,:,None] - self.centers[:,:,None].transpose((2,1,0))
        self.cluster_id = np.argmin(np.sum(diff**2, axis=1), axis=1)

    def _update(self):
        """
        Update step, in which we compute the new cluster centers given the 
        updated cluster assignments
        """
        for i in range(self.k):
            self.centers[i,:] = np.mean(self.data[self.cluster_id==i,:], axis=0)

    def _compute_loss(self):
        """
        Compute the squared intra-cluster L2 norm
        """
        return np.sum((self.data - self.centers[self.cluster_id,:])**2)
    

class PCA(object):
    """Class which implements PCA"""

    def __init__(self, k):
        """
        Initialize visible to hidden weights using a PCA projection
         
        Args:
            k (int): number of hidden layers
           
        Returns:
            PCA object
        """
        self.k = k
       
    def fit(self, data):
        """
        Args:
            data (tensor (num_samples, num_dim)): 
                Input data matrix.

        Returns:
            tensor (num_visible, num_hidden): initialization for new weights
        """
        assert data.shape[1] > self.k, "must be more initial dimensions than \
                                        the final mapping"
        N = data.shape[0]
        data_zm = data - np.mean(data, axis=0)[None,:]
        cov = np.dot(data_zm, data_zm.transpose())/(N-1)
        eig_vals, eig_vecs = np.linalg.eigh(cov)
        pc_inds = np.argsort(-eig_vals)[:self.k] 
        return eig_vecs[:, pc_inds]


def archetype(data):
    pass


if __name__=='__main__':
    import matplotlib.pyplot as plt
   
    data = np.concatenate((np.random.randn(50,2), \
                           10+np.random.randn(50,2)), axis=0)
   
    method = 'kmeans'
    if method=='kmeans': 
        kmeans = KMeans(2)
        kmeans.fit(data)
        plt.scatter(data[:,0], data[:,1], 30, kmeans.cluster_id)
        plt.plot(kmeans.centers[:,0], kmeans.centers[:,1], 'r.')
        plt.show()
    elif method=='pca':
        pca = PCA(1)
        new_data = pca.fit(data)
        plt.scatter(new_data, np.ones((new_data.size,)))
        plt.show()
