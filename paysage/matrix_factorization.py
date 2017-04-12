import numpy as np

class KMeans(object):
    """Class which implements kmeans clustering a la 
    Lloyd's algorithm
    """
    def __init__(self, data, k):
        self.k = k
        
    def fit(self, data, max_it=50, num_attempts=3):
        """Main call to compute a kmeans clustering
        of the data. Averages multiple successful runs
        to achieve a better result.
        """
        self.data = data
        self.N = self.data.shape[0]
        num_success = 0
        final_fit = np.zeros((self.k, self.data.shape[1]), dtype=self.data.dtype)
        for i in range(num_attempts):
            self._initialize()
            res = self.compute_kmeans(max_it)
            if res:
                final_result += self.centers
                num_success += 1

    def compute_kmeans(self, max_it):
        """A single run of the kmeans algorithm
        """
        i_it, converged = 0, False
        while not converged and i_it < max_it:
            old_ids = self.cluster_id
            self.assign()
            if np.all(old_ids==self.cluster_id):
                converged = True
            self.update()        
            i_it += 1

        return converged

    def _initialize(self):
        """Initialization using the Forgy method
        """
        assert self.k < self.N, "Must have more samples than clusters"
        center_ind = np.random.choice(self.N, self.k)
        self.centers = self.data[center_ind,:]
        self.cluster_id = np.zeros((self,N,), dtype=int)

    def assign(self):
        """Assign step, in which we find the closest center to each
        sample
        """
        self.cluster_id = np.argmin(np.sum((self.data[:,:,None] - self.centers[:,:,None].reshape((1,-1,self.k)))**2, axis=1),)

    def update(self):
        """Update step, in which we compute the new cluster centers
        given the updated cluster assignments
        """
        for i in range(self.k):
            self.centers[i,:] = np.mean(self.data[self.cluster_id==i,:], axis=0)
    
        
    

class PCA(object):
    def __init__(self, k):
        self.k = k
       
    def fit(self, data):
        N = data.shape[0]
        data_zm = data - np.mean(data, axis=0)[None,:]
        cov = np.dot(data_zm, data_zm.transpose())/(N-1)
        eig_vals, eig_vecs = np.linalg.eigh(cov)
        pc_inds = np.argsort(-eig_vals)[:self.k] 
        return eig_vecs[:, pc_inds]


def archetype(data):
    pass

