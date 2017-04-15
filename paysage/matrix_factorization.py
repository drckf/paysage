import numpy as np

class KMeans(object):
    """Class which implements kmeans clustering a la 
    Lloyd's algorithm
 
    TODO: fix convergence issues, prevent empty clusters
    """
    def __init__(self, k):
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
            res = self._compute_kmeans(max_it)
            if res:
                final_fit += self.centers
                num_success += 1
        self.centers = final_fit/num_success
        self._assign()       

    def _compute_kmeans(self, max_it):
        """A single run of the kmeans algorithm
        """
        i_it, converged = 0, False
        while not converged and i_it < max_it:
            old_ids = self.cluster_id
            print(self.centers)
            self._assign()
            if np.all(old_ids==self.cluster_id):
                converged = True
            self._update()        
            i_it += 1

        return converged

    def _initialize(self):
        """Initialization using the Forgy method
        """
        assert self.k < self.N, "Must have more samples than clusters"
        center_ind = np.random.choice(self.N, self.k)
        self.centers = self.data[center_ind,:]
        self._assign()        

    def _assign(self):
        """Assign step, in which we find the closest center to each
        sample
        """
        self.cluster_id = np.argmin(np.sum((self.data[:,:,None] - self.centers[:,:,None].reshape((1,-1,self.k)))**2, axis=1), axis=1)

    def _update(self):
        """Update step, in which we compute the new cluster centers
        given the updated cluster assignments
        """
        for i in range(self.k):
            self.centers[i,:] = np.mean(self.data[self.cluster_id==i,:], axis=0)
       
    
class PCA(object):
    def __init__(self, k):
        self.k = k
       
    def fit(self, data):
        assert data.shape[1] > self.k, "must be more initial dimensions \
                                        than the final mapping"
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
   
    data = np.concatenate((np.random.randn(50,2), 10+np.random.randn(50,2)), axis=0)
   
    method = 'pca'
    if method=='kmeans': 
        kmeans = KMeans(2)
        kmeans.fit(data)
        plt.scatter(data[:,0], data[:,1])
        plt.plot(kmeans.centers[:,0], kmeans.centers[:,1], 'r.')
        plt.show()
    elif method=='pca':
        pca = PCA(1)
        new_data = pca.fit(data)
        plt.scatter(new_data, np.ones((new_data.size,)))
        plt.show()
