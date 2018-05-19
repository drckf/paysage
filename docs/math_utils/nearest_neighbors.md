# Documentation for Nearest_Neighbors (nearest_neighbors.py)

## functions

### find\_k\_nearest\_neighbors
```py

def find_k_nearest_neighbors(x: Union[numpy.ndarray, torch.IntTensor, torch.cuda.IntTensor, torch.ShortTensor, torch.cuda.ShortTensor, torch.LongTensor, torch.cuda.LongTensor, torch.ByteTensor, torch.cuda.ByteTensor, torch.FloatTensor, torch.cuda.FloatTensor, torch.DoubleTensor, torch.cuda.DoubleTensor], y: Union[numpy.ndarray, torch.IntTensor, torch.cuda.IntTensor, torch.ShortTensor, torch.cuda.ShortTensor, torch.LongTensor, torch.cuda.LongTensor, torch.ByteTensor, torch.cuda.ByteTensor, torch.FloatTensor, torch.cuda.FloatTensor, torch.DoubleTensor, torch.cuda.DoubleTensor], k: int, callbacks=None) -> Tuple[Union[numpy.ndarray, torch.IntTensor, torch.cuda.IntTensor, torch.ShortTensor, torch.cuda.ShortTensor, torch.LongTensor, torch.cuda.LongTensor, torch.ByteTensor, torch.cuda.ByteTensor, torch.FloatTensor, torch.cuda.FloatTensor, torch.DoubleTensor, torch.cuda.DoubleTensor], Union[numpy.ndarray, torch.IntTensor, torch.cuda.IntTensor, torch.ShortTensor, torch.cuda.ShortTensor, torch.LongTensor, torch.cuda.LongTensor, torch.ByteTensor, torch.cuda.ByteTensor, torch.FloatTensor, torch.cuda.FloatTensor, torch.DoubleTensor, torch.cuda.DoubleTensor]]

```



For each row in x, find the kth nearest row in y.<br />The algorithm actually computes all K <= k neighbors.<br />Callbacks can be used to learn from this sequence.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;x (tensor (num_samples_x, num_units))<br />&nbsp;&nbsp;&nbsp;&nbsp;y (tensor (num_samples_y, num_units))<br />&nbsp;&nbsp;&nbsp;&nbsp;k (int > 0)<br />&nbsp;&nbsp;&nbsp;&nbsp;callbacks (optional; List[callable]): a list of functions with signature<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;func(neighbor_indices, neighbor_distances) -> None<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;indices (long_tensor (num_samples_x,)),<br />&nbsp;&nbsp;&nbsp;&nbsp;distances (float_tensor (num_samples_x))


### find\_nearest\_neighbors
```py

def find_nearest_neighbors(x: Union[numpy.ndarray, torch.IntTensor, torch.cuda.IntTensor, torch.ShortTensor, torch.cuda.ShortTensor, torch.LongTensor, torch.cuda.LongTensor, torch.ByteTensor, torch.cuda.ByteTensor, torch.FloatTensor, torch.cuda.FloatTensor, torch.DoubleTensor, torch.cuda.DoubleTensor], y: Union[numpy.ndarray, torch.IntTensor, torch.cuda.IntTensor, torch.ShortTensor, torch.cuda.ShortTensor, torch.LongTensor, torch.cuda.LongTensor, torch.ByteTensor, torch.cuda.ByteTensor, torch.FloatTensor, torch.cuda.FloatTensor, torch.DoubleTensor, torch.cuda.DoubleTensor], k: int, callbacks=None) -> Tuple[Union[numpy.ndarray, torch.IntTensor, torch.cuda.IntTensor, torch.ShortTensor, torch.cuda.ShortTensor, torch.LongTensor, torch.cuda.LongTensor, torch.ByteTensor, torch.cuda.ByteTensor, torch.FloatTensor, torch.cuda.FloatTensor, torch.DoubleTensor, torch.cuda.DoubleTensor], Union[numpy.ndarray, torch.IntTensor, torch.cuda.IntTensor, torch.ShortTensor, torch.cuda.ShortTensor, torch.LongTensor, torch.cuda.LongTensor, torch.ByteTensor, torch.cuda.ByteTensor, torch.FloatTensor, torch.cuda.FloatTensor, torch.DoubleTensor, torch.cuda.DoubleTensor]]

```



For each row in x, find the nearest row in y for each j <= k<br />The algorithm actually computes all K <= k neighbors.<br />Callbacks can be used to learn from this sequence.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;x (tensor (num_samples_x, num_units))<br />&nbsp;&nbsp;&nbsp;&nbsp;y (tensor (num_samples_y, num_units))<br />&nbsp;&nbsp;&nbsp;&nbsp;k (int > 0)<br />&nbsp;&nbsp;&nbsp;&nbsp;callbacks (optional; List[callable]): a list of functions with signature<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;func(neighbor_indices, neighbor_distances) -> None<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;indices long_tensor (j, num_samples_x),<br />&nbsp;&nbsp;&nbsp;&nbsp;distances float_tensor (j, num_samples_x)


### pdist
```py

def pdist(x: Union[numpy.ndarray, torch.IntTensor, torch.cuda.IntTensor, torch.ShortTensor, torch.cuda.ShortTensor, torch.LongTensor, torch.cuda.LongTensor, torch.ByteTensor, torch.cuda.ByteTensor, torch.FloatTensor, torch.cuda.FloatTensor, torch.DoubleTensor, torch.cuda.DoubleTensor], y: Union[numpy.ndarray, torch.IntTensor, torch.cuda.IntTensor, torch.ShortTensor, torch.cuda.ShortTensor, torch.LongTensor, torch.cuda.LongTensor, torch.ByteTensor, torch.cuda.ByteTensor, torch.FloatTensor, torch.cuda.FloatTensor, torch.DoubleTensor, torch.cuda.DoubleTensor]) -> Union[numpy.ndarray, torch.IntTensor, torch.cuda.IntTensor, torch.ShortTensor, torch.cuda.ShortTensor, torch.LongTensor, torch.cuda.LongTensor, torch.ByteTensor, torch.cuda.ByteTensor, torch.FloatTensor, torch.cuda.FloatTensor, torch.DoubleTensor, torch.cuda.DoubleTensor]

```



Compute the pairwise distance matrix between the rows of x and y.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;x (tensor (num_samples_1, num_units))<br />&nbsp;&nbsp;&nbsp;&nbsp;y (tensor (num_samples_2, num_units))<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor (num_samples_1, num_samples_2)

