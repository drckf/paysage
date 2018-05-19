from paysage import backends as be

def pdist(x: be.Tensor, y: be.Tensor) -> be.Tensor:
    """
    Compute the pairwise distance matrix between the rows of x and y.

    Args:
        x (tensor (num_samples_1, num_units))
        y (tensor (num_samples_2, num_units))

    Returns:
        tensor (num_samples_1, num_samples_2)

    """
    inner = be.dot(x, be.transpose(y))
    x_mag = be.norm(x, axis=1) ** 2
    y_mag = be.norm(y, axis=1) ** 2
    squared = be.add(be.unsqueeze(y_mag, axis=0), be.add(be.unsqueeze(x_mag, axis=1), -2*inner))
    return be.sqrt(be.clip(squared, a_min=0))

def find_k_nearest_neighbors(x: be.Tensor, y: be.Tensor, k: int, callbacks=None) \
                                    -> be.Tuple[be.Tensor, be.Tensor]:
    """
    For each row in x, find the kth nearest row in y.
    The algorithm actually computes all K <= k neighbors.
    Callbacks can be used to learn from this sequence.

    Args:
        x (tensor (num_samples_x, num_units))
        y (tensor (num_samples_y, num_units))
        k (int > 0)
        callbacks (optional; List[callable]): a list of functions with signature
            func(neighbor_indices, neighbor_distances) -> None

    Returns:
        indices (long_tensor (num_samples_x,)),
        distances (float_tensor (num_samples_x))

    """
    index = be.trange(0, len(x), dtype=be.Long)
    dist = pdist(x, y)
    max_dist = be.tmax(dist)
    for _ in range(k):
        neighbors = be.argmin(dist, axis=1)
        neighbor_dist = dist[index, neighbors]
        dist[index, neighbors] = max_dist
        if callbacks is not None:
            for func in callbacks:
                func(neighbors, neighbor_dist)
    return neighbors, neighbor_dist

def find_nearest_neighbors(x: be.Tensor, y: be.Tensor, k: int, callbacks=None) \
                                -> be.Tuple[be.Tensor, be.Tensor]:
    """
    For each row in x, find the nearest row in y for each j <= k
    The algorithm actually computes all K <= k neighbors.
    Callbacks can be used to learn from this sequence.

    Args:
        x (tensor (num_samples_x, num_units))
        y (tensor (num_samples_y, num_units))
        k (int > 0)
        callbacks (optional; List[callable]): a list of functions with signature
            func(neighbor_indices, neighbor_distances) -> None

    Returns:
        indices long_tensor (j, num_samples_x),
        distances float_tensor (j, num_samples_x)

    """
    index = be.trange(0, len(x), dtype=be.Long)
    dist = pdist(x, y)
    max_dist = be.tmax(dist)
    #NOTE: big memory allocation here
    num_samples = len(x)
    neighbor_dist = be.zeros((k, num_samples))
    neighbors = be.zeros((k, num_samples), dtype=be.Long)
    for j in range(k):
        neighbors[j,:] = be.argmin(dist, axis=1)
        neighbor_dist[j,:] = dist[index, neighbors[j,:]]
        dist[index, neighbors[j,:]] = max_dist
        if callbacks is not None:
            for func in callbacks:
                func(neighbors[j,:], neighbor_dist[j,:])
    return neighbors, neighbor_dist
