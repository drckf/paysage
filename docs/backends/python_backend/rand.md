# Documentation for Rand (rand.py)

## functions

### rand
```py

def rand(shape: Tuple[int]) -> numpy.ndarray

```



Generate a tensor of the specified shape filled with uniform random numbers<br />between 0 and 1.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;shape: Desired shape of the random tensor.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor: Random numbers between 0 and 1.


### rand\_int
```py

def rand_int(a: int, b: int, shape: Tuple[int]) -> numpy.ndarray

```



Generate random integers in [a, b).<br />Fills a tensor of a given shape<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;a (int): the minimum (inclusive) of the range.<br />&nbsp;&nbsp;&nbsp;&nbsp;b (int): the maximum (exclusive) of the range.<br />&nbsp;&nbsp;&nbsp;&nbsp;shape: the shape of the output tensor.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor (shape): the random integer samples.


### rand\_like
```py

def rand_like(tensor: numpy.ndarray) -> numpy.ndarray

```



Generate a tensor of the same shape as the specified tensor<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor: tensor with desired shape.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor: Random numbers between 0 and 1.


### rand\_samples
```py

def rand_samples(tensor: numpy.ndarray, num: int) -> numpy.ndarray

```



Collect a random number samples from a tensor with replacement.<br />Only supports the input tensor being a vector.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor ((num_samples)): a vector of values.<br />&nbsp;&nbsp;&nbsp;&nbsp;num (int): the number of samples to take.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;samples ((num)): a vector of sampled values.


### rand\_softmax
```py

def rand_softmax(phi: numpy.ndarray) -> numpy.ndarray

```



Draw random 1-hot samples according to softmax probabilities.<br /><br />Given an effective field vector v,<br />the softmax probabilities are p = exp(v) / sum(exp(v))<br /><br />A 1-hot vector x is sampled according to p.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;phi (tensor (batch_size, num_units)): the effective field<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor (batch_size, num_units): random 1-hot samples<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;from the softmax distribution.


### rand\_softmax\_units
```py

def rand_softmax_units(phi: numpy.ndarray) -> numpy.ndarray

```



Draw random unit values according to softmax probabilities.<br /><br />Given an effective field vector v,<br />the softmax probabilities are p = exp(v) / sum(exp(v))<br /><br />The unit values (the on-units for a 1-hot encoding)<br />are sampled according to p.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;phi (tensor (batch_size, num_units)): the effective field<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor (batch_size,): random unit values from the softmax distribution.


### randn
```py

def randn(shape: Tuple[int]) -> numpy.ndarray

```



Generate a tensor of the specified shape filled with random numbers<br />drawn from a standard normal distribution (mean = 0, variance = 1).<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;shape: Desired shape of the random tensor.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor: Random numbers between from a standard normal distribution.


### randn\_like
```py

def randn_like(tensor: numpy.ndarray) -> numpy.ndarray

```



Generate a tensor of the same shape as the specified tensor<br />filled with normal(0,1) random numbers<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor: tensor with desired shape.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor: Random numbers between from a standard normal distribution.


### set\_seed
```py

def set_seed(n: int = 137) -> None

```



Set the seed of the random number generator.<br /><br />Notes:<br />&nbsp;&nbsp;&nbsp;&nbsp;Default seed is 137.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;n: Random seed.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;None


### shuffle\_
```py

def shuffle_(tensor: numpy.ndarray) -> None

```



Shuffle the rows of a tensor.<br /><br />Notes:<br />&nbsp;&nbsp;&nbsp;&nbsp;Modifies tensor in place.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor (shape): a tensor to shuffle.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;None

