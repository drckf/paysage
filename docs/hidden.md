hidden
## class GaussianRestrictedBoltzmannMachine
GaussianRestrictedBoltzmanMachine
RBM with Gaussian visible units.

Hinton, Geoffrey.
"A practical guide to training restricted Boltzmann machines."
Momentum 9.1 (2010): 926.
### \_\_init\_\_
```py

def __init__(self, nvis, nhid, hid_type='bernoulli')

```



### add\_constraints
```py

def add_constraints(self, cons)

```



### add\_weight\_decay
```py

def add_weight_decay(self, penalty, method='l2_penalty')

```



### derivatives
```py

def derivatives(self, visible)

```



### deterministic\_iteration
```py

def deterministic_iteration(self, vis, steps, beta=None)

```



mean_field_iteration(v, n):
v -> h -> v_1 -> h_1 -> ... -> v_n
return v_n


### deterministic\_step
```py

def deterministic_step(self, vis, beta=None)

```



deterministic_step(v):
v -> h -> v'
return v'


### enforce\_constraints
```py

def enforce_constraints(self)

```



### hidden\_mean
```py

def hidden_mean(self, visible, beta=None)

```



### hidden\_mode
```py

def hidden_mode(self, visible, beta=None)

```



### initialize
```py

def initialize(self, data, method='hinton')

```



### joint\_energy
```py

def joint_energy(self, visible, hidden, beta=None)

```



### marginal\_free\_energy
```py

def marginal_free_energy(self, visible, beta=None)

```



### markov\_chain
```py

def markov_chain(self, vis, steps, beta=None)

```



markov_chain(v, n):
v -> h -> v_1 -> h_1 -> ... -> v_n
return v_n


### mcstep
```py

def mcstep(self, vis, beta=None)

```



mcstep(v):
v -> h -> v'
return v'


### mean\_field\_iteration
```py

def mean_field_iteration(self, vis, steps, beta=None)

```



mean_field_iteration(v, n):
v -> h -> v_1 -> h_1 -> ... -> v_n
return v_n


### mean\_field\_step
```py

def mean_field_step(self, vis, beta=None)

```



mean_field_step(v):
v -> h -> v'
return v'

It may be worth looking into extended approaches:
Gabrié, Marylou, Eric W. Tramel, and Florent Krzakala.
"Training Restricted Boltzmann Machine via the￼
Thouless-Anderson-Palmer free energy."
Advances in Neural Information Processing Systems. 2015.


### random
```py

def random(self, visible)

```



### sample\_hidden
```py

def sample_hidden(self, visible, beta=None)

```



### sample\_visible
```py

def sample_visible(self, hidden, beta=None)

```



### visible\_mean
```py

def visible_mean(self, hidden, beta=None)

```



### visible\_mode
```py

def visible_mode(self, hidden, beta=None)

```





## class RestrictedBoltzmannMachine
RestrictedBoltzmanMachine

Hinton, Geoffrey.
"A practical guide to training restricted Boltzmann machines."
Momentum 9.1 (2010): 926.
### \_\_init\_\_
```py

def __init__(self, nvis, nhid, vis_type='ising', hid_type='bernoulli')

```



### add\_constraints
```py

def add_constraints(self, cons)

```



### add\_weight\_decay
```py

def add_weight_decay(self, penalty, method='l2_penalty')

```



### derivatives
```py

def derivatives(self, visible)

```



### deterministic\_iteration
```py

def deterministic_iteration(self, vis, steps, beta=None)

```



mean_field_iteration(v, n):
v -> h -> v_1 -> h_1 -> ... -> v_n
return v_n


### deterministic\_step
```py

def deterministic_step(self, vis, beta=None)

```



deterministic_step(v):
v -> h -> v'
return v'


### enforce\_constraints
```py

def enforce_constraints(self)

```



### hidden\_mean
```py

def hidden_mean(self, visible, beta=None)

```



### hidden\_mode
```py

def hidden_mode(self, visible, beta=None)

```



### initialize
```py

def initialize(self, data, method='hinton')

```



### joint\_energy
```py

def joint_energy(self, visible, hidden, beta=None)

```



### marginal\_free\_energy
```py

def marginal_free_energy(self, visible, beta=None)

```



### markov\_chain
```py

def markov_chain(self, vis, steps, beta=None)

```



markov_chain(v, n):
v -> h -> v_1 -> h_1 -> ... -> v_n
return v_n


### mcstep
```py

def mcstep(self, vis, beta=None)

```



mcstep(v):
v -> h -> v'
return v'


### mean\_field\_iteration
```py

def mean_field_iteration(self, vis, steps, beta=None)

```



mean_field_iteration(v, n):
v -> h -> v_1 -> h_1 -> ... -> v_n
return v_n


### mean\_field\_step
```py

def mean_field_step(self, vis, beta=None)

```



mean_field_step(v):
v -> h -> v'
return v'

It may be worth looking into extended approaches:
Gabrié, Marylou, Eric W. Tramel, and Florent Krzakala.
"Training Restricted Boltzmann Machine via the￼
Thouless-Anderson-Palmer free energy."
Advances in Neural Information Processing Systems. 2015.


### random
```py

def random(self, visible)

```



### sample\_hidden
```py

def sample_hidden(self, visible, beta=None)

```



### sample\_visible
```py

def sample_visible(self, hidden, beta=None)

```



### visible\_mean
```py

def visible_mean(self, hidden, beta=None)

```



### visible\_mode
```py

def visible_mode(self, hidden, beta=None)

```





## class HopfieldModel
HopfieldModel
A model of associative memory with binary visible units and
Gaussian hidden units.

Hopfield, John J.
"Neural networks and physical systems with emergent collective
computational abilities."
Proceedings of the national academy of sciences 79.8 (1982): 2554-2558.
### \_\_init\_\_
```py

def __init__(self, nvis, nhid, vis_type='ising')

```



### add\_constraints
```py

def add_constraints(self, cons)

```



### add\_weight\_decay
```py

def add_weight_decay(self, penalty, method='l2_penalty')

```



### derivatives
```py

def derivatives(self, visible)

```



### deterministic\_iteration
```py

def deterministic_iteration(self, vis, steps, beta=None)

```



mean_field_iteration(v, n):
v -> h -> v_1 -> h_1 -> ... -> v_n
return v_n


### deterministic\_step
```py

def deterministic_step(self, vis, beta=None)

```



deterministic_step(v):
v -> h -> v'
return v'


### enforce\_constraints
```py

def enforce_constraints(self)

```



### hidden\_mean
```py

def hidden_mean(self, visible, beta=None)

```



### hidden\_mode
```py

def hidden_mode(self, visible, beta=None)

```



### initialize
```py

def initialize(self, data, method='hinton')

```



### joint\_energy
```py

def joint_energy(self, visible, hidden, beta=None)

```



### marginal\_free\_energy
```py

def marginal_free_energy(self, visible, beta=None)

```



### markov\_chain
```py

def markov_chain(self, vis, steps, beta=None)

```



markov_chain(v, n):
v -> h -> v_1 -> h_1 -> ... -> v_n
return v_n


### mcstep
```py

def mcstep(self, vis, beta=None)

```



mcstep(v):
v -> h -> v'
return v'


### mean\_field\_iteration
```py

def mean_field_iteration(self, vis, steps, beta=None)

```



mean_field_iteration(v, n):
v -> h -> v_1 -> h_1 -> ... -> v_n
return v_n


### mean\_field\_step
```py

def mean_field_step(self, vis, beta=None)

```



mean_field_step(v):
v -> h -> v'
return v'

It may be worth looking into extended approaches:
Gabrié, Marylou, Eric W. Tramel, and Florent Krzakala.
"Training Restricted Boltzmann Machine via the￼
Thouless-Anderson-Palmer free energy."
Advances in Neural Information Processing Systems. 2015.


### random
```py

def random(self, visible)

```



### sample\_hidden
```py

def sample_hidden(self, visible, beta=None)

```



### sample\_visible
```py

def sample_visible(self, hidden, beta=None)

```



### visible\_mean
```py

def visible_mean(self, hidden, beta=None)

```



### visible\_mode
```py

def visible_mode(self, hidden, beta=None)

```





## class LatentModel
LatentModel
Abstract class for a 2-layer neural network.
### \_\_init\_\_
```py

def __init__(self)

```



### add\_constraints
```py

def add_constraints(self, cons)

```



### add\_weight\_decay
```py

def add_weight_decay(self, penalty, method='l2_penalty')

```



### deterministic\_iteration
```py

def deterministic_iteration(self, vis, steps, beta=None)

```



mean_field_iteration(v, n):
v -> h -> v_1 -> h_1 -> ... -> v_n
return v_n


### deterministic\_step
```py

def deterministic_step(self, vis, beta=None)

```



deterministic_step(v):
v -> h -> v'
return v'


### enforce\_constraints
```py

def enforce_constraints(self)

```



### marginal\_free\_energy
```py

def marginal_free_energy(self, visible, beta=None)

```



### markov\_chain
```py

def markov_chain(self, vis, steps, beta=None)

```



markov_chain(v, n):
v -> h -> v_1 -> h_1 -> ... -> v_n
return v_n


### mcstep
```py

def mcstep(self, vis, beta=None)

```



mcstep(v):
v -> h -> v'
return v'


### mean\_field\_iteration
```py

def mean_field_iteration(self, vis, steps, beta=None)

```



mean_field_iteration(v, n):
v -> h -> v_1 -> h_1 -> ... -> v_n
return v_n


### mean\_field\_step
```py

def mean_field_step(self, vis, beta=None)

```



mean_field_step(v):
v -> h -> v'
return v'

It may be worth looking into extended approaches:
Gabrié, Marylou, Eric W. Tramel, and Florent Krzakala.
"Training Restricted Boltzmann Machine via the￼
Thouless-Anderson-Palmer free energy."
Advances in Neural Information Processing Systems. 2015.


### random
```py

def random(self, visible)

```



### sample\_hidden
```py

def sample_hidden(self, visible, beta=None)

```



### sample\_visible
```py

def sample_visible(self, hidden, beta=None)

```




