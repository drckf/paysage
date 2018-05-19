# Documentation for Common (common.py)

## functions

### accumulate
```py

def accumulate(func, a)

```



Accumulates the result of a function over iterable a.<br /><br />For example:<br /><br />'''<br />from collections import namedtuple<br /><br />def square(x):<br />&nbsp;&nbsp;&nbsp;&nbsp;return x**2<br /><br />coords = namedtuple("coordinates", ["x", "y"])<br /><br />a = coords(1,2)<br />b = accumulate(square, a) # 5<br /><br />a = list(a)<br />b = accumulate(add, a) # 5<br /><br />'''<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;func (callable): a function with one argument<br />&nbsp;&nbsp;&nbsp;&nbsp;a (iterable: e.g., list or named tuple)<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;float


### apply
```py

def apply(func, a)

```



Applies a function over iterable a, giving back an<br />object of the same type as a. That is, b[i] = func(a[i]).<br /><br />Warning: this is not meant to be applied to a tensor --it will not work<br /><br />For example:<br /><br />'''<br />from collections import namedtuple<br />from operator import mul<br />from cytoolz import partial<br /><br /># create a function to divide by 2<br />halve = partial(mul, 0.5)<br /><br />coords = namedtuple("coordinates", ["x", "y"])<br /><br />a = coords(1,2)<br />b = apply(halve, a) # coordinates(x=0.5, y=1.0)<br /><br />a = list(a)<br />b = apply(halve, a) # [0.5,1.0]<br /><br />'''<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;func (callable): a function with a single argument<br />&nbsp;&nbsp;&nbsp;&nbsp;a (iterable: e.g., list or named tuple)<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;object of type(a)


### apply\_
```py

def apply_(func_, a)

```



Applies an in place function over iterable a.<br /><br />That is, a[i] = func(a[i]).<br /><br />Warning: this is not meant to be applied to a tensor --it will not work<br /><br />For example:<br /><br />'''<br />from collections import namedtuple<br />import numpy as np<br />import numexpr as ne<br /><br /># create an in place function to divide an array by 2<br />def halve_(x: np.ndarray) -> None:<br />&nbsp;&nbsp;&nbsp;&nbsp;ne.evaluate('0.5 * x', out=x)<br /><br />coords = namedtuple("coordinates", ["x", "y"])<br /><br />a = coords(np.ones(1), 2 * np.ones(1))<br />apply_(halve_, a) # a = coordinates(x=np.array(0.5), y=np.array(1.0))<br /><br />a = list(a)<br />apply_(halve_, a) # a = [np.array(0.25), np.array(0.5)]<br /><br />'''<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;func_ (callable): an in place function of a single argument<br />&nbsp;&nbsp;&nbsp;&nbsp;a (iterable: e.g., list or named tuple)<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;None


### do\_nothing
```py

def do_nothing(anything)

```



Identity function.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;Anything.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;Anything.


### force\_list
```py

def force_list(anything)

```



Wraps anything into a list, if it is not already a list.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;Anything.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;[Anything]


### force\_unlist
```py

def force_unlist(anything)

```



Returns the first element of a list, only if it is a list.<br />Useful for turning [x] into x.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;[Anything]<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;Anything


### is\_namedtuple
```py

def is_namedtuple(obj)

```



This is a dangerous function!<br /><br />We are often applying functions over iterables, but need to handle<br />the namedtuple case specially.<br /><br />This function *is a quick and dirty* check for a namedtuple.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;obj (an object)<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;bool: a bool that should be pretty correlated with whether or<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;not the object is a namedtuple


### mapzip
```py

def mapzip(func, a, b)

```



Applies a function over the zip of iterables a and b,<br />giving back an object of the same type as a. That is,<br />c[i] = func(a[i], b[i]).<br /><br />Warning: this is not meant to be applied to a tensor --it will not work<br /><br />For example:<br /><br />```<br />from collections import namedtuple<br />from operator import add<br /><br />coords = namedtuple("coordinates", ["x", "y"])<br /><br />a = coords(1,2)<br />b = coords(2,3)<br /><br />c = mapzip(add, a, b) # coordinates(x=2, y=4)<br /><br />a = list(a)<br />b = list(b)<br /><br />c = mapzip(add, a, b) # [2, 4]<br />```<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;func (callable): a function with two arguments<br />&nbsp;&nbsp;&nbsp;&nbsp;a (iterable; e.g., list or namedtuple)<br />&nbsp;&nbsp;&nbsp;&nbsp;b (iterable; e.g., list or namedtuple)<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;object of type(a)


### mapzip\_
```py

def mapzip_(func_, a, b)

```



Applies an in place function over the zip of iterables a and b,<br />func(a[i], b[i]).<br /><br />Warning: this is not meant to be applied to a tensor --it will not work<br /><br />For example:<br /><br />```<br />from collections import namedtuple<br />import numpy as np<br />import numexpr as ne<br /><br />def add_(x: np.ndarray, y: np.ndarray) -> None:<br />&nbsp;&nbsp;&nbsp;&nbsp;ne.evaluate('x + y', out=x)<br /><br />coords = namedtuple("coordinates", ["x", "y"])<br /><br />a = coords(np.array([1]), np.array([2]))<br />b = coords(np.array([3]), np.array([4]))<br /><br />mapzip_(add_, a, b) # a = coordinates(x=4, y=6)<br /><br />a = list(a)<br />b = list(b)<br /><br />mapzip_(add_, a, b) # a = [7, 10]<br />```<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;func (callable): an in place function with two arguments<br />&nbsp;&nbsp;&nbsp;&nbsp;a (iterable; e.g., list or namedtuple)<br />&nbsp;&nbsp;&nbsp;&nbsp;b (iterable; e.g., list or namedtuple)<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;None


### maybe\_a
```py

def maybe_a(a, b, func)

```



Compute func(a, b) when a could be None.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;a (any; maybe None)<br />&nbsp;&nbsp;&nbsp;&nbsp;b (any)<br />&nbsp;&nbsp;&nbsp;&nbsp;func (callable)<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;func(a, b) or b if a is None


### maybe\_key
```py

def maybe_key(dictionary, key, default=None, func=<function do_nothing>)

```



Compute func(dictionary['key']) when dictionary has key key, else return default.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;dictionary (dict)<br />&nbsp;&nbsp;&nbsp;&nbsp;default (optional; any): default return value<br />&nbsp;&nbsp;&nbsp;&nbsp;func (callable)<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;func(dictionary[key]) or b if dictionary has no such key


### maybe\_print
```py

def maybe_print(*args, verbose=True, **kwargs)

```



An optional print statement.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;args: some arguments to print<br />&nbsp;&nbsp;&nbsp;&nbsp;verbose (bool): only print if set to True<br />&nbsp;&nbsp;&nbsp;&nbsp;kwargs: some keyword arguments to print<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;None

