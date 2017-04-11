# Documentation for Gradient_Util (gradient_util.py)

## class Gradient
Gradient(layers, weights)


## functions

### grad\_accumulate
```py

def grad_accumulate(func, grad)

```



Apply a funciton entrywise over a Gradient object,<br />accumulating the result.<br /><br />Args:<br /> ~ func (callable): function with one argument<br /> ~ grad (Gradient)<br /><br />returns:<br /> ~ float


### grad\_apply
```py

def grad_apply(func, grad)

```



Apply a function entrywise over a Gradient object.<br /><br />Args:<br /> ~ func (callable)<br /> ~ grad (Gradient)<br /><br />Returns:<br /> ~ Gradient


### grad\_apply\_
```py

def grad_apply_(func_, grad)

```



Apply a function entrywise over a Gradient object.<br /><br />Notes:<br /> ~ Modifies elements of grad in place.<br /><br />Args:<br /> ~ func_ (callable, in place operation)<br /> ~ grad (Gradient)<br /><br />Returns:<br /> ~ None


### grad\_fold
```py

def grad_fold(func, grad)

```



Apply a function entrywise over a Gradient objet,<br />combining the result.<br /><br />Args:<br /> ~ func (callable): function with two arguments<br /> ~ grad (Gradient)<br /><br />returns:<br /> ~ float


### grad\_magnitude
```py

def grad_magnitude(grad)

```



Compute the root-mean-square of the gradient.<br /><br />Args:<br /> ~ grad (Gradient)<br /><br />Returns:<br /> ~ magnitude (float)


### grad\_mapzip
```py

def grad_mapzip(func, grad1, grad2)

```



Apply a function entrywise over the zip of two Gradient objects.<br /><br />Args:<br /> ~ func_ (callable, in place operation)<br /> ~ grad (Gradient)<br /><br />Returns:<br /> ~ Gradient


### grad\_mapzip\_
```py

def grad_mapzip_(func_, grad1, grad2)

```



Apply an in place function entrywise over the zip of two Gradient objects.<br /><br />Notes:<br /> ~ Modifies elements of grad1 in place.<br /><br />Args:<br /> ~ func_ (callable, in place operation)<br /> ~ grad1 (Gradient)<br /> ~ grad2 (Gradient)<br /><br />Returns:<br /> ~ None


### namedtuple
```py

def namedtuple(typename, field_names, *, verbose=False, rename=False, module=None)

```



Returns a new subclass of tuple with named fields.<br /><br />>>> Point = namedtuple('Point', ['x', 'y'])<br />>>> Point.__doc__ ~  ~  ~  ~    # docstring for the new class<br />'Point(x, y)'<br />>>> p = Point(11, y=22) ~  ~  ~  # instantiate with positional args or keywords<br />>>> p[0] + p[1] ~  ~  ~  ~  ~  # indexable like a plain tuple<br />33<br />>>> x, y = p ~  ~  ~  ~  ~  ~ # unpack like a regular tuple<br />>>> x, y<br />(11, 22)<br />>>> p.x + p.y ~  ~  ~  ~  ~    # fields also accessible by name<br />33<br />>>> d = p._asdict() ~  ~  ~  ~  # convert to a dictionary<br />>>> d['x']<br />11<br />>>> Point(**d) ~  ~  ~  ~  ~   # convert from a dictionary<br />Point(x=11, y=22)<br />>>> p._replace(x=100) ~  ~  ~    # _replace() is like str.replace() but targets named fields<br />Point(x=100, y=22)

