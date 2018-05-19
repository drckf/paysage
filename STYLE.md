Unless stated otherwise, refer to the [Google Style Guide for Python](https://google.github.io/styleguide/pyguide.html).

The most important things:

1) Use informative names for variables, even if that makes your code longer. For example,
```
num_rows, num_columns = shape(matrix)
for row in range(num_rows):
  for column in range(num_columns):
    print(matrix[row, column])
```

2) Follow the docstring format.
```
"""
General description.

# include the following if the function modifies any of its arguments
Notes:
  Modifies argument in place.

Args:
  argument 1 (type): description
  # if argument 1 is a tensor, include its shape
  # e.g., argument 1 (tensor (num_rows, num_columns))
  
Returns:
  description (type)
  # if the function returns a tensor, include its shape
  # e.g. description (tensor (num_rows, num_columns))
"""
```

3) Never modify global variables. Generally, try to avoid side effects 
by writing pure functions that do not modify their arguments.

4) Use the backend functions for algebra and numeric computations. It
is probably better to write a new backend function than to use a version
from numpy.

5) Backend functions that modify their arguments should be named as
`function_name_` with a trailing underscore.

6) Use classes for objects that hold a state. 

7) If you are implementing something that doesn't hold a state, it should
be a function, not a class.

8) Use the functional style methods `apply`, `mapzip`, etc to apply functions
to iterable objecs such as lists or namedtuples (and the `grad_apply` etc 
functions to apply functions to gradient objects).
