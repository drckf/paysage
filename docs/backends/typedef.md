# Documentation for Typedef (typedef.py)

## class DoubleTensorCPU


## class DoubleTensorGPU


## class FloatTensorCPU


## class FloatTensorGPU


## class ShortTensorCPU


## class ShortTensorGPU


## class ByteTensorCPU


## class ByteTensorGPU


## class LongTensorCPU


## class LongTensorGPU


## class IntTensorCPU


## class IntTensorGPU


## class NumpyTensor
ndarray(shape, dtype=float, buffer=None, offset=0,<br />        strides=None, order=None)<br /><br />An array object represents a multidimensional, homogeneous array<br />of fixed-size items.  An associated data-type object describes the<br />format of each element in the array (its byte-order, how many bytes it<br />occupies in memory, whether it is an integer, a floating point number,<br />or something else, etc.)<br /><br />Arrays should be constructed using `array`, `zeros` or `empty` (refer<br />to the See Also section below).  The parameters given here refer to<br />a low-level method (`ndarray(...)`) for instantiating an array.<br /><br />For more information, refer to the `numpy` module and examine the<br />methods and attributes of an array.<br /><br />Parameters<br />----------<br />(for the __new__ method; see Notes below)<br /><br />shape : tuple of ints<br />    Shape of created array.<br />dtype : data-type, optional<br />    Any object that can be interpreted as a numpy data type.<br />buffer : object exposing buffer interface, optional<br />    Used to fill the array with data.<br />offset : int, optional<br />    Offset of array data in buffer.<br />strides : tuple of ints, optional<br />    Strides of data in memory.<br />order : {'C', 'F'}, optional<br />    Row-major (C-style) or column-major (Fortran-style) order.<br /><br />Attributes<br />----------<br />T : ndarray<br />    Transpose of the array.<br />data : buffer<br />    The array's elements, in memory.<br />dtype : dtype object<br />    Describes the format of the elements in the array.<br />flags : dict<br />    Dictionary containing information related to memory use, e.g.,<br />    'C_CONTIGUOUS', 'OWNDATA', 'WRITEABLE', etc.<br />flat : numpy.flatiter object<br />    Flattened version of the array as an iterator.  The iterator<br />    allows assignments, e.g., ``x.flat = 3`` (See `ndarray.flat` for<br />    assignment examples; TODO).<br />imag : ndarray<br />    Imaginary part of the array.<br />real : ndarray<br />    Real part of the array.<br />size : int<br />    Number of elements in the array.<br />itemsize : int<br />    The memory use of each array element in bytes.<br />nbytes : int<br />    The total number of bytes required to store the array data,<br />    i.e., ``itemsize * size``.<br />ndim : int<br />    The array's number of dimensions.<br />shape : tuple of ints<br />    Shape of the array.<br />strides : tuple of ints<br />    The step-size required to move from one element to the next in<br />    memory. For example, a contiguous ``(3, 4)`` array of type<br />    ``int16`` in C-order has strides ``(8, 2)``.  This implies that<br />    to move from element to element in memory requires jumps of 2 bytes.<br />    To move from row-to-row, one needs to jump 8 bytes at a time<br />    (``2 * 4``).<br />ctypes : ctypes object<br />    Class containing properties of the array needed for interaction<br />    with ctypes.<br />base : ndarray<br />    If the array is a view into another array, that array is its `base`<br />    (unless that array is also a view).  The `base` array is where the<br />    array data is actually stored.<br /><br />See Also<br />--------<br />array : Construct an array.<br />zeros : Create an array, each element of which is zero.<br />empty : Create an array, but leave its allocated memory unchanged (i.e.,<br />        it contains "garbage").<br />dtype : Create a data-type.<br /><br />Notes<br />-----<br />There are two modes of creating an array using ``__new__``:<br /><br />1. If `buffer` is None, then only `shape`, `dtype`, and `order`<br />   are used.<br />2. If `buffer` is an object exposing the buffer interface, then<br />   all keywords are interpreted.<br /><br />No ``__init__`` method is needed because the array is fully initialized<br />after the ``__new__`` method.<br /><br />Examples<br />--------<br />These examples illustrate the low-level `ndarray` constructor.  Refer<br />to the `See Also` section above for easier ways of constructing an<br />ndarray.<br /><br />First mode, `buffer` is None:<br /><br />>>> np.ndarray(shape=(2,2), dtype=float, order='F')<br />array([[ -1.13698227e+002,   4.25087011e-303],<br />       [  2.88528414e-306,   3.27025015e-309]])         #random<br /><br />Second mode:<br /><br />>>> np.ndarray((2,), buffer=np.array([1,2,3]),<br />...            offset=np.int_().itemsize,<br />...            dtype=int) # offset = 1*itemsize, i.e. skip first element<br />array([2, 3])


## class Iterable
Abstract base class for generic types.<br /><br />A generic type is typically declared by inheriting from<br />this class parameterized with one or more type variables.<br />For example, a generic mapping type might be defined as::<br /><br />  class Mapping(Generic[KT, VT]):<br />      def __getitem__(self, key: KT) -> VT:<br />          ...<br />      # Etc.<br /><br />This class can then be used as follows::<br /><br />  def lookup_name(mapping: Mapping[KT, VT], key: KT, default: VT) -> VT:<br />      try:<br />          return mapping[key]<br />      except KeyError:<br />          return default


## class Dtype


## class Tuple
Tuple type; Tuple[X, Y] is the cross-product type of X and Y.<br /><br />Example: Tuple[T1, T2] is a tuple of two elements corresponding<br />to type variables T1 and T2.  Tuple[int, float, str] is a tuple<br />of an int, a float and a string.<br /><br />To specify a variable-length tuple of homogeneous type, use Tuple[T, ...].


## class Dict
dict() -> new empty dictionary<br />dict(mapping) -> new dictionary initialized from a mapping object's<br />    (key, value) pairs<br />dict(iterable) -> new dictionary initialized as if via:<br />    d = {}<br />    for k, v in iterable:<br />        d[k] = v<br />dict(**kwargs) -> new dictionary initialized with the name=value pairs<br />    in the keyword argument list.  For example:  dict(one=1, two=2)


## class List
list() -> new empty list<br />list(iterable) -> new list initialized from iterable's items

