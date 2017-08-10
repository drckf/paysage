from . import read_config as config

# import the functions from the specified backend
if config.BACKEND == 'python':
    from .python_backend.matrix import *
    from .python_backend.nonlinearity import *
    from .python_backend.rand import *
    from .python_backend.typedef import *
elif config.BACKEND == 'pytorch':
    from .pytorch_backend.matrix import *
    from .pytorch_backend.nonlinearity import *
    from .pytorch_backend.rand import *
    from .python_backend.typedef import *

from .common import *
