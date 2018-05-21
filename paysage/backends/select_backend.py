from . import read_config as config

if config.BACKEND == 'python':
    from .python_backend.matrix import *
    from .python_backend.nonlinearity import *
    from .python_backend.rand import *
    from .python_backend.typedef import *
elif config.BACKEND == 'pytorch':
    from .pytorch_backend.matrix import *
    from .pytorch_backend.nonlinearity import *
    from .pytorch_backend.rand import *
    from .pytorch_backend.typedef import *
