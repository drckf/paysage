import json, os

filedir = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(filedir,"config.json"), "r") as infile:
    config = json.load(infile)

if config['backend'] == 'python':
    from .python_backend.matrix import *
    from .python_backend.nonlinearity import *
elif config['backend'] == 'pytorch':
    from .pytorch_backend.matrix import *
    from .pytorch_backend.nonlinearity import *
else:
    raise ValueError(
    "Unknown backend {}".format(config['backend'])
    )
