import json
import os

# load the configuration file with the backend specification
filedir = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(filedir,"config.json"), "r") as infile:
    config = json.load(infile)

# read the processor type
PROCESSOR = config['processor']
BACKEND = config['backend']

# check validity
assert PROCESSOR in ["cpu", "gpu"], "processor must by cpu or gpu"
assert BACKEND in ["python", "pytorch"], "backend must be python or pytorch"

if PROCESSOR == "gpu":
    assert BACKEND == "pytorch", "must specify pytorch backend to use gpu"
    def test_has_cuda():
        try:
            import torch
        except ImportError:
            assert False, "must have pytorch installed to use pytorch backend"
        try:
            torch.cuda.FloatTensor()
        except Exception:
            assert False, "must have cuda enabled pytorch to use gpu"
    test_has_cuda()

print("Running paysage with the {} backend on the {}".format(BACKEND, PROCESSOR))
