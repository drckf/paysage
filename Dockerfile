FROM kaixhin/cuda-torch

# System Dependencies
RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get install -y \
    libhdf5-dev \
    llvm-3.8 \
    python3.6

# Pytorch
RUN pip install http://download.pytorch.org/whl/cu75/torch-0.1.10.post2-cp36-cp36m-linux_x86_64.whl 
RUN pip install torchvision

# Checks
CMD echo "testing"
CMD python3 --version
