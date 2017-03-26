FROM ubuntu:16.10

# System Dependencies
RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get install -y \
    libhdf5-dev \
    llvm-3.8 \
    python3.6 \
    python3-pip

# Pytorch
RUN pip3 install --upgrade pip
# RUN pip3 install -r requirements.txt 

# RUN pip3 install http://download.pytorch.org/whl/cu75/torch-0.1.10.post2-cp36-cp36m-linux_x86_64.whl 
# RUN pip3 install torchvision

# Checks
CMD echo "testing"
CMD python3 --version
