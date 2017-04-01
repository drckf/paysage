FROM avctrh/pytorch

# Needed to avoid debconf Display errors
ENV DEBIAN_FRONTEND noninteractive

# System Dependencies
RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get install -y \
    libhdf5-dev \
    llvm-3.8
ENV LLVM_CONFIG /usr/lib/llvm-3.8/bin/llvm-config

# Switch to python 3.6
RUN conda install -y python=3.6
RUN alias python=/opt/conda/bin/python3.6
RUN alias pip=/opt/conda/bin/pip

# Add repo to container
ADD . /opt/paysage

# Install dependencies
RUN pip install --upgrade pip
RUN pip install -r /opt/paysage/requirements.txt

# Install pytorch
# (torch wheel already included in pytorch docker image)
# RUN pip install http://download.pytorch.org/whl/cu75/torch-0.1.10.post2-cp35-cp35m-linux_x86_64.whl
RUN pip install torchvision

# Install paysage
RUN pip install -e /opt/paysage/

# Download MNIST
RUN python /opt/paysage/mnist/download_mnist.py

# Test
CMD pytest /opt/paysage/test/test_backends.py
