# Paysage

Paysage is library for unsupervised learning and probabilistic generative models written in Python. The library is still in the early stages and is not yet stable, so new features will be added frequently.

Currently, paysage can be used to train things like:

* Bernoulli Restricted Boltzmann Machines
* Gaussian Restricted Boltzmann Machines
* Hopfield Models

Using advanced mean field and Markov Chain Monte Carlo methods.

## Physics-inspired machine learning
* **Better performance through better algorithms**. We are focused on making better Monte Carlo samplers, initialization methods, and optimizers that allow you to train Boltzmann machines without emptying your wallet for a new computer.
* **Stay close to Python**. Everybody loves Python, but sometimes it is too slow to get the job done. We want to minimize the amount of computation that gets shifted to the backend by targeting efforts for acceleration to the main bottlenecks in training.


## Installation:
We recommend using paysage with Anaconda3. Simply,

1. Clone this git repo
2. Move into the directory with setup.py
3. Run “pip install -e .”

Running the examples requires a file mnist.h5 containing the MNIST dataset of handwritten images. The script download_mnist.py in the mnist/ folder will fetch the file from the web.

## Using PyTorch
Paysage uses one of two backends for performing computations. By default, computations are performed using numpy/numexpr/numba on the CPU. If you have installed [PyTorch](http://pytorch.org), then you can switch to the pytorch backend by changing the setting in `paysage/backends/config.json` to `pytorch`. If you have a CUDA enabled version of pytorch, you can change the setting in `paysage/backends/config.json` from `cpu` to `gpu` to run on the GPU.

## System Dependencies

- hdf5, 1.8 required required by tables
- llvm, llvm-config required by scikit-learn

## About the name:
Boltzmann machines encode information in an ["energy landscape"](https://en.wikipedia.org/wiki/Energy_landscape) where highly probable states have low energy and lowly probable states have high energy. The name "landscape" was already taken, but the French translation "paysage" was not.
