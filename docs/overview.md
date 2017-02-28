# Neural networks and Boltzmann machines

There are two general classes of models in machine learning: discriminative
and generative. Discriminative models can be trained to *describe* data --
e.g., to label the objects in a picture. Generative models can be trained
to *simulate* data (and, also, to describe it).

[Boltzmann machines](https://en.wikipedia.org/wiki/Boltzmann_machine) are
a type of stochastic neural network that make excellent generative models.
A Boltzmann machine represents a probability distribution using a simple
function that can be trained from data by maximizing the log-likelihood
using gradient ascent. Computing the gradient is challenging, however,
because one has to compute averages with respect to the model distribution.

Paysage provides tools for training Boltzmann machines through
1) fast approximate inference algorithms (called "initialization methods")
and
2) inference based on sampling from the model distribution using sequential
Monte Carlo methods.

## Visible Boltzmann machines

A visible neuron describes one dimension of the input, such as the pixels
in an image or the letters in a word. A visible Boltzmann machine describes
interactions between visible neurons. The probability distribution P(v) is
determined from an energy function E(v) by P(v) = exp(-E(v))/Z where

E(v) = -sum_i a_i(v_i) - sum_{i<j} W_{ij} v_i v_j

and Z is a normalizing constant. Here, a_i(v_i) is a function and W_{ij} is
a parameter that determines the interaction between neurons i and j.

## Boltzmann machines with a single hidden layer

A hidden neuron captures an unobserved latent variable that controls the
interactions between visible neurons. The joint probability distribuiton
P(v, h) is determined from an energy function E(v, h) by
P(v, h) = exp(-E(v, h ))/Z where

E(v, h) = -sum_i a_i(v_i) - sum_j b_j(h_j) - \sum_{ij} W_{ij} v_i h_j

and Z is a normalizing constant. Here, a_i(v_i) and b_j(h_j) are functions and
W_{ij} is a parameter that determines the interaction between visible neuron i
and hidden neuron j.

# The structure of paysage
