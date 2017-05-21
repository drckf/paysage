from paysage import batch
from paysage import layers
from paysage.models import model
from paysage import fit
from paysage import optimizers
from paysage import backends as be
from paysage import schedules

be.set_seed(137) # for determinism

import example_util as util

paysage_path = None

num_hidden_units = 256
batch_size = 100
learning_rate = schedules.power_law_decay(initial=0.1, coefficient=0.1)

(_, _, shuffled_filepath) = \
        util.default_paths(paysage_path)

# set up the reader to get minibatches
data = batch.HDFBatch(shuffled_filepath,
                     'train/images',
                      batch_size,
                      transform=batch.binarize_color,
                      train_fraction=0.95)

# set up the model and initialize the parameters
vis_layer = layers.BernoulliLayer(data.ncols)
hid_layer = layers.BernoulliLayer(num_hidden_units)

rbm = model.Model([vis_layer, hid_layer])
rbm.initialize(data, 'glorot_normal')

state = model.StateTAP.from_model_rand(rbm)

# test the variance
var = state.cumulants[0].variance
other_var = state.cumulants[0].mean * (1 - state.cumulants[0].mean)
assert be.allclose(var, other_var), "variance is incorrect"

lagrange = rbm.layers[0].lagrange_multiplers(state.cumulants[0])
entropy = rbm.layers[0].TAP_entropy(lagrange, state.cumulants[0])
print(entropy)


gfe = rbm.gibbs_free_energy(state)
print(gfe)

