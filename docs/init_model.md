# Documentation for Init_Model (init_model.py)

## functions

### glorot\_normal
```py

def glorot_normal(batch, model)

```



Initialize the parameters of an RBM.<br /><br />Identical to the 'hinton' method above<br />with the variation that we initialize the weights according to<br />the prescription of Glorot and Bengio from<br /><br />"Understanding the difficulty of training deep feedforward neural networks", 2010:<br /><br />Initalize the weights from N(0, \sigma)<br />with \sigma = \sqrt(2 / (num_vis_units + num_hidden_units)).<br /><br />Set hidden_bias = 0<br />Set visible_bias = inverse_mean( \< v_i \> )<br />If visible_scale: set visible_scale = \< v_i^2 \> - \< v_i \>^2<br /><br />Notes:<br /> ~ Modifies the model parameters in place.<br /><br />Args:<br /> ~ batch: A batch object that provides minibatches of data.<br /> ~ model: A model to inialize.<br /><br />Returns:<br /> ~ None


### hinton
```py

def hinton(batch, model)

```



Initialize the parameters of an RBM.<br /><br />Based on the method described in:<br /><br />Hinton, Geoffrey.<br />"A practical guide to training restricted Boltzmann machines."<br />Momentum 9.1 (2010): 926.<br /><br />Initalize the weights from N(0, \sigma)<br />Set hidden_bias = 0<br />Set visible_bias = inverse_mean( \< v_i \> )<br />If visible_scale: set visible_scale = \< v_i^2 \> - \< v_i \>^2<br /><br />Notes:<br /> ~ Modifies the model parameters in place.<br /><br />Args:<br /> ~ batch: A batch object that provides minibatches of data.<br /> ~ model: A model to initialize.<br /><br />Returns:<br /> ~ None

