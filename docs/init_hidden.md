# Documentation for Init_Hidden (init_hidden.py)

## functions

### hinton
```py

def hinton(batch, model)

```



Initialize the parameters of an RBM.<br /><br />Based on the method described in:<br /><br />Hinton, Geoffrey.<br />"A practical guide to training restricted Boltzmann machines."<br />Momentum 9.1 (2010): 926.<br /><br />Initalize the weights from N(0, 0.01)<br />Set hidden_bias = 0<br />Set visible_bias = inverse_mean( \< v_i \> )<br />If visible_scale: set visible_scale = \< v_i^2 \> - \< v_i \>^2<br /><br />Notes:<br /> ~ Modifies the model parameters in place.<br /><br />Args:<br /> ~ batch: A batch object that provides minibatches of data.<br /> ~ model: A model to inialize.<br /><br />Returns:<br /> ~ None

