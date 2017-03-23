# Documentation for Init_Hidden (init_hidden.py)

## functions

### hinton
```py

def hinton(batch, model)

```



Hinton says to initalize the weights from N(0, 0.01)<br />hidden_bias = 0<br />visible_bias = inverse_mean( \< v_i \> )<br />if visible_scale:<br /> ~ visible_scale = \< v_i^2 \> - \< v_i \>^2<br /><br />Hinton, Geoffrey. "A practical guide to training restricted Boltzmann machines." Momentum 9.1 (2010): 926.

