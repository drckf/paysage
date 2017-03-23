init_hidden
## functions

### hinton
```py

def hinton(batch, model)

```



Hinton says to initalize the weights from N(0, 0.01)
hidden_bias = 0
visible_bias = inverse_mean( \< v_i \> )
if visible_scale:
    visible_scale = \< v_i^2 \> - \< v_i \>^2

Hinton, Geoffrey. "A practical guide to training restricted Boltzmann machines." Momentum 9.1 (2010): 926.

