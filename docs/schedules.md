# Documentation for Schedules (schedules.py)

## functions

### constant
```py

def constant(initial=1.0)

```



Constant i.e. x(t) = x.<br /><br />Args:<br /> ~ initial (float)<br /><br />Returns:<br /> ~ generator


### exponential\_decay
```py

def exponential_decay(initial=1.0, coefficient=0.9)

```



Exponential decay with coefficient alpha, i.e. x(t) = alpha^t.<br />Sets x(0) = 1 and uses the recursive formula x(t+1) = alpha * x(t).<br /><br />Args:<br /> ~ initial (float)<br /> ~ coefficient (float in [0,1])<br /><br />Returns:<br /> ~ generator


### power\_law\_decay
```py

def power_law_decay(initial=1.0, coefficient=0.1)

```



Power law decay with coefficient alpha, i.e. x(t) = 1 / (1 + alpha * t).<br />Sets x(0) = 1 and uses the recursive formula 1/x(t+1) = alpha + 1/x(t).<br /><br />Args:<br /> ~ initial (float)<br /> ~ coefficient (float in [0,1])<br /><br />Returns:<br /> ~ generator

