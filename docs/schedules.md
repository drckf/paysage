# Documentation for Schedules (schedules.py)

## class ExponentialDecay
Base schedule class
### \_\_init\_\_
```py

def __init__(self, initial=1.0, coefficient=0.9, value=None)

```



Exponential decay with coefficient alpha, i.e. x(t) = alpha^t.<br />Sets x(0) = 1 and uses the recursive formula x(t+1) = alpha * x(t).<br /><br />Args:<br /> ~ initial (float)<br /> ~ coefficient (float in [0,1])<br /><br />Returns:<br /> ~ ExponentialDecay


### copy
```py

def copy(self)

```



### get\_config
```py

def get_config(self)

```



### reset
```py

def reset(self)

```





## class PowerLawDecay
Base schedule class
### \_\_init\_\_
```py

def __init__(self, initial=1.0, coefficient=0.9, value=None)

```



Power law decay with coefficient alpha, i.e. x(t) = 1 / (1 + alpha * t).<br />Sets x(0) = 1 and uses the recursive formula 1/x(t+1) = alpha + 1/x(t).<br /><br />Args:<br /> ~ initial (float)<br /> ~ coefficient (float in [0,1])<br /><br />Returns:<br /> ~ PowerLawDecay


### copy
```py

def copy(self)

```



### get\_config
```py

def get_config(self)

```



### reset
```py

def reset(self)

```





## class Constant
Base schedule class
### \_\_init\_\_
```py

def __init__(self, initial=1.0, value=None)

```



Constant learning rate x(t) = x(0).<br /><br />Args:<br /> ~ initial (float)<br /><br />Returns:<br /> ~ Constant


### copy
```py

def copy(self)

```



### get\_config
```py

def get_config(self)

```



### reset
```py

def reset(self)

```





## class Schedule
Base schedule class
### copy
```py

def copy(self)

```



### get\_config
```py

def get_config(self)

```





## functions

### schedule\_from\_config
```py

def schedule_from_config(config)

```



Construct a schedule from a configuration.<br /><br />Args:<br /> ~ A dictionary configuration of the metadata.<br /><br />Returns:<br /> ~ Schedule

