# Documentation for Schedules (schedules.py)

## class ExponentialDecay
Base schedule class
### \_\_init\_\_
```py

def __init__(self, initial=1.0, coefficient=0.9, value=None)

```



Exponential decay with coefficient alpha, i.e. x(t) = alpha^t.<br />Sets x(0) = 1 and uses the recursive formula x(t+1) = alpha * x(t).<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;initial (float)<br />&nbsp;&nbsp;&nbsp;&nbsp;coefficient (float in [0,1])<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;ExponentialDecay


### copy
```py

def copy(self)

```



Copy a schedule.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;None<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;Schedule


### get\_config
```py

def get_config(self)

```



Get a configuration dictionary for the schedule.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;None<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;dict


### reset
```py

def reset(self)

```



Reset the value of the schedule to the initial value.<br /><br />Notes:<br />&nbsp;&nbsp;&nbsp;&nbsp;Modifies the value attribute in place!<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;None<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;None


### set\_value
```py

def set_value(self, value)

```



Set the value of the schedule to the given value.<br /><br />Notes:<br />&nbsp;&nbsp;&nbsp;&nbsp;Modifies the value attribute in place!<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;None<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;None




## class PowerLawDecay
Base schedule class
### \_\_init\_\_
```py

def __init__(self, initial=1.0, coefficient=0.9, value=None)

```



Power law decay with coefficient alpha, i.e. x(t) = 1 / (1 + alpha * t).<br />Sets x(0) = 1 and uses the recursive formula 1/x(t+1) = alpha + 1/x(t).<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;initial (float)<br />&nbsp;&nbsp;&nbsp;&nbsp;coefficient (float in [0,1])<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;PowerLawDecay


### copy
```py

def copy(self)

```



Copy a schedule.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;None<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;Schedule


### get\_config
```py

def get_config(self)

```



Get a configuration dictionary for the schedule.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;None<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;dict


### reset
```py

def reset(self)

```



Reset the value of the schedule to the initial value.<br /><br />Notes:<br />&nbsp;&nbsp;&nbsp;&nbsp;Modifies the value attribute in place!<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;None<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;None


### set\_value
```py

def set_value(self, value)

```



Set the value of the schedule to the given value.<br /><br />Notes:<br />&nbsp;&nbsp;&nbsp;&nbsp;Modifies the value attribute in place!<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;None<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;None




## class Constant
Base schedule class
### \_\_init\_\_
```py

def __init__(self, initial=1.0, value=None)

```



Constant learning rate x(t) = x(0).<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;initial (float)<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;Constant


### copy
```py

def copy(self)

```



Copy a schedule.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;None<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;Schedule


### get\_config
```py

def get_config(self)

```



Get a configuration dictionary for the schedule.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;None<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;dict


### reset
```py

def reset(self)

```



Reset the value of the schedule to the initial value.<br /><br />Notes:<br />&nbsp;&nbsp;&nbsp;&nbsp;Modifies the value attribute in place!<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;None<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;None


### set\_value
```py

def set_value(self, value)

```



Set the value of the schedule to the given value.<br /><br />Notes:<br />&nbsp;&nbsp;&nbsp;&nbsp;Modifies the value attribute in place!<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;None<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;None




## class Schedule
Base schedule class
### copy
```py

def copy(self)

```



Copy a schedule.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;None<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;Schedule


### get\_config
```py

def get_config(self)

```



Get a configuration dictionary for the schedule.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;None<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;dict




## class Linear
Base schedule class
### \_\_init\_\_
```py

def __init__(self, initial=1.0, delta=0.0, value=None, minval=0.0, maxval=1.0)

```



Linear schedule x(t) = x(0) - delta t.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;initial (float)<br />&nbsp;&nbsp;&nbsp;&nbsp;delta (float)<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;Linear


### copy
```py

def copy(self)

```



Copy a schedule.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;None<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;Schedule


### get\_config
```py

def get_config(self)

```



Get a configuration dictionary for the schedule.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;None<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;dict


### reset
```py

def reset(self)

```



Reset the value of the schedule to the initial value.<br /><br />Notes:<br />&nbsp;&nbsp;&nbsp;&nbsp;Modifies the value attribute in place!<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;None<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;None


### set\_value
```py

def set_value(self, value)

```



Set the value of the schedule to the given value.<br /><br />Notes:<br />&nbsp;&nbsp;&nbsp;&nbsp;Modifies the value attribute in place!<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;None<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;None




## class Step
Base schedule class
### \_\_init\_\_
```py

def __init__(self, initial=1.0, final=0.0, steps=1, value=None)

```



Step function schedule:<br />&nbsp;&nbsp;&nbsp;&nbsp;x(t) = initial if t < steps<br />&nbsp;&nbsp;&nbsp;&nbsp;x(t) = final if t >= steps<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;initial (float)<br />&nbsp;&nbsp;&nbsp;&nbsp;delta (float)<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;Linear


### copy
```py

def copy(self)

```



Copy a schedule.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;None<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;Schedule


### get\_config
```py

def get_config(self)

```



Get a configuration dictionary for the schedule.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;None<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;dict


### reset
```py

def reset(self)

```



Reset the value of the schedule to the initial value.<br /><br />Notes:<br />&nbsp;&nbsp;&nbsp;&nbsp;Modifies the value attribute in place!<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;None<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;None


### set\_value
```py

def set_value(self, value)

```



Set the value of the schedule to the given value.<br /><br />Notes:<br />&nbsp;&nbsp;&nbsp;&nbsp;Modifies the value attribute in place!<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;None<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;None




## functions

### schedule\_from\_config
```py

def schedule_from_config(config)

```



Construct a schedule from a configuration.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;A dictionary configuration of the metadata.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;Schedule

