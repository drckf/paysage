# Documentation for Graph (graph.py)

## class Connection
### W
```py

def W(self, trans=False)

```



Convience function to access the weight matrix.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;trans (optional; bool): transpose the weight matrix<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor ~ (number of target units, number of domain units)


### \_\_init\_\_
```py

def __init__(self, target_index, domain_index, weights)

```



Create an object to manage the connections in models.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;target_index (int): index of target layer<br />&nbsp;&nbsp;&nbsp;&nbsp;domain_index (int): index of domain_layer<br />&nbsp;&nbsp;&nbsp;&nbsp;weights (Weights): weights object with<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;weights.matrix = tensor ~ (number of target units, number of domain units)<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;Connection


### get\_config
```py

def get_config(self)

```



Return a dictionary describing the configuration of the connections.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;None<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;dict



