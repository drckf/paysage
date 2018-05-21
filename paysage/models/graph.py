from ..layers import weights as ww

class Connection(object):

    def __init__(self, target_index, domain_index, weights):
        """
        Create an object to manage the connections in models.

        Args:
            target_index (int): index of target layer
            domain_index (int): index of domain_layer
            weights (Weights): weights object with
                weights.matrix = tensor ~ (number of target units, number of domain units)

        Returns:
            Connection

        """
        self.target_index = target_index
        self.domain_index = domain_index
        self.weights = weights
        self.shape = self.weights.shape

    def get_config(self):
        """
        Return a dictionary describing the configuration of the connections.

        Args:
            None

        Returns:
            dict

        """
        return {"target_index": self.target_index,
                "domain_index": self.domain_index,
                "weights": self.weights.get_config()}

    @classmethod
    def from_config(cls, config):
        """
        Create a connection from a configuration dictionary.

        Args:
            config (dict)

        Returns:
            Connection

        """
        return cls(config["target_index"],
                   config["domain_index"],
                   ww.weights_from_config(config["weights"]))

    def W(self, trans=False):
        """
        Convience function to access the weight matrix.

        Args:
            trans (optional; bool): transpose the weight matrix

        Returns:
            tensor ~ (number of target units, number of domain units)

        """
        return self.weights.W(trans)
