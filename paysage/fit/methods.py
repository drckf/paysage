from ..models.state import State


def contrastive_divergence(vdata, model, positive_phase, negative_phase):
    """
    Compute an approximation to the likelihood gradient using the CD-k
    algorithm for approximate maximum likelihood inference.

    Hinton, Geoffrey E.
    "Training products of experts by minimizing contrastive divergence."
    Neural computation 14.8 (2002): 1771-1800.

    Carreira-Perpinan, Miguel A., and Geoffrey Hinton.
    "On Contrastive Divergence Learning."
    AISTATS. Vol. 10. 2005.

    Notes:
        Modifies the state of the sampler.
        Modifies the sampling attributes of the model

    Args:
        vdata (tensor): observed visible units
        model: a model object
        positive_phase: a sampler object
        negative_phase: a sampler object

    Returns:
        gradient

    """
    target_layer = model.num_layers - 1

    # compute the update of the positive phase
    data_state = State.from_visible(vdata, model)
    positive_phase.set_state(data_state)
    positive_phase.update_state()
    grad_data_state = positive_phase.state_for_grad(target_layer)

    # CD resets the sampler from the visible data at each iteration
    model_state = State.from_visible(vdata, model)
    negative_phase.set_state(model_state)
    negative_phase.update_state()
    grad_model_state = negative_phase.state_for_grad(target_layer)

    # compute the gradient
    return model.gradient(grad_data_state, grad_model_state)

# alias
cd = contrastive_divergence


def persistent_contrastive_divergence(vdata, model, positive_phase, negative_phase):
    """
    PCD-k algorithm for approximate maximum likelihood inference.

    Tieleman, Tijmen.
    "Training restricted Boltzmann machines using approximations to the
    likelihood gradient."
    Proceedings of the 25th international conference on Machine learning.
    ACM, 2008.

    Notes:
        Modifies the state of the sampler.
        Modifies the sampling attributes of the model

    Args:
        vdata (List[tensor]): observed visible units
        model: a model object
        positive_phase: a sampler object
        negative_phase: a sampler object

    Returns:
        gradient

    """
    target_layer = model.num_layers - 1

    # compute the update of the positive phase
    data_state = State.from_visible(vdata, model)
    positive_phase.set_state(data_state)
    positive_phase.update_state()
    grad_data_state = positive_phase.state_for_grad(target_layer)

    # PCD persists the state of the sampler from the previous iteration
    negative_phase.update_state()
    grad_model_state = negative_phase.state_for_grad(target_layer)

    return model.gradient(grad_data_state, grad_model_state)

# alias
pcd = persistent_contrastive_divergence


class TAP(object):
    def __init__(self, use_GD=True, init_lr=0.1, tolerance=0.01, max_iters=25,
                 ratchet=False, decrease_on_neg=0.9, mean_weight=0.9,
                 mean_square_weight=0.999):
        #TODO: Fill in this docstring to explain what these arguments mean
        """

        """
        self.use_GD = use_GD
        self.init_lr = init_lr
        self.tolerance = tolerance
        self.max_iters = max_iters
        self.decrease_on_neg = decrease_on_neg
        self.ratchet = ratchet
        self.mean_weight = mean_weight
        self.mean_square_weight = mean_square_weight

    def tap_update(self, vdata, model, positive_phase, negative_phase=None):
        """
        Compute the gradient using the Thouless-Anderson-Palmer (TAP)
        mean field approximation.

        Modifications on the methods in

        Eric W Tramel, Marylou Gabrie, Andre Manoel, Francesco Caltagirone,
        and Florent Krzakala
        "A Deterministic and Generalized Framework for Unsupervised Learning
        with Restricted Boltzmann Machines"

        Args:
            vdata (tensor): observed visible units
            model (BoltzmannMachine): model to train
            positive_phase (Sampler): postive phase data sampler
            negative_phase (Sampler): unused

        Returns:
            gradient object

        """
        # compute the positive phase
        target_layer = model.num_layers - 1
        data_state = State.from_visible(vdata, model)
        positive_phase.set_state(data_state)
        positive_phase.update_state()

        grad_data_state = positive_phase.state_for_grad(target_layer)

        return model.TAP_gradient(grad_data_state, self.use_GD,
                                  self.init_lr, self.tolerance,
                                  self.max_iters, self.ratchet, self.decrease_on_neg,
                                  self.mean_weight, self.mean_square_weight)
