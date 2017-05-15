from collections import namedtuple
from cytoolz import partial
from copy import deepcopy

from .. import layers
from .. import backends as be
from ..models.initialize import init_model as init
from . import gradient_util as gu
from . import model

# Derives from Model object defined in model.py
class TAP_rbm(model.Model):
    """
    RBM with TAP formula-based gradient which supports deterministic training

    Example usage:
    '''
    vis = BernoulliLayer(nvis)
    hid = BernoulliLayer(nhid)
    rbm = TAP_rbm([vis, hid])
    '''

    """

    def __init__(self, layer_list, terms=2, init_lr_EMF=0.1, tolerance_EMF=1e-7,
                 max_iters_EMF=50, num_random_samples=1, num_persistent_samples=0):
        """
        Create a TAP RBM model.

        Notes:
            Only 2-layer models currently supported.

        Args:
            layer_list: A list of layers objects.
            terms: number of terms to use in the TAP expansion
            #TODO: deprecate this attribute when
            we turn tap training into a method and use tap1,tap2,tap3 as methods

            EMF computation parameters:
                init_lr float: initial learning rate which is halved whenever necessary to enforce descent.
                tol float: tolerance for quitting minimization.
                max_iters: maximum gradient decsent steps
                num_random_samples: number of Gibbs FE seeds to start from random
                num_persistent_samples: number of persistent magnetization parameters to keep as seeds
                    for Gibbs FE estimation.

        Returns:
            model: A TAP RBM model.

        """
        super().__init__(layer_list)
        self.tolerance_EMF = tolerance_EMF
        self.max_iters_EMF = max_iters_EMF
        self.init_lr_EMF = init_lr_EMF
        self.num_random_samples = num_random_samples
        self.persistent_samples = []
        for i in range (num_persistent_samples):
            self.persistent_samples.append(None)
        self.tap_seed = None

        if terms not in [1, 2, 3]:
            raise ValueError("Must specify one, two, or three terms in TAP expansion training method")
        self.terms = terms

        if num_random_samples + num_persistent_samples <= 0:
            raise ValueError("Must specify at least one random or persistent sample for Gibbs FE seeding")


    def TAP_free_energy(self, seed=None, init_lr=0.1, tol=1e-7, max_iters=50, terms=2, method='gd'):
        """
        Compute the Helmholtz free engergy of the model according to the TAP
        expansion around infinite temperature to second order.

        If the energy is,
        '''
            E(v, h) := -\langle a,v \rangle - \langle b,h \rangle - \langle v,W \cdot h \rangle,
        '''
        with Boltzmann probability distribution,
        '''
            P(v,h)  := 1/\sum_{v,h} \exp{-E(v,h)} * \exp{-E(v,h)},
        '''
        and the marginal,
        '''
            P(v)    := \sum_{h} P(v,h),
        '''
        then the Helmholtz free energy is,
        '''
            F(v) := -log\sum_{v,h} \exp{-E(v,h)}.
        '''
        We add an auxiliary local field q, and introduce the inverse temperature variable \beta to define
        '''
            \beta F(v;q) := -log\sum_{v,h} \exp{-\beta E(v,h) + \beta \langle q, v \rangle}
        '''
        Let \Gamma(m) be the Legendre transform of F(v;q) as a function of q, the Gibbs free energy.
        The TAP formula is Taylor series of \Gamma in \beta, around \beta=0.
        Setting \beta=1 and regarding the first two terms of the series as an approximation of \Gamma[m],
        we can minimize \Gamma in m to obtain an approximation of F(v;q=0) = F(v)

        This implementation uses gradient descent from a random starting location to minimize the function

        Args:
            seed 'None' or Magnetization: initial seed for the minimization routine.
                                          Chosing 'None' will result in a random seed
            init_lr float: initial learning rate which is halved whenever necessary to enforce descent.
            tol float: tolerance for quitting minimization.
            max_iters: maximum gradient decsent steps.
            terms: number of terms to use (1, 2, or 3 allowed)
            method: one of 'gd' or 'constraint' picking which Gibbs FE minimization method to use.

        Returns:
            tuple (magnetization, TAP-approximated Helmholtz free energy)
                  (Magnetization, float)

        """
        if terms not in [1, 2, 3]:
            raise ValueError("Must specify one, two, or three terms in TAP expansion training method")

        if method not in ['gd', 'constraint']:
            raise ValueError("Must specify a valid method for minimizing the Gibbs free energy")

        def minimize_gibbs_free_energy_GD(m=None, init_lr=0.01, tol=1e-6, max_iters=1, terms=2):
            """
            Simple gradient descent routine to minimize gibbs_FE

            """
            mag = deepcopy(m)
            eps = 1e-6
            its = 0

            gam = self.gibbs_free_energy(mag)
            lr = init_lr
            clip_ = partial(be.clip_inplace, a_min=eps, a_max=1.0-eps)
            lr_ = partial(be.tmul_, be.float_scalar(lr))
            #print(gam)
            while (its < max_iters):
                its += 1
                grad = self.grad_magnetization_GFE(mag)
                for g in grad:
                    be.apply_(lr_, g)
                m_provisional = [be.mapzip(be.subtract, grad[l], mag[l]) for l in range(self.num_layers)]

                # Warning: in general a lot of clipping gets done here
                for m_l in m_provisional:
                    be.apply_(clip_, m_l)

                #HACK:
                for l in range(self.num_layers):
                    m_provisional[l].c[:] = m_provisional[l].a - be.square(m_provisional[l].a)

                gam_provisional = self.gibbs_free_energy(m_provisional)
                if (gam - gam_provisional < 0):
                    lr *= 0.5
                    lr_ = partial(be.tmul_, be.float_scalar(lr))
                    #print("decreased lr" + str(its))
                    if (lr < 1e-10):
                        #print("tol reached on iter" + str(its))
                        break
                elif (gam - gam_provisional < tol):
                    break
                else:
                    #print(gam - gam_provisional)
                    mag = m_provisional
                    gam = gam_provisional

            return (mag, gam)

        # generate random sample in domain to use as a starting location for gradient descent
        if seed==None :
            seed = [be.apply(be.rand_like, lay.magnetization) for lay in self.layers]
            clip_ = partial(be.clip_inplace, a_min=0.005, a_max=0.995)
            for m in seed:
                be.apply_(clip_, m)


        #TODO: remove special constraint for Bernoulli case
        for mag in seed:
            mag.c[:] = mag.a - be.square(mag.a)

        if method == 'gd':
            return minimize_gibbs_free_energy_GD(seed, init_lr, tol, max_iters, terms=terms)
        elif method == 'constraint':
            assert False, \
                   "Constraint satisfaction is not currently supported"
            return minimize_gibbs_free_energy_GD(seed, init_lr, tol, max_iters, terms=terms)

    def grad_TAP_free_energy(self, num_r, num_p):
        """
        Compute the gradient of the Helmholtz free engergy of the model according to the TAP
        expansion around infinite temperature.

        This function will use the class members which specify the parameters for the
        Gibbs FE minimization.
        The gradients are taken as the average over the gradients computed at each of the
        minimial magnetizations for the Gibbs FE.

        Args:
            num_r: (int>=0) number of random seeds to use for Gibbs FE minimization
            num_p: (int>=0) number of persistent seeds to use for Gibbs FE minimization
        Returns:
            namedtuple: Gradient: containing gradients of the model parameters.

        """

        # compute the TAP approximation to the Helmholtz free energy:
        grad_EMF = gu.Gradient(
            [be.apply(be.zeros_like, lay.params) for lay in self.layers],
            [be.apply(be.zeros_like, way.params) for way in self.weights]
        )

        # compute minimizing magnetizations from random initializations
        for s in range(num_r):
            (mag,EMF) = self.TAP_free_energy(None,
                                                   self.init_lr_EMF,
                                                   self.tolerance_EMF,
                                                   self.max_iters_EMF,
                                                   self.terms)
            # Compute the gradients at this minimizing magnetization
            grad_gfe = self.grad_gibbs_free_energy(mag)
            def accum_(x,y): x[:] = be.add(x,y)
            gu.grad_mapzip_(accum_, grad_EMF, grad_gfe)

        # compute minimizing magnetizations from seeded initializations
        for s in range(num_p): # persistent seeds
            (self.persistent_samples[s],EMF) = \
             self.TAP_free_energy(self.persistent_samples[s],
                                        self.init_lr_EMF,
                                        self.tolerance_EMF,
                                        self.max_iters_EMF,
                                        self.terms)
            # Compute the gradients at this minimizing magnetization
            grad_gfe = self.grad_gibbs_free_energy(self.persistent_samples[s])
            def accum_(x,y): x[:] = be.add(x,y)
            gu.grad_mapzip_(accum_, grad_EMF, grad_gfe)

        # average
        scale = partial(be.tmul_, be.float_scalar(1/(num_p + num_r)))
        gu.grad_apply_(scale, grad_EMF)

        return grad_EMF

    def TAP_gradient(self, data_state, model_state):
        """
        Gradient of -\ln P(v) with respect to the model parameters

        Args:
            data_state (State object): The observed visible units and sampled hidden units.
            model_state (State objects): The visible and hidden units sampled from the model.

        Returns:
            namedtuple: Gradient: containing gradients of the model parameters.

        """
        # compute average grad_F_marginal over the minibatch
        grad_MFE = self.grad_marginal_free_energy(data_state)
        # compute the gradient of the Helmholtz FE via TAP_gradient
        grad_HFE = self.grad_TAP_free_energy(
                        num_r=self.num_random_samples, num_p=len(self.persistent_samples))
        return gu.grad_mapzip(be.subtract, grad_MFE, grad_HFE)

