from collections import namedtuple

from .. import layers
from .. import backends as be
from ..models.initialize import init_model as init
from . import gradient_util as gu
from . import model

class Magnetization(object):
    def __init__(self, v=None, h=None):
        self.v = v
        self.h = h

# Derives from Model object defined in hidden.py
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

    def __init__(self, layer_list, init_lr_EMF=0.1, tolerance_EMF=1e-2, max_iters_EMF=100, num_persistent_samples=0):
        """
        Create a TAP RBM model.

        Notes:
            Only 2-layer models currently supported.

        Args:
            layer_list: A list of layers objects.

            EMF computation parameters:
                init_lr float: initial learning rate which is halved whenever necessary to enforce descent.
                tol float: tolerance for quitting minimization.
                max_iters: maximum gradient decsent steps
                number of persistent magnetization parameters to keep as seeds for gradient descent.
                    0 implies we use a random seed each iteration

        Returns:
            model: A TAP RBM model.

        """
        super().__init__(layer_list)
        self.tolerance_EMF = tolerance_EMF
        self.max_iters_EMF = max_iters_EMF
        self.init_lr_EMF = init_lr_EMF
        self.persistent_samples = []
        for i in range (num_persistent_samples):
            self.persistent_samples.append(None)
        self.tap_seed = None


    def gibbs_free_energy_TAP2(self, seed=None, init_lr=0.1, tol=1e-4, max_iters=500):
        """
        Compute the Gibbs free engergy of the model according to the TAP
        expansion around infinite temperature to second order.

        If the energy is:
        '''
            E(v, h) := -\langle a,v \rangle - \langle b,h \rangle - \langle v,W \cdot h \rangle, with state probability distribution:
            P(v,h)  := 1/\sum_{v,h} \exp{-E(v,h)} * \exp{-E(v,h)}, and the marginal
            P(v)    := \sum_{h} P(v,h)
        '''
        Then the Gibbs free energy is:
        '''
            F(v) := -log\sum_{v,h} \exp{-E(v,h)}
        '''
        We add an auxiliary local field q, and introduce the inverse temperature variable \beta to define
        '''
            \beta F(v;q) := -log\sum_{v,h} \exp{-\beta E(v,h) + \beta \langle q, v \rangle}
        '''
        Let \Gamma(m) be the Legendre transform of F(v;q) as a function of q
        The TAP formula is Taylor series of \Gamma in \beta, around \beta=0.
        Setting \beta=1 and regarding the first two terms of the series as an approximation of \Gamma[m],
        we can minimize \Gamma in m to obtain an approximation of F(v;q=0) = F(v)

        This implementation uses gradient descent from a random starting location to minimize the function

        Args:
            seed 'None' or Magnetization: initial seed for the minimization routine.
                                          Chosing 'None' will result in a random seed
            init_lr float: initial learning rate which is halved whenever necessary to enforce descent.
            tol float: tolerance for quitting minimization.
            max_iters: maximum gradient decsent steps

        Returns:
            tuple (magnetization, TAP2-approximated Gibbs free energy)
                  (Magnetization, float)

        """

        w = self.weights[0].int_params.matrix
        a = self.layers[0].int_params.loc
        b = self.layers[1].int_params.loc

        def grad_v_gamma_TAP2(m, w, a):
            m_h_quad = m.h - be.square(m.h)
            ww = be.square(w)
            return be.log(be.divide(1.0 - m.v, m.v)) - a - be.dot(w, m.h) - be.multiply(0.5 - m.v,be.dot(ww, m_h_quad))

        def grad_h_gamma_TAP2(m, w, b):
            m_v_quad = m.v - be.square(m.v)
            ww = be.square(w)
            return be.log(be.divide(1.0 - m.h, m.h)) - b - be.dot(m.v,w) - \
                   be.multiply(be.dot(m_v_quad,ww),0.5 - m.h)

        def minimize_gamma_GD(w, a, b, m, init_lr, tol, max_iters):
            """
            Simple gradient descent routine to minimize Gamma

            Warning: this function modifies seed to avoid an extra copy and allocation

            """
            eps = 1e-6
            its = 0
            lr = init_lr
            gam = self.gamma_TAP2(m, w, a, b)

            while (its < max_iters):
                its += 1
                m_provisional = Magnetization(m.v - lr*grad_v_gamma_TAP2(m, w, a),
                                              m.h - lr*grad_h_gamma_TAP2(m, w, b))
                # Warning: in general a lot of clipping gets done here
                be.clip_inplace(m_provisional.v, eps, 1.0-eps)
                be.clip_inplace(m_provisional.h, eps, 1.0-eps)

                gam_provisional = self.gamma_TAP2(m_provisional, w, a, b)
                if (gam - gam_provisional < 0):
                    lr *= 0.5
                    if (lr < 1e-10):
                        break
                elif (gam - gam_provisional < tol):
                    break
                else:
                    m = m_provisional
                    gam = gam_provisional

            return (m, gam)

        def minimize_gamma_constraint_sat(w, a, b, m, tol, max_iters, interpolation_factor=0.9):
            """
            Minimize Gamma via repeated application of the self-consistent constraint

            Warning: this function modifies seed to avoid an extra copy and allocation

            """
            its = 0
            gam = self.gamma_TAP2(m, w, a, b)
            ww = be.multiply(w,w)
            while (its < max_iters):
                its += 1
                m_v_quad = m.v - be.multiply(m.v,m.v)
                m_h_provisional = be.expit(b + be.dot(m.v,w) - be.dot(m_v_quad, be.dot(ww, m.h - 0.5)))
                m.h *= (1.0 - interpolation_factor)
                m.h += interpolation_factor * m_h_provisional

                m_h_quad = m.h - be.multiply(m.h,m.h)
                m_v_provisional = be.expit(a + be.dot(w,m.h)-be.dot(m.v - 0.5, be.dot(ww, m_h_quad)))
                m.v *= (1.0 - interpolation_factor)
                m.v += interpolation_factor * m_v_provisional

                gam_update = self.gamma_TAP2(m, w, a, b)
                if (abs(gam_update - gam) < tol):
                    break
                gam = gam_update

            return (m, gam)


        # generate random sample in domain to use as a starting location for gradient descent
        if seed==None :
            num_visible_units = be.shape(a)[0]
            num_hidden_units = be.shape(b)[0]
            seed = Magnetization(
                0.99 * be.float_tensor(be.rand((num_visible_units,))) + 0.005,
                0.99 * be.float_tensor(be.rand((num_hidden_units,))) + 0.005
            )

        return minimize_gamma_GD(w, a, b, seed, init_lr, tol, max_iters)
        #return minimize_gamma_constraint_sat(w, a, b, seed, tol, max_iters)

    # The Legendre transform of F(v;q) as a function of q according to TAP expansion 2 terms
    def gamma_TAP2(self, m, w, a, b):
        m_v_quad = m.v - be.square(m.v)
        m_h_quad = m.h - be.square(m.h)
        ww = be.square(w)
        return \
            be.tsum(be.multiply(m.v, be.log(m.v)) + be.multiply(1.0 - m.v, be.log(1.0 - m.v))) + \
            be.tsum(be.multiply(m.h, be.log(m.h)) + be.multiply(1.0 - m.h, be.log(1.0 - m.h))) - \
            be.dot(a,m.v) - be.dot(b,m.h) - be.dot(m.v, be.dot(w,m.h)) - \
            0.5*be.dot(m_v_quad, be.dot(ww, m_h_quad))
    def marginal_free_energy(self, v, w, a, b):
        """
        '''
        -\log \sum_h \exp{-E(v,h)}
        '''
        """
        return -be.dot(a,v) - be.tsum(be.logaddexp((b + be.dot(v,w)), be.zeros_like(b)))

    def gradient(self, data_state, model_state):
        """
        Gradient of -\ln P(v) with respect to the weights and biases
        """

        batch_size = be.shape(data_state.units[0])[0]
        # alias weights and biases
        w = self.weights[0].int_params.matrix
        a = self.layers[0].int_params.loc
        b = self.layers[1].int_params.loc

        # compute the TAP2 approximation to the Gibbs free energy:
        EMF = 1e6
        m = None
        if len(self.persistent_samples) == 0: # random seed
                (m,EMF) = self.gibbs_free_energy_TAP2(m,
                                                      self.init_lr_EMF,
                                                      self.tolerance_EMF,
                                                      self.max_iters_EMF)
        else:
            best_EMF = 1e7
            for s in range(len(self.persistent_samples)): # persistent seeds
                (self.persistent_samples[s],EMF) = self.gibbs_free_energy_TAP2(self.persistent_samples[s],
                                                                               self.init_lr_EMF,
                                                                               self.tolerance_EMF,
                                                                               self.max_iters_EMF)
                if EMF < best_EMF:
                    best_EMF = EMF
                    m = self.persistent_samples[s]

        # Compute the gradients at this minimizing magnetization
        m_v_quad = m.v - be.square(m.v)
        m_h_quad = m.h - be.square(m.h)

        dw_EMF = -be.outer(m.v, m.h) - be.multiply(w, be.outer(m_v_quad, m_h_quad))
        da_EMF = -m.v
        db_EMF = -m.h

        # compute average grad_F_marginal over the minibatch
        intermediate = be.expit(be.dot(data_state.units[0],w) + b)

        da = be.mean(data_state.units[0], axis=0)
        db = be.mean(intermediate, axis=0)
        # This is the same as \sum_{i} data_state.units[0][i] \outer intermediate[i]
        # TODO: is this efficient?
        dw = be.dot(be.transpose(data_state.units[0]), intermediate) / batch_size

        grad = gu.Gradient(
            [None for l in self.layers],
            [None for w in self.weights]
        )

        grad.weights[0] = layers.IntrinsicParamsWeights(dw + dw_EMF)
        grad.layers[0] = layers.IntrinsicParamsBernoulli(da + da_EMF)
        grad.layers[1] = layers.IntrinsicParamsBernoulli(db + db_EMF)

        #print(score / batch_size + EMF)
        return grad
