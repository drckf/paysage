import os
import pandas

from .. import layers
from .. import backends as be
from ..models.initialize import init_model as init
from . import gradient_util as gu
from . import model


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

    def gibbs_free_energy_TAP2(self, init_lr=0.1, tol=1e-4, max_iters=500):
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
            vis (batch_size, num_visible): Observed visible units.
            init_lr float: initial learning rate which is halved whenever necessary to enforce descent.
            tol float: tolerance for quitting minimization.
            max_iters: maximum gradient decsent steps

        Returns:
            tuple (visible magnetization, hidden magnetization, TAP2-approximated Gibbs free energy)
            (num_visible_neurons, num_hidden_neurons, 1)

        """

        w = self.weights[0].int_params.matrix
        a = self.layers[0].int_params.loc
        b = self.layers[1].int_params.loc

        def gamma_TAP2(m_v, m_h, w, a, b):
            m_v_quad = m_v - be.square(m_v)
            m_h_quad = m_h - be.square(m_h)
            ww = be.square(w)
            return \
                be.tsum(be.multiply(m_v, be.log(m_v)) + be.multiply(1.0 - m_v, be.log(1.0 - m_v))) + \
                be.tsum(be.multiply(m_h, be.log(m_h)) + be.multiply(1.0 - m_h, be.log(1.0 - m_h))) - \
                be.dot(a,m_v) - be.dot(b,m_h) - be.dot(m_v, be.dot(w,m_h)) - \
                0.5*be.dot(m_v_quad, be.dot(ww, m_h_quad))

        def grad_v_gamma_TAP2(m_v, m_h, w, a):
            m_h_quad = m_h - be.square(m_h)
            ww = be.square(w)
            return be.log(be.divide(1.0 - m_v, m_v)) - a - be.dot(w, m_h) - be.multiply(0.5 - m_v,be.dot(ww, m_h_quad))

        def grad_h_gamma_TAP2(m_v, m_h, w, b):
            m_v_quad = m_v - be.square(m_v)
            ww = be.square(w)
            return be.log(be.divide(1.0 - m_h, m_h)) - b - be.dot(m_v,w) - \
                   be.multiply(be.dot(m_v_quad,ww),0.5 - m_h)

        def minimize_gamma_GD(w, a, b, m_v, m_h, init_lr, tol, max_iters):
            """
            Simple gradient descent routine to minimize Gamma
            """
            eps = 1e-6
            its = 0
            lr = init_lr
            gam = gamma_TAP2(m_v, m_h, w, a, b)
            while (its < max_iters):
                its += 1
                m_v_provisional = m_v - lr*grad_v_gamma_TAP2(m_v, m_h, w, a)
                m_h_provisional = m_h - lr*grad_h_gamma_TAP2(m_v, m_h, w, b)
                #TODO: worry about how much clipping gets done here!
                #perhaps rescale to avoid clipping?
                be.clip_inplace(m_h_provisional, eps, 1.0-eps)
                be.clip_inplace(m_v_provisional, eps, 1.0-eps)
                gam_provisional = gamma_TAP2(m_v_provisional, m_h_provisional, w, a, b)
                if (gam - gam_provisional < 0):
                    lr *= 0.5
                    if (lr < 1e-10):
                        break
                elif (gam - gam_provisional < tol):
                    break
                else:
                    m_v = m_v_provisional
                    m_h = m_h_provisional
                    gam = gam_provisional
            return (m_v, m_h, gam)

        # generate random sample in domain to use as a starting location for gradient descent
        num_visible_units = be.shape(a)[0]
        num_hidden_units = be.shape(b)[0]
        m_v = 0.99 * be.float_tensor(be.rand((num_visible_units,))) + 0.005
        m_h = 0.99 * be.float_tensor(be.rand((num_hidden_units,))) + 0.005
        return minimize_gamma_GD(w, a, b, m_v, m_h, init_lr, tol, max_iters)

    def clamped_free_energy(self, v, w, a, b):
        """
        '''
        -\log \sum_h \exp{-E(v,h)}
        '''
        """
        return -be.dot(a,v) - be.tsum(be.logaddexp((b + be.dot(v,w)), be.zeros_like(b)))

    def gradient(self, vdata, vmodel):
        """
        Gradient of -\ln P(v) with respect to the weights and biases
        """

        batch_size = be.shape(vdata)[0]
        # alias weights and biases
        w = self.weights[0].int_params.matrix
        a = self.layers[0].int_params.loc
        b = self.layers[1].int_params.loc

        # compute the TAP2 approximation to the Gibbs free energy:
        (m_v_min, m_h_min, EMF) = self.gibbs_free_energy_TAP2()
        # Compute the gradients at this minimizing magnetization
        m_v_quad = m_v_min - be.square(m_v_min)
        m_h_quad = m_h_min - be.square(m_h_min)

        dw_EMF = -be.outer(m_v_min, m_h_min) - be.multiply(w, be.outer(m_v_quad, m_h_quad))
        da_EMF = -m_v_min
        db_EMF = -m_h_min

        # compute average grad_F_c over the minibatch
        #score = 0.0
        dw = be.zeros_like(w)
        da = be.zeros_like(a)
        db = be.zeros_like(b)

        for v in vdata:
            #score -= self.clamped_free_energy(v,w,a,b)
            dw += be.outer(v,be.expit(be.dot(v,w) + b))
            da += v
            db += be.expit(b + be.dot(v,w))

        grad = gu.Gradient(
            [None for l in self.layers],
            [None for w in self.weights]
        )

        grad.weights[0] = layers.IntrinsicParamsWeights(dw/batch_size + dw_EMF)
        grad.layers[0] = layers.IntrinsicParamsBernoulli(da/batch_size + da_EMF)
        grad.layers[1] = layers.IntrinsicParamsBernoulli(db/batch_size + db_EMF)

        #print(score / batch_size + EMF)
        return grad
