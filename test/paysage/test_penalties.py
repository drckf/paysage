from paysage import penalties as pen
from paysage import backends as be

import pytest

penalty_types=[pen.trivial_penalty, pen.l2_penalty, pen.l2_norm, pen.l1_penalty,
               pen.l1_adaptive_decay_penalty_2, pen.exp_l2_penalty,
               pen.log_norm, pen.log_penalty]

def test_penalties():
    for p in penalty_types:
        penalty = p(1.0, (slice(1,100,1), slice(0,200,2)))
        t = be.rand((100,200))*2.0 - be.ones((100,200))
        v1 = penalty.value(t)
        g = penalty.grad(t)
        t -= be.EPSILON*g
        v2 = penalty.value(t)
        assert v1 >= v2, "penalty {} gradient is not working properly".format(p)

    penalty = pen.logdet_penalty(1.0, (slice(0,100,1),))
    t = be.identity(100) + be.rand((100,100))*0.2 - be.ones((100,100))*0.1
    v1 = penalty.value(t)
    g = penalty.grad(t)
    t -= be.EPSILON*g
    v2 = penalty.value(t)
    assert v1 >= v2, "logdet_penalty gradient is not working properly"

if __name__ == "__main__":
    pytest.main([__file__])
