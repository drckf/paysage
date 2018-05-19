import numpy as np

from paysage import backends as be
from paysage import preprocess as pre

import pytest

tensors = [be.rand((100, 10)) for _ in range(8)]

def compare_lists(a, b):
    return all([be.allclose(ai, bi) for ai, bi in zip(a, b)])

def test_scale():
    # test function
    result_pre = [pre.scale(tensor, 2) for tensor in tensors]
    result_ref = [0.5 * tensor for tensor in tensors]
    assert compare_lists(result_pre, result_ref)
    # test transform
    transformer = pre.Transformation(pre.scale, kwargs={'denominator': 2})
    result_pre_2 = [transformer.compute(tensor) for tensor in tensors]
    assert compare_lists(result_pre, result_pre_2)

def test_l2_normalize():
    result_pre = [be.norm(pre.l2_normalize(tensor), axis=1) for tensor in tensors]
    result_ref = [be.ones((len(tensor),)) for tensor in tensors]
    assert compare_lists(result_pre, result_ref)

def test_l1_normalize():
    result_pre = [be.tsum(pre.l1_normalize(tensor), axis=1) for tensor in tensors]
    result_ref = [be.ones((len(tensor),)) for tensor in tensors]
    assert compare_lists(result_pre, result_ref)

def test_binarize_color():
    result_pre = [pre.binarize_color(pre.scale(tensor, 1/255)) for tensor in tensors]
    result_ref = [be.float_tensor(be.tround(tensor)) for tensor in tensors]
    assert compare_lists(result_pre, result_ref)

def test_one_hot():
    categories = range(10)
    labels = be.unsqueeze(be.long_tensor(np.arange(100) // 10), 1)
    hots = pre.one_hot(labels, categories)
    hots_ref = be.zeros((len(labels), len(categories)))
    be.scatter_(hots_ref, be.long_tensor(np.arange(100) // 10), 1)
    assert be.allclose(hots, hots_ref)

def test_transformation_config():
    transformer = pre.Transformation(pre.scale, kwargs={'denominator': 2})
    transformer_result = [transformer.compute(tensor) for tensor in tensors]
    config = transformer.get_config()
    transformer_from_config = pre.Transformation.from_config(config)
    config_result = [transformer_from_config.compute(tensor) for tensor in tensors]
    assert compare_lists(transformer_result, config_result)

if __name__ == "__main__":
    pytest.main([__file__])
