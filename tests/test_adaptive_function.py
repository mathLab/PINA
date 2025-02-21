import torch
import pytest

from pina.adaptive_function import (AdaptiveReLU, AdaptiveSigmoid, AdaptiveTanh,
                            AdaptiveSiLU, AdaptiveMish, AdaptiveELU,
                            AdaptiveCELU, AdaptiveGELU, AdaptiveSoftmin,
                            AdaptiveSoftmax, AdaptiveSIREN, AdaptiveExp)


adaptive_function = (AdaptiveReLU, AdaptiveSigmoid, AdaptiveTanh,
                      AdaptiveSiLU, AdaptiveMish, AdaptiveELU,
                      AdaptiveCELU, AdaptiveGELU, AdaptiveSoftmin,
                      AdaptiveSoftmax, AdaptiveSIREN, AdaptiveExp)
x = torch.rand(10, requires_grad=True)

@pytest.mark.parametrize("Func", adaptive_function)
def test_constructor(Func):
    if Func.__name__ == 'AdaptiveExp':
        # simple
        Func()
        # setting values
        af = Func(alpha=1., beta=2.)
        assert af.alpha.requires_grad
        assert af.beta.requires_grad
        assert af.alpha == 1.
        assert af.beta == 2.
    else:
        # simple
        Func()
        # setting values
        af = Func(alpha=1., beta=2., gamma=3.)
        assert af.alpha.requires_grad
        assert af.beta.requires_grad
        assert af.gamma.requires_grad
        assert af.alpha == 1.
        assert af.beta == 2.
        assert af.gamma == 3.

    # fixed variables
    af = Func(alpha=1., beta=2., fixed=['alpha'])
    assert af.alpha.requires_grad is False
    assert af.beta.requires_grad
    assert af.alpha == 1.
    assert af.beta == 2.

    with pytest.raises(TypeError):
        Func(alpha=1., beta=2., fixed=['delta'])

    with pytest.raises(ValueError):
        Func(alpha='s')
        Func(alpha=1)

@pytest.mark.parametrize("Func", adaptive_function)      
def test_forward(Func):
    af = Func()
    af(x)

@pytest.mark.parametrize("Func", adaptive_function)
def test_backward(Func):
    af = Func()
    y = af(x)
    y.mean().backward()