import torch
import pytest

from pina.model.layers.adaptive_func import AdaptiveActivationFunction

x = torch.rand(5)
torchfunc = torch.nn.Tanh()

def test_constructor():
    # simple
    AdaptiveActivationFunction(torchfunc)

    # setting values
    af = AdaptiveActivationFunction(torchfunc, alpha=1., beta=2., gamma=3.)
    assert af.alpha.requires_grad
    assert af.beta.requires_grad
    assert af.gamma.requires_grad
    assert af.alpha == 1.
    assert af.beta == 2.
    assert af.gamma == 3.

    # fixed variables
    af = AdaptiveActivationFunction(torchfunc, alpha=1., beta=2.,
                                    gamma=3., fixed=['alpha'])
    assert af.alpha.requires_grad is False
    assert af.beta.requires_grad
    assert af.gamma.requires_grad
    assert af.alpha == 1.
    assert af.beta == 2.
    assert af.gamma == 3.

    with pytest.raises(TypeError):
        AdaptiveActivationFunction(torchfunc, alpha=1., beta=2.,
                                    gamma=3., fixed=['delta'])

    with pytest.raises(ValueError):
        AdaptiveActivationFunction(torchfunc, alpha='s')
        AdaptiveActivationFunction(torchfunc, alpha=1., fixed='alpha')
        AdaptiveActivationFunction(torchfunc, alpha=1)
        
def test_forward():
    af = AdaptiveActivationFunction(torchfunc)
    af(x)

def test_backward():
    af = AdaptiveActivationFunction(torchfunc)
    y = af(x)
    y.mean().backward()