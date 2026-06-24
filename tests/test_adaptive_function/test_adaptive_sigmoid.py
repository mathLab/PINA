import torch
import pytest
from pina import LabelTensor
from pina.adaptive_function import AdaptiveSigmoid


x = LabelTensor(torch.rand(10, 1), labels=["x"])
fixed_variables = [None, "alpha", ["beta", "gamma"], ["alpha", "beta", "gamma"]]
param_names = ["alpha", "beta", "gamma"]


@pytest.mark.parametrize("fixed", fixed_variables)
@pytest.mark.parametrize("alpha", [1, 2.5])
@pytest.mark.parametrize("beta", [1, 2.5])
@pytest.mark.parametrize("gamma", [1, 2.5])
def test_constructor(alpha, beta, gamma, fixed):

    # Construct the adaptive activation function with the specified parameters
    activation = AdaptiveSigmoid(
        alpha=alpha, beta=beta, gamma=gamma, fixed=fixed
    )

    # Build expected values for parameters
    expected = {"alpha": alpha, "beta": beta, "gamma": gamma}
    if fixed is None:
        fixed_set = set()
    else:
        fixed_set = {fixed} if isinstance(fixed, str) else set(fixed)

    # Verify fixed parameters consistency
    for v in fixed_set:
        param = getattr(activation, v)
        assert param.requires_grad is False
        assert param.item() == expected[v]

    # Verify trainable parameters consistency
    for v in param_names:
        if v not in fixed_set:
            param = getattr(activation, v)
            assert param.requires_grad is True
            assert param.item() == expected[v]

    # Should fail if alpha is not a number
    with pytest.raises(ValueError):
        AdaptiveSigmoid(alpha="s")

    # Should fail if beta is not a number
    with pytest.raises(ValueError):
        AdaptiveSigmoid(beta="s")

    # Should fail if gamma is not a number
    with pytest.raises(ValueError):
        AdaptiveSigmoid(gamma="s")

    # Should fail if fixed is not a string or list of strings
    with pytest.raises(ValueError):
        AdaptiveSigmoid(fixed=123)

    # Should fail if fixed contains invalid parameter names
    with pytest.raises(ValueError):
        AdaptiveSigmoid(fixed="delta")


@pytest.mark.parametrize("fixed", fixed_variables)
@pytest.mark.parametrize("alpha", [1, 2.5])
@pytest.mark.parametrize("beta", [1, 2.5])
@pytest.mark.parametrize("gamma", [1, 2.5])
def test_forward(alpha, beta, gamma, fixed):

    # Compute the output
    activation = AdaptiveSigmoid(
        alpha=alpha, beta=beta, gamma=gamma, fixed=fixed
    )
    output_ = activation(x)

    # Verify the output shape is the same as the input shape
    assert output_.shape == x.shape


@pytest.mark.parametrize("fixed", fixed_variables)
@pytest.mark.parametrize("alpha", [1, 2.5])
@pytest.mark.parametrize("beta", [1, 2.5])
@pytest.mark.parametrize("gamma", [1, 2.5])
def test_backward(alpha, beta, gamma, fixed):

    # Compute the output and perform backpropagation
    activation = AdaptiveSigmoid(
        alpha=alpha, beta=beta, gamma=gamma, fixed=fixed
    )
    output_ = activation(x.requires_grad_())
    loss = torch.mean(output_)
    loss.backward()

    # Verify that the gradients shape is the same as the input shape
    assert x.grad.shape == x.shape
