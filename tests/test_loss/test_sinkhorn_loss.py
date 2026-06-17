import torch
import pytest

from pina.loss import SinkhornLoss

# Fixed random tensors for reproducibility
torch.manual_seed(0)
input_ = torch.rand(10, 2)
target_ = torch.rand(8, 2)


@pytest.mark.parametrize("p", [1, 2, 3])
@pytest.mark.parametrize("eps", [0.01, 0.1, 1.0])
@pytest.mark.parametrize("max_iter", [10, 100])
@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
def test_constructor(p, eps, max_iter, reduction):

    SinkhornLoss(p=p, eps=eps, max_iter=max_iter, reduction=reduction)

    # Should fail if p is not numeric
    with pytest.raises(ValueError):
        SinkhornLoss(p="invalid", eps=eps, max_iter=max_iter, reduction=reduction)

    # Should fail if eps is not a float
    with pytest.raises(ValueError):
        SinkhornLoss(p=p, eps=1, max_iter=max_iter, reduction=reduction)

    # Should fail if eps is not positive
    with pytest.raises(ValueError):
        SinkhornLoss(p=p, eps=-0.1, max_iter=max_iter, reduction=reduction)

    # Should fail if max_iter is not a positive integer
    with pytest.raises(AssertionError):
        SinkhornLoss(p=p, eps=eps, max_iter=0, reduction=reduction)

    # Should fail if reduction is invalid
    with pytest.raises(ValueError):
        SinkhornLoss(p=p, eps=eps, max_iter=max_iter, reduction="invalid")


@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
def test_forward_shape(reduction):

    loss_fn = SinkhornLoss(reduction=reduction)
    value = loss_fn(input_, target_)
    assert value.shape == torch.Size([1])


def test_forward_finite():

    # The (non-debiased) Sinkhorn dual can be negative due to the entropy
    # regularization term, but it must always be finite.
    loss_fn = SinkhornLoss()
    value = loss_fn(input_, target_)
    assert torch.isfinite(value).all()


def test_forward_same_distribution_smaller():

    # Sinkhorn loss on identical data should be smaller than on different data
    loss_same = SinkhornLoss(eps=1e-3, max_iter=500)(input_, input_)
    loss_diff = SinkhornLoss(eps=1e-3, max_iter=500)(input_, target_)
    assert loss_same.item() < loss_diff.item()


def test_forward_asymmetric_sizes():

    # input and target may have different numbers of rows
    x = torch.rand(5, 3)
    y = torch.rand(8, 3)
    value = SinkhornLoss()(x, y)
    assert value.shape == torch.Size([1])
    assert torch.isfinite(value).all()


def test_forward_approaches_wasserstein():

    # For 1-D sorted distributions, W_2^2 = sum |x_i - y_i|^2 / N
    x = torch.tensor([[1.0], [2.0], [3.0]])
    y = torch.tensor([[4.0], [5.0], [6.0]])
    # W_2^2 = ((1-4)^2 + (2-5)^2 + (3-6)^2) / 3 = 9
    value = SinkhornLoss(p=2, eps=1e-3, max_iter=5000)(x, y)
    assert abs(value.item() - 9.0) < 0.1
