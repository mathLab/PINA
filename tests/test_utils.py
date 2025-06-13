import torch
import pytest

from pina import LabelTensor
from pina.utils import merge_tensors, check_consistency, check_positive_integer
from pina.domain import EllipsoidDomain, CartesianDomain, DomainInterface


def test_merge_tensors():
    tensor1 = LabelTensor(torch.rand((20, 3)), ["a", "b", "c"])
    tensor2 = LabelTensor(torch.zeros((20, 3)), ["d", "e", "f"])
    tensor3 = LabelTensor(torch.ones((30, 3)), ["g", "h", "i"])

    merged_tensor = merge_tensors((tensor1, tensor2, tensor3))
    assert tuple(merged_tensor.labels) == (
        "a",
        "b",
        "c",
        "d",
        "e",
        "f",
        "g",
        "h",
        "i",
    )
    assert merged_tensor.shape == (20 * 20 * 30, 9)
    assert torch.all(merged_tensor.extract(("d", "e", "f")) == 0)
    assert torch.all(merged_tensor.extract(("g", "h", "i")) == 1)


def test_check_consistency_correct():
    ellipsoid1 = EllipsoidDomain({"x": [1, 2], "y": [-2, 1]})
    example_input_pts = LabelTensor(torch.tensor([[0, 0, 0]]), ["x", "y", "z"])

    check_consistency(example_input_pts, torch.Tensor)
    check_consistency(CartesianDomain, DomainInterface, subclass=True)
    check_consistency(ellipsoid1, DomainInterface)


def test_check_consistency_incorrect():
    ellipsoid1 = EllipsoidDomain({"x": [1, 2], "y": [-2, 1]})
    example_input_pts = LabelTensor(torch.tensor([[0, 0, 0]]), ["x", "y", "z"])

    with pytest.raises(ValueError):
        check_consistency(example_input_pts, DomainInterface)
    with pytest.raises(ValueError):
        check_consistency(torch.Tensor, DomainInterface, subclass=True)
    with pytest.raises(ValueError):
        check_consistency(ellipsoid1, torch.Tensor)


@pytest.mark.parametrize("value", [0, 1, 2, 3, 10])
@pytest.mark.parametrize("strict", [True, False])
def test_check_positive_integer(value, strict):
    if value != 0:
        check_positive_integer(value, strict=strict)
    else:
        check_positive_integer(value, strict=False)

    # Should fail if value is negative
    with pytest.raises(AssertionError):
        check_positive_integer(-1, strict=strict)

    # Should fail if value is not an integer
    with pytest.raises(AssertionError):
        check_positive_integer(1.5, strict=strict)

    # Should fail if value is not a number
    with pytest.raises(AssertionError):
        check_positive_integer("string", strict=strict)
