import torch
import pytest
import math

from pina.model.block.rbf_block import RBFBlock

x = torch.linspace(-1, 1, 100)
toy_params = torch.linspace(0, 1, 10).unsqueeze(1)
toy_snapshots = torch.vstack([torch.exp(-(x**2)) * c for c in toy_params])
toy_params_test = torch.linspace(0, 1, 3).unsqueeze(1)
toy_snapshots_test = torch.vstack(
    [torch.exp(-(x**2)) * c for c in toy_params_test]
)

kernels = [
    "linear",
    "thin_plate_spline",
    "cubic",
    "quintic",
    "multiquadric",
    "inverse_multiquadric",
    "inverse_quadratic",
    "gaussian",
]

noscale_invariant_kernels = [
    "multiquadric",
    "inverse_multiquadric",
    "inverse_quadratic",
    "gaussian",
]

scale_invariant_kernels = ["linear", "thin_plate_spline", "cubic", "quintic"]


def test_constructor_default():
    rbf = RBFBlock()
    assert rbf.kernel == "thin_plate_spline"
    assert rbf.epsilon == 1
    assert rbf.smoothing == 0.0


@pytest.mark.parametrize("kernel", kernels)
@pytest.mark.parametrize("epsilon", [0.1, 1.0, 10.0])
def test_constructor_epsilon(kernel, epsilon):
    if kernel in scale_invariant_kernels:
        rbf = RBFBlock(kernel=kernel)
        assert rbf.kernel == kernel
        assert rbf.epsilon == 1
    elif kernel in noscale_invariant_kernels:
        with pytest.raises(ValueError):
            rbf = RBFBlock(kernel=kernel)
        rbf = RBFBlock(kernel=kernel, epsilon=epsilon)
        assert rbf.kernel == kernel
        assert rbf.epsilon == epsilon

    assert rbf.smoothing == 0.0


@pytest.mark.parametrize("kernel", kernels)
@pytest.mark.parametrize("epsilon", [0.1, 1.0, 10.0])
@pytest.mark.parametrize("degree", [2, 3, 4])
@pytest.mark.parametrize("smoothing", [1e-5, 1e-3, 1e-1])
def test_constructor_all(kernel, epsilon, degree, smoothing):
    rbf = RBFBlock(
        kernel=kernel, epsilon=epsilon, degree=degree, smoothing=smoothing
    )
    assert rbf.kernel == kernel
    assert rbf.epsilon == epsilon
    assert rbf.degree == degree
    assert rbf.smoothing == smoothing
    assert rbf.y == None
    assert rbf.d == None
    assert rbf.powers == None
    assert rbf._shift == None
    assert rbf._scale == None
    assert rbf._coeffs == None


def test_fit():
    rbf = RBFBlock()
    rbf.fit(toy_params, toy_snapshots)
    ndim = toy_params.shape[1]
    torch.testing.assert_close(rbf.y, toy_params)
    torch.testing.assert_close(rbf.d, toy_snapshots)
    assert rbf.powers.shape == (math.comb(rbf.degree + ndim, ndim), ndim)
    assert rbf._shift.shape == (ndim,)
    assert rbf._scale.shape == (ndim,)
    assert rbf._coeffs.shape == (
        rbf.powers.shape[0] + toy_snapshots.shape[0],
        toy_snapshots.shape[1],
    )


def test_forward():
    rbf = RBFBlock()
    rbf.fit(toy_params, toy_snapshots)
    c = rbf(toy_params)
    assert c.shape == toy_snapshots.shape
    torch.testing.assert_close(c, toy_snapshots)


def test_forward_unseen_parameters():
    rbf = RBFBlock()
    rbf.fit(toy_params, toy_snapshots)
    c = rbf(toy_params_test)
    assert c.shape == toy_snapshots_test.shape
    torch.testing.assert_close(c, toy_snapshots_test)
