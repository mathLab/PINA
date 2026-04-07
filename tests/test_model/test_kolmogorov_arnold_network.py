import torch
import pytest
from pina.model import KolmogorovArnoldNetwork

# Data
input_dim = 3
data = torch.rand((10, input_dim))


@pytest.mark.parametrize("use_base_linear", [True, False])
@pytest.mark.parametrize("use_bias", [True, False])
@pytest.mark.parametrize("grid_range", [[-1, 1], [0, 2]])
@pytest.mark.parametrize("layers", [[input_dim, 5, 1], [input_dim, 2]])
def test_constructor(use_base_linear, use_bias, grid_range, layers):

    # Constructor
    KolmogorovArnoldNetwork(
        layers=layers,
        spline_order=3,
        n_knots=10,
        grid_range=grid_range,
        base_function=torch.nn.SiLU,
        use_base_linear=use_base_linear,
        use_bias=use_bias,
        init_scale_spline=1e-2,
        init_scale_base=1.0,
    )

    # Should fail if grid_range is not of length 2
    with pytest.raises(ValueError):
        KolmogorovArnoldNetwork(layers=layers, grid_range=[-1, 0, 1])

    # Should fail if layers has less than 2 elements
    with pytest.raises(ValueError):
        KolmogorovArnoldNetwork(layers=[input_dim])


@pytest.mark.parametrize("use_base_linear", [True, False])
@pytest.mark.parametrize("use_bias", [True, False])
@pytest.mark.parametrize("grid_range", [[-1, 1], [0, 2]])
@pytest.mark.parametrize("layers", [[input_dim, 5, 1], [input_dim, 2]])
def test_forward(use_base_linear, use_bias, grid_range, layers):

    model = KolmogorovArnoldNetwork(
        layers=layers,
        spline_order=3,
        n_knots=10,
        grid_range=grid_range,
        base_function=torch.nn.SiLU,
        use_base_linear=use_base_linear,
        use_bias=use_bias,
        init_scale_spline=1e-2,
        init_scale_base=1.0,
    )

    output_ = model(data)
    assert output_.shape == (data.shape[0], layers[-1])


@pytest.mark.parametrize("use_base_linear", [True, False])
@pytest.mark.parametrize("use_bias", [True, False])
@pytest.mark.parametrize("grid_range", [[-1, 1], [0, 2]])
@pytest.mark.parametrize("layers", [[input_dim, 5, 1], [input_dim, 2]])
def test_backward(use_base_linear, use_bias, grid_range, layers):

    model = KolmogorovArnoldNetwork(
        layers=layers,
        spline_order=3,
        n_knots=10,
        grid_range=grid_range,
        base_function=torch.nn.SiLU,
        use_base_linear=use_base_linear,
        use_bias=use_bias,
        init_scale_spline=1e-2,
        init_scale_base=1.0,
    )

    data.requires_grad_()
    output_ = model(data)

    loss = torch.mean(output_)
    loss.backward()
    assert data.grad.shape == data.shape
