import torch
import pytest
from pina.model.block import KANBlock

# Data
input_dim = 3
data = torch.rand((10, input_dim))


@pytest.mark.parametrize("output_dimensions", [1, 5])
@pytest.mark.parametrize("spline_order", [3, 4])
@pytest.mark.parametrize("n_knots", [10, 20])
@pytest.mark.parametrize("init_scale_spline", [1e-2, 1e-1])
@pytest.mark.parametrize("init_scale_base", [1.0, 0.1])
def test_constructor(
    output_dimensions, spline_order, n_knots, init_scale_spline, init_scale_base
):

    KANBlock(
        input_dimensions=data.shape[1],
        output_dimensions=output_dimensions,
        spline_order=spline_order,
        n_knots=n_knots,
        init_scale_spline=init_scale_spline,
        init_scale_base=init_scale_base,
    )

    # Should fail if input_dimensions is not a positive integer
    with pytest.raises(AssertionError):
        KANBlock(input_dimensions=-1, output_dimensions=output_dimensions)

    # Should fail if output_dimensions is not a positive integer
    with pytest.raises(AssertionError):
        KANBlock(input_dimensions=data.shape[1], output_dimensions=-1)

    # Should fail if spline_order is not a positive integer
    with pytest.raises(AssertionError):
        KANBlock(
            input_dimensions=data.shape[1],
            output_dimensions=output_dimensions,
            spline_order=-1,
        )

    # Should fail if n_knots is not a positive integer
    with pytest.raises(AssertionError):
        KANBlock(
            input_dimensions=data.shape[1],
            output_dimensions=output_dimensions,
            n_knots=-1,
        )

    # Should fail if grid_range is not of length 2
    with pytest.raises(ValueError):
        KANBlock(
            input_dimensions=data.shape[1],
            output_dimensions=output_dimensions,
            grid_range=[-1, 0, 1],
        )

    # Should fail if base_function is not a torch.nn.Module subclass
    with pytest.raises(ValueError):
        KANBlock(
            input_dimensions=data.shape[1],
            output_dimensions=output_dimensions,
            base_function="not_a_module",
        )

    # Should fail if use_base_linear is not a boolean
    with pytest.raises(ValueError):
        KANBlock(
            input_dimensions=data.shape[1],
            output_dimensions=output_dimensions,
            use_base_linear="not_a_bool",
        )

    # Should fail if use_bias is not a boolean
    with pytest.raises(ValueError):
        KANBlock(
            input_dimensions=data.shape[1],
            output_dimensions=output_dimensions,
            use_bias="not_a_bool",
        )

    # Should fail if init_scale_spline is not a float or int
    with pytest.raises(ValueError):
        KANBlock(
            input_dimensions=data.shape[1],
            output_dimensions=output_dimensions,
            init_scale_spline="not_a_number",
        )

    # Should fail if init_scale_base is not a float or int
    with pytest.raises(ValueError):
        KANBlock(
            input_dimensions=data.shape[1],
            output_dimensions=output_dimensions,
            init_scale_base="not_a_number",
        )


@pytest.mark.parametrize("output_dimensions", [1, 5])
@pytest.mark.parametrize("spline_order", [3, 4])
@pytest.mark.parametrize("n_knots", [10, 20])
@pytest.mark.parametrize("init_scale_spline", [1e-2, 1e-1])
@pytest.mark.parametrize("init_scale_base", [1.0, 0.1])
def test_forward(
    output_dimensions, spline_order, n_knots, init_scale_spline, init_scale_base
):

    model = KANBlock(
        input_dimensions=data.shape[1],
        output_dimensions=output_dimensions,
        spline_order=spline_order,
        n_knots=n_knots,
        init_scale_spline=init_scale_spline,
        init_scale_base=init_scale_base,
    )

    output_ = model(data)
    assert output_.shape == (data.shape[0], output_dimensions)


@pytest.mark.parametrize("output_dimensions", [1, 5])
@pytest.mark.parametrize("spline_order", [3, 4])
@pytest.mark.parametrize("n_knots", [10, 20])
@pytest.mark.parametrize("init_scale_spline", [1e-2, 1e-1])
@pytest.mark.parametrize("init_scale_base", [1.0, 0.1])
def test_backward(
    output_dimensions, spline_order, n_knots, init_scale_spline, init_scale_base
):

    model = KANBlock(
        input_dimensions=data.shape[1],
        output_dimensions=output_dimensions,
        spline_order=spline_order,
        n_knots=n_knots,
        init_scale_spline=init_scale_spline,
        init_scale_base=init_scale_base,
    )

    data.requires_grad_()
    output_ = model(data)

    loss = torch.mean(output_)
    loss.backward()
    assert data.grad.shape == data.shape
