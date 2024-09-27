import torch
import pytest

from pina.model import Spline

data = torch.rand((20, 3))
input_vars = 3
output_vars = 4

valid_args = [
    {
        'knots': torch.tensor([0., 0., 0., 1., 2., 3., 3., 3.]),
        'control_points': torch.tensor([0., 0., 1., 0., 0.]),
        'order': 3
    },
    {
        'knots': torch.tensor([-2., -2., -2., -2., -1., 0., 1., 2., 2., 2., 2.]),
        'control_points': torch.tensor([0., 0., 0., 6., 0., 0., 0.]),
        'order': 4
    },
    # {'control_points': {'n': 5, 'dim': 1}, 'order': 2},
    # {'control_points': {'n': 7, 'dim': 1}, 'order': 3}
]
 
def scipy_check(model, x, y):
    from scipy.interpolate._bsplines import BSpline
    import numpy as np
    spline = BSpline(
        t=model.knots.detach().numpy(),
        c=model.control_points.detach().numpy(),
        k=model.order-1
    )
    y_scipy = spline(x).flatten()
    y = y.detach().numpy()
    np.testing.assert_allclose(y, y_scipy, atol=1e-5)

@pytest.mark.parametrize("args", valid_args)
def test_constructor(args):
    Spline(**args)

def test_constructor_wrong():
    with pytest.raises(ValueError):
        Spline()

@pytest.mark.parametrize("args", valid_args)
def test_forward(args):
    min_x = args['knots'][0]
    max_x = args['knots'][-1]
    xi = torch.linspace(min_x, max_x, 1000)
    model = Spline(**args)
    yi = model(xi).squeeze()
    scipy_check(model, xi, yi)
    return 
    

@pytest.mark.parametrize("args", valid_args)
def test_backward(args):
    min_x = args['knots'][0]
    max_x = args['knots'][-1]
    xi = torch.linspace(min_x, max_x, 100)
    model = Spline(**args)
    yi = model(xi)
    fake_loss = torch.sum(yi)
    assert model.control_points.grad is None
    fake_loss.backward()
    assert model.control_points.grad is not None

    # dim_in, dim_out = 3, 2
    # fnn = FeedForward(dim_in, dim_out)
    # data.requires_grad = True
    # output_ = fnn(data)
    # l=torch.mean(output_)
    # l.backward()
    # assert data._grad.shape == torch.Size([20,3])
