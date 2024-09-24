import torch
import pytest

from pina.model import Spline

data = torch.rand((20, 3))
input_vars = 3
output_vars = 4

valid_args = [
    {'control_points': {'n': 5, 'dim': 1}, 'order': 3},
    {'control_points': {'n': 8, 'dim': 1}, 'order': 3}
]
 
def scipy_check(model, x, y):

    from scipy.interpolate._bsplines import BSpline
    import numpy as np
    spline = BSpline(
        t=model.knots.detach(),
        c=model.control_points.detach(),
        k=model.order-1
    )
    y_scipy = spline(x)
    y = y.detach().numpy()
    np.testing.assert_allclose(y, y_numpy, atol=1e-5)

@pytest.mark.parametrize("args", valid_args)
def test_constructor(args):
    Spline(**args)

def test_constructor_wrong():
    with pytest.raises(TypeError):
        Spline()

@pytest.mark.parametrize("args", valid_args)
def test_forward(args):
    xi = torch.linspace(0, 1, 100)
    model = Spline(*args)
    yi = model(xi).squeeze()
    scipy_check(model, xi, yi)
    return 
    

def test_backward():
    pass
    # dim_in, dim_out = 3, 2
    # fnn = FeedForward(dim_in, dim_out)
    # data.requires_grad = True
    # output_ = fnn(data)
    # l=torch.mean(output_)
    # l.backward()
    # assert data._grad.shape == torch.Size([20,3])
