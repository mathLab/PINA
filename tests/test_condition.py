import torch
import pytest

from pina import LabelTensor, Condition, Span, PINN
from pina.problem import SpatialProblem
from pina.model import FeedForward
from pina.operators import nabla


example_domain = Span({'x': [0, 1], 'y': [0, 1]})
def example_dirichlet(input_, output_):
    value = 0.0
    return output_.extract(['u']) - value
example_input_pts = LabelTensor(torch.tensor([[0, 0, 0]]), ['x', 'y', 'z'])
example_output_pts = LabelTensor(torch.tensor([[1, 2]]), ['a', 'b'])

def test_init_inputoutput():
    Condition(input_points=example_input_pts, output_points=example_output_pts)
    with pytest.raises(ValueError):
        Condition(example_input_pts, example_output_pts)
    with pytest.raises(TypeError):
        Condition(input_points=3., output_points='example')
    with pytest.raises(TypeError):
        Condition(input_points=example_domain, output_points=example_dirichlet)

def test_init_locfunc():
    Condition(location=example_domain, function=example_dirichlet)
    with pytest.raises(ValueError):
        Condition(example_domain, example_dirichlet)
    with pytest.raises(TypeError):
        Condition(location=3., function='example')
    with pytest.raises(TypeError):
        Condition(location=example_input_pts, function=example_output_pts)

def test_init_inputfunc():
    Condition(input_points=example_input_pts, function=example_dirichlet)
    with pytest.raises(ValueError):
        Condition(example_domain, example_dirichlet)
    with pytest.raises(TypeError):
        Condition(input_points=3., function='example')
    with pytest.raises(TypeError):
        Condition(input_points=example_domain, function=example_output_pts)