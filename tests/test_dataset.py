import torch
import pytest

from pina.data.dataset import SamplePointDataset, SamplePointLoader, DataPointDataset
from pina import LabelTensor, Condition
from pina.equation import Equation
from pina.domain import CartesianDomain
from pina.problem import SpatialProblem
from pina.model import FeedForward
from pina.operators import laplacian
from pina.equation.equation_factory import FixedValue


def laplace_equation(input_, output_):
    force_term = (torch.sin(input_.extract(['x'])*torch.pi) *
                    torch.sin(input_.extract(['y'])*torch.pi))
    delta_u = laplacian(output_.extract(['u']), input_)
    return delta_u - force_term

my_laplace = Equation(laplace_equation)
in_ = LabelTensor(torch.tensor([[0., 1.]]), ['x', 'y'])
out_ = LabelTensor(torch.tensor([[0.]]), ['u'])
in2_ = LabelTensor(torch.rand(60, 2), ['x', 'y'])
out2_ = LabelTensor(torch.rand(60, 1), ['u'])

class Poisson(SpatialProblem):
    output_variables = ['u']
    spatial_domain = CartesianDomain({'x': [0, 1], 'y': [0, 1]})

    conditions = {
        'gamma1': Condition(
            location=CartesianDomain({'x': [0, 1], 'y':  1}),
            equation=FixedValue(0.0)),
        'gamma2': Condition(
            location=CartesianDomain({'x': [0, 1], 'y': 0}),
            equation=FixedValue(0.0)),
        'gamma3': Condition(
            location=CartesianDomain({'x':  1, 'y': [0, 1]}),
            equation=FixedValue(0.0)),
        'gamma4': Condition(
            location=CartesianDomain({'x': 0, 'y': [0, 1]}),
            equation=FixedValue(0.0)),
        'D': Condition(
            input_points=LabelTensor(torch.rand(size=(100, 2)), ['x', 'y']),
            equation=my_laplace),
        'data': Condition(
            input_points=in_,
            output_points=out_),
        'data2': Condition(
            input_points=in2_,
            output_points=out2_)
    }

boundaries = ['gamma1', 'gamma2', 'gamma3', 'gamma4']
poisson = Poisson()
poisson.discretise_domain(10, 'grid', locations=boundaries)

def test_sample():
    sample_dataset = SamplePointDataset(poisson, device='cpu')
    assert len(sample_dataset) == 140
    assert sample_dataset.pts.shape == (140, 2)
    assert sample_dataset.pts.labels == ['x', 'y']
    assert sample_dataset.condition_indeces.dtype == torch.int64
    assert sample_dataset.condition_indeces.max() == torch.tensor(4)
    assert sample_dataset.condition_indeces.min() == torch.tensor(0)

def test_data():
    dataset = DataPointDataset(poisson, device='cpu')
    assert len(dataset) == 61
    assert dataset.input_pts.shape == (61, 2)
    assert dataset.input_pts.labels == ['x', 'y']
    assert dataset.output_pts.shape == (61, 1 )
    assert dataset.output_pts.labels == ['u']
    assert dataset.condition_indeces.dtype == torch.int64
    assert dataset.condition_indeces.max() == torch.tensor(1)
    assert dataset.condition_indeces.min() == torch.tensor(0)

def test_loader():
    sample_dataset = SamplePointDataset(poisson, device='cpu')
    data_dataset = DataPointDataset(poisson, device='cpu')
    loader = SamplePointLoader(sample_dataset, data_dataset, batch_size=10)

    for batch in loader:
        assert len(batch) in [2, 3]
        assert batch['pts'].shape[0] <= 10
        assert batch['pts'].requires_grad == True
        assert batch['pts'].labels == ['x', 'y']

    loader2 = SamplePointLoader(sample_dataset, data_dataset, batch_size=None)
    assert len(list(loader2)) == 2

def test_loader2():
    poisson2 = Poisson()
    del poisson.conditions['data2']
    del poisson2.conditions['data']
    poisson2.discretise_domain(10, 'grid', locations=boundaries)
    sample_dataset = SamplePointDataset(poisson, device='cpu')
    data_dataset = DataPointDataset(poisson, device='cpu')
    loader = SamplePointLoader(sample_dataset, data_dataset, batch_size=10)

    for batch in loader:
        assert len(batch) == 2 # only phys condtions
        assert batch['pts'].shape[0] <= 10
        assert batch['pts'].requires_grad == True
        assert batch['pts'].labels == ['x', 'y']

def test_loader3():
    poisson2 = Poisson()
    del poisson.conditions['gamma1']
    del poisson.conditions['gamma2']
    del poisson.conditions['gamma3']
    del poisson.conditions['gamma4']
    del poisson.conditions['D']
    sample_dataset = SamplePointDataset(poisson, device='cpu')
    data_dataset = DataPointDataset(poisson, device='cpu')
    loader = SamplePointLoader(sample_dataset, data_dataset, batch_size=10)

    for batch in loader:
        assert len(batch) == 2 # only phys condtions
        assert batch['pts'].shape[0] <= 10
        assert batch['pts'].requires_grad == True
        assert batch['pts'].labels == ['x', 'y']
