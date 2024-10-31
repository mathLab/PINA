import math
import torch
from pina.data import SamplePointDataset, SupervisedDataset, PinaDataModule, \
    UnsupervisedDataset
from pina.data import PinaDataLoader
from pina import LabelTensor, Condition
from pina.equation import Equation
from pina.domain import CartesianDomain
from pina.problem import SpatialProblem, AbstractProblem
from pina.operators import laplacian
from pina.equation.equation_factory import FixedValue
from pina.graph import Graph


def laplace_equation(input_, output_):
    force_term = (torch.sin(input_.extract(['x']) * torch.pi) *
                  torch.sin(input_.extract(['y']) * torch.pi))
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
        'gamma1':
            Condition(domain=CartesianDomain({
                'x': [0, 1],
                'y': 1
            }),
                equation=FixedValue(0.0)),
        'gamma2':
            Condition(domain=CartesianDomain({
                'x': [0, 1],
                'y': 0
            }),
                equation=FixedValue(0.0)),
        'gamma3':
            Condition(domain=CartesianDomain({
                'x': 1,
                'y': [0, 1]
            }),
                equation=FixedValue(0.0)),
        'gamma4':
            Condition(domain=CartesianDomain({
                'x': 0,
                'y': [0, 1]
            }),
                equation=FixedValue(0.0)),
        'D':
            Condition(input_points=LabelTensor(torch.rand(size=(100, 2)),
                                               ['x', 'y']),
                      equation=my_laplace),
        'data':
            Condition(input_points=in_, output_points=out_),
        'data2':
            Condition(input_points=in2_, output_points=out2_),
        'unsupervised':
            Condition(
                input_points=LabelTensor(torch.rand(size=(45, 2)), ['x', 'y']),
                conditional_variables=LabelTensor(torch.ones(size=(45, 1)),
                                                  ['alpha']),
            ),
        'unsupervised2':
            Condition(
                input_points=LabelTensor(torch.rand(size=(90, 2)), ['x', 'y']),
                conditional_variables=LabelTensor(torch.ones(size=(90, 1)),
                                                  ['alpha']),
            )
    }


boundaries = ['gamma1', 'gamma2', 'gamma3', 'gamma4']
poisson = Poisson()
poisson.discretise_domain(10, 'grid', locations=boundaries)


def test_sample():
    sample_dataset = SamplePointDataset(poisson, device='cpu')
    assert len(sample_dataset) == 140
    assert sample_dataset.input_points.shape == (140, 2)
    assert sample_dataset.input_points.labels == ['x', 'y']
    assert sample_dataset.condition_indices.dtype == torch.uint8
    assert sample_dataset.condition_indices.max() == torch.tensor(4)
    assert sample_dataset.condition_indices.min() == torch.tensor(0)


def test_data():
    dataset = SupervisedDataset(poisson, device='cpu')
    assert len(dataset) == 61
    assert dataset['input_points'].shape == (61, 2)
    assert dataset.input_points.shape == (61, 2)
    assert dataset['input_points'].labels == ['x', 'y']
    assert dataset.input_points.labels == ['x', 'y']
    assert dataset.input_points[3:].shape == (58, 2)
    assert dataset.output_points[:3].labels == ['u']
    assert dataset.output_points.shape == (61, 1)
    assert dataset.output_points.labels == ['u']
    assert dataset.condition_indices.dtype == torch.uint8
    assert dataset.condition_indices.max() == torch.tensor(1)
    assert dataset.condition_indices.min() == torch.tensor(0)


def test_unsupervised():
    dataset = UnsupervisedDataset(poisson, device='cpu')
    assert len(dataset) == 135
    assert dataset.input_points.shape == (135, 2)
    assert dataset.input_points.labels == ['x', 'y']
    assert dataset.input_points[3:].shape == (132, 2)

    assert dataset.conditional_variables.shape == (135, 1)
    assert dataset.conditional_variables.labels == ['alpha']
    assert dataset.condition_indices.dtype == torch.uint8
    assert dataset.condition_indices.max() == torch.tensor(1)
    assert dataset.condition_indices.min() == torch.tensor(0)


def test_data_module():
    data_module = PinaDataModule(poisson, device='cpu')
    data_module.setup()
    loader = data_module.train_dataloader()
    assert isinstance(loader, PinaDataLoader)
    assert isinstance(loader, PinaDataLoader)

    data_module = PinaDataModule(poisson,
                                 device='cpu',
                                 batch_size=10,
                                 shuffle=False)
    data_module.setup()
    loader = data_module.train_dataloader()
    assert len(loader) == 24
    for i in loader:
        assert len(i) <= 10
    len_ref = sum(
        [math.ceil(len(dataset) * 0.7) for dataset in data_module.datasets])
    len_real = sum(
        [len(dataset) for dataset in data_module.splits['train'].values()])
    assert len_ref == len_real

    supervised_dataset = SupervisedDataset(poisson, device='cpu')
    data_module = PinaDataModule(poisson,
                                 device='cpu',
                                 batch_size=10,
                                 shuffle=False,
                                 datasets=[supervised_dataset])
    data_module.setup()
    loader = data_module.train_dataloader()
    for batch in loader:
        assert len(batch) <= 10

    physics_dataset = SamplePointDataset(poisson, device='cpu')
    data_module = PinaDataModule(poisson,
                                 device='cpu',
                                 batch_size=10,
                                 shuffle=False,
                                 datasets=[physics_dataset])
    data_module.setup()
    loader = data_module.train_dataloader()
    for batch in loader:
        assert len(batch) <= 10

    unsupervised_dataset = UnsupervisedDataset(poisson, device='cpu')
    data_module = PinaDataModule(poisson,
                                 device='cpu',
                                 batch_size=10,
                                 shuffle=False,
                                 datasets=[unsupervised_dataset])
    data_module.setup()
    loader = data_module.train_dataloader()
    for batch in loader:
        assert len(batch) <= 10


def test_loader():
    data_module = PinaDataModule(poisson, device='cpu', batch_size=10)
    data_module.setup()
    loader = data_module.train_dataloader()
    assert isinstance(loader, PinaDataLoader)
    assert len(loader) == 24
    for i in loader:
        assert len(i) <= 10
        assert i.supervised.input_points.labels == ['x', 'y']
        assert i.physics.input_points.labels == ['x', 'y']
        assert i.unsupervised.input_points.labels == ['x', 'y']
        assert i.supervised.input_points.requires_grad == True
        assert i.physics.input_points.requires_grad == True
        assert i.unsupervised.input_points.requires_grad == True


coordinates = LabelTensor(torch.rand((100, 100, 2)), labels=['x', 'y'])
data = LabelTensor(torch.rand((100, 100, 3)), labels=['ux', 'uy', 'p'])


class GraphProblem(AbstractProblem):
    output = LabelTensor(torch.rand((100, 3)), labels=['ux', 'uy', 'p'])
    input = [
        Graph.build('radius',
                    nodes_coordinates=coordinates[i, :, :],
                    nodes_data=data[i, :, :],
                    radius=0.2) for i in range(100)
    ]
    output_variables = ['u']

    conditions = {
        'graph_data': Condition(input_points=input, output_points=output)
    }


graph_problem = GraphProblem()


def test_loader_graph():
    data_module = PinaDataModule(graph_problem, device='cpu', batch_size=10)
    data_module.setup()
    loader = data_module.train_dataloader()
    for i in loader:
        assert len(i) <= 10
        assert isinstance(i.supervised.input_points, list)
        assert all(isinstance(x, Graph) for x in i.supervised.input_points)
