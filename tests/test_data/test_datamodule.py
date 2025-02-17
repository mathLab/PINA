import torch
import pytest
from pina.data import PinaDataModule
from pina.data.dataset import PinaTensorDataset, PinaGraphDataset
from pina.problem.zoo import SupervisedProblem
from pina.graph import RadiusGraph
from pina.data.data_module import DummyDataloader
from pina import Trainer
from pina.solvers import SupervisedSolver
from torch_geometric.data import Batch
from torch.utils.data import DataLoader

input_tensor = torch.rand((100, 10))
output_tensor = torch.rand((100, 2))

x = torch.rand((100, 50 , 10))
pos = torch.rand((100, 50 , 2))
input_graph = RadiusGraph(x, pos, r=.1, build_edge_attr=True)
output_graph = torch.rand((100, 50 , 10))


@pytest.mark.parametrize(
    "input_, output_",
    [
        (input_tensor, output_tensor),
        (input_graph, output_graph)
    ]
)
def test_constructor(input_, output_):
    problem = SupervisedProblem(input_=input_, output_=output_)
    PinaDataModule(problem)

@pytest.mark.parametrize(
    "input_, output_",
    [
        (input_tensor, output_tensor),
        (input_graph, output_graph)
    ]
)
@pytest.mark.parametrize(
    "train_size, val_size, test_size",
    [
        (.7, .2, .1),
        (.7, .3, 0)
    ]
)
def test_setup_train(input_, output_, train_size, val_size, test_size):
    problem = SupervisedProblem(input_=input_, output_=output_)
    dm = PinaDataModule(problem, train_size=train_size, val_size=val_size, test_size=test_size)
    dm.setup()
    assert hasattr(dm, "train_dataset")
    if isinstance(input_, torch.Tensor):
        assert isinstance(dm.train_dataset, PinaTensorDataset)
    else:
        assert isinstance(dm.train_dataset, PinaGraphDataset)
    #assert len(dm.train_dataset) == int(len(input_) * train_size)
    if test_size > 0:
        assert hasattr(dm, "test_dataset")
        assert dm.test_dataset is None
    else:
        assert not hasattr(dm, "test_dataset")
    assert hasattr(dm, "val_dataset")
    if isinstance(input_, torch.Tensor):
        assert isinstance(dm.val_dataset, PinaTensorDataset)
    else:
        assert isinstance(dm.val_dataset, PinaGraphDataset)
    #assert len(dm.val_dataset) == int(len(input_) * val_size)

@pytest.mark.parametrize(
    "input_, output_",
    [
        (input_tensor, output_tensor),
        (input_graph, output_graph)
    ]
)
@pytest.mark.parametrize(
    "train_size, val_size, test_size",
    [
        (.7, .2, .1),
        (0., 0., 1.)
    ]
)
def test_setup_test(input_, output_, train_size, val_size, test_size):
    problem = SupervisedProblem(input_=input_, output_=output_)
    dm = PinaDataModule(problem, train_size=train_size, val_size=val_size, test_size=test_size)
    dm.setup(stage='test')
    if train_size > 0:
        assert hasattr(dm, "train_dataset")
        assert dm.train_dataset is None
    else:
        assert not hasattr(dm, "train_dataset")
    if val_size > 0:
        assert hasattr(dm, "val_dataset")
        assert dm.val_dataset is None
    else:
        assert not hasattr(dm, "val_dataset")
    
    assert hasattr(dm, "test_dataset")
    if isinstance(input_, torch.Tensor):
        assert isinstance(dm.test_dataset, PinaTensorDataset)
    else:
        assert isinstance(dm.test_dataset, PinaGraphDataset)
    #assert len(dm.test_dataset) == int(len(input_) * test_size)

@pytest.mark.parametrize(
    "input_, output_",
    [
        (input_tensor, output_tensor),
        (input_graph, output_graph)
    ]
)
def test_dummy_dataloader(input_, output_):
    problem = SupervisedProblem(input_=input_, output_=output_)
    solver = SupervisedSolver(problem=problem, model=torch.nn.Linear(10, 10))
    trainer = Trainer(solver, batch_size=None, train_size=.7, val_size=.3, test_size=0.)
    dm = trainer.data_module
    dm.setup()
    dm.trainer = trainer
    dataloader = dm.train_dataloader()
    assert isinstance(dataloader, DummyDataloader)
    assert len(dataloader) == 1
    data = next(dataloader)
    assert isinstance(data, list)
    assert isinstance(data[0], tuple)
    if isinstance(input_, RadiusGraph):
        assert isinstance(data[0][1]['input_points'], Batch)
    else:
        assert isinstance(data[0][1]['input_points'], torch.Tensor)
    assert isinstance(data[0][1]['output_points'], torch.Tensor)

    dataloader = dm.val_dataloader()
    assert isinstance(dataloader, DummyDataloader)
    assert len(dataloader) == 1
    data = next(dataloader)
    assert isinstance(data, list)
    assert isinstance(data[0], tuple)
    if isinstance(input_, RadiusGraph):
        assert isinstance(data[0][1]['input_points'], Batch)
    else:
        assert isinstance(data[0][1]['input_points'], torch.Tensor)
    assert isinstance(data[0][1]['output_points'], torch.Tensor)

@pytest.mark.parametrize(
    "input_, output_",
    [
        (input_tensor, output_tensor),
        (input_graph, output_graph)
    ]
)
def test_dataloader(input_, output_):
    problem = SupervisedProblem(input_=input_, output_=output_)
    solver = SupervisedSolver(problem=problem, model=torch.nn.Linear(10, 10))
    trainer = Trainer(solver, batch_size=10, train_size=.7, val_size=.3, test_size=0.)
    dm = trainer.data_module
    dm.setup()
    dm.trainer = trainer
    dataloader = dm.train_dataloader()
    assert isinstance(dataloader, DataLoader)
    assert len(dataloader) == 7
    data = next(iter(dataloader))
    assert isinstance(data, dict)
    if isinstance(input_, RadiusGraph):
        assert isinstance(data['data']['input_points'], Batch)
    else:
        assert isinstance(data['data']['input_points'], torch.Tensor)
    assert isinstance(data['data']['output_points'], torch.Tensor)

    dataloader = dm.val_dataloader()
    assert isinstance(dataloader, DataLoader)
    assert len(dataloader) == 3
    data = next(iter(dataloader))
    assert isinstance(data, dict)
    if isinstance(input_, RadiusGraph):
        assert isinstance(data['data']['input_points'], Batch)
    else:
        assert isinstance(data['data']['input_points'], torch.Tensor)
    assert isinstance(data['data']['output_points'], torch.Tensor)

