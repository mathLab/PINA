import torch
import pytest
from pina.data import PinaDataModule

# from pina.data import PinaTensorDataset, PinaGraphDataset
from pina.problem.zoo import SupervisedProblem
from pina.graph import RadiusGraph

# from pina.data import DummyDataloader
from pina._src.data.data_module import _ConditionSubset
from pina import Trainer
from pina.solver import SupervisedSolver
from torch_geometric.data import Batch
from torch.utils.data import DataLoader
from pina.problem.zoo import Poisson2DSquareProblem
from pina._src.data.aggregator import _Aggregator
from pina.solver import PINN


def _create_tensor_data():
    input_tensor = torch.rand((100, 10))
    output_tensor = torch.rand((100, 2))
    return input_tensor, output_tensor


def _create_graph_data():
    x = torch.rand((100, 50, 10))
    pos = torch.rand((100, 50, 2))
    input_graph = [
        RadiusGraph(x=x_, pos=pos_, radius=0.2) for x_, pos_, in zip(x, pos)
    ]
    output_graph = torch.rand((100, 50, 2))
    return input_graph, output_graph


def test_init_tensor():
    input_tensor, output_tensor = _create_tensor_data()
    problem = SupervisedProblem(input_=input_tensor, output_=output_tensor)
    dm = PinaDataModule(problem)
    assert dm.problem == problem
    assert dm.trainer is None
    assert hasattr(dm, "split_idxs")
    assert isinstance(dm.split_idxs, dict)
    assert set(dm.split_idxs.keys()) == {"data"}
    assert isinstance(dm.split_idxs["data"], dict)
    assert set(dm.split_idxs["data"].keys()) == {"train", "val", "test"}
    assert isinstance(dm.split_idxs["data"]["train"], list)
    assert isinstance(dm.split_idxs["data"]["val"], list)
    assert isinstance(dm.split_idxs["data"]["test"], list)
    assert len(dm.split_idxs["data"]["train"]) == 70
    assert len(dm.split_idxs["data"]["val"]) == 10
    assert len(dm.split_idxs["data"]["test"]) == 20


def test_init_graph():
    input_graph, output_graph = _create_graph_data()
    problem = SupervisedProblem(input_=input_graph, output_=output_graph)
    dm = PinaDataModule(problem)
    assert dm.problem == problem
    assert dm.trainer is None
    assert hasattr(dm, "split_idxs")
    assert isinstance(dm.split_idxs, dict)
    assert set(dm.split_idxs.keys()) == {"data"}
    assert isinstance(dm.split_idxs["data"], dict)
    assert set(dm.split_idxs["data"].keys()) == {"train", "val", "test"}
    assert isinstance(dm.split_idxs["data"]["train"], list)
    assert isinstance(dm.split_idxs["data"]["val"], list)
    assert isinstance(dm.split_idxs["data"]["test"], list)
    assert len(dm.split_idxs["data"]["train"]) == 70
    assert len(dm.split_idxs["data"]["val"]) == 10
    assert len(dm.split_idxs["data"]["test"]) == 20


def test_init_poisson():
    problem = Poisson2DSquareProblem()
    problem.discretise_domain(n=10, mode="grid")
    dm = PinaDataModule(problem)
    assert dm.problem == problem
    assert dm.trainer is None
    assert hasattr(dm, "split_idxs")
    assert isinstance(dm.split_idxs, dict)
    assert set(dm.split_idxs.keys()) == {"D", "boundary"}
    assert isinstance(dm.split_idxs["D"], dict)
    assert set(dm.split_idxs["D"].keys()) == {"train", "val", "test"}
    assert isinstance(dm.split_idxs["D"]["train"], list)
    assert isinstance(dm.split_idxs["D"]["val"], list)
    assert isinstance(dm.split_idxs["D"]["test"], list)
    assert len(dm.split_idxs["D"]["train"]) == 70
    assert len(dm.split_idxs["D"]["val"]) == 10
    assert len(dm.split_idxs["D"]["test"]) == 20

    assert isinstance(dm.split_idxs["boundary"], dict)
    assert set(dm.split_idxs["boundary"].keys()) == {"train", "val", "test"}
    assert isinstance(dm.split_idxs["boundary"]["train"], list)
    assert isinstance(dm.split_idxs["boundary"]["val"], list)
    assert isinstance(dm.split_idxs["boundary"]["test"], list)
    assert len(dm.split_idxs["boundary"]["train"]) == 7
    assert len(dm.split_idxs["boundary"]["val"]) == 1
    assert len(dm.split_idxs["boundary"]["test"]) == 2


def test_setup_tensor():
    input_tensor, output_tensor = _create_tensor_data()
    problem = SupervisedProblem(input_=input_tensor, output_=output_tensor)
    dm = PinaDataModule(problem)
    dm.setup()
    assert hasattr(dm, "train_datasets")
    assert isinstance(dm.train_datasets, dict)
    assert set(dm.train_datasets.keys()) == {"data"}
    assert isinstance(dm.train_datasets["data"], _ConditionSubset)
    assert hasattr(dm, "val_datasets")
    assert isinstance(dm.val_datasets, dict)
    assert set(dm.val_datasets.keys()) == {"data"}
    assert isinstance(dm.val_datasets["data"], _ConditionSubset)
    assert hasattr(dm, "test_datasets")
    assert isinstance(dm.test_datasets, dict)
    assert set(dm.test_datasets.keys()) == {"data"}
    assert isinstance(dm.test_datasets["data"], _ConditionSubset)


def test_setup_graph():
    input_graph, output_graph = _create_graph_data()
    problem = SupervisedProblem(input_=input_graph, output_=output_graph)
    dm = PinaDataModule(problem)
    dm.setup()
    assert hasattr(dm, "train_datasets")
    assert isinstance(dm.train_datasets, dict)
    assert set(dm.train_datasets.keys()) == {"data"}
    assert isinstance(dm.train_datasets["data"], _ConditionSubset)
    assert hasattr(dm, "val_datasets")
    assert isinstance(dm.val_datasets, dict)
    assert set(dm.val_datasets.keys()) == {"data"}
    assert isinstance(dm.val_datasets["data"], _ConditionSubset)
    assert hasattr(dm, "test_datasets")
    assert isinstance(dm.test_datasets, dict)
    assert set(dm.test_datasets.keys()) == {"data"}
    assert isinstance(dm.test_datasets["data"], _ConditionSubset)


def test_setup_poisson():
    problem = Poisson2DSquareProblem()
    problem.discretise_domain(n=10, mode="grid")
    dm = PinaDataModule(problem)
    dm.setup()
    assert hasattr(dm, "train_datasets")
    assert isinstance(dm.train_datasets, dict)
    assert set(dm.train_datasets.keys()) == {"D", "boundary"}
    assert isinstance(dm.train_datasets["D"], _ConditionSubset)
    assert isinstance(dm.train_datasets["boundary"], _ConditionSubset)
    assert hasattr(dm, "val_datasets")
    assert isinstance(dm.val_datasets, dict)
    assert set(dm.val_datasets.keys()) == {"D", "boundary"}
    assert isinstance(dm.val_datasets["D"], _ConditionSubset)
    assert isinstance(dm.val_datasets["boundary"], _ConditionSubset)
    assert hasattr(dm, "test_datasets")
    assert isinstance(dm.test_datasets, dict)
    assert set(dm.test_datasets.keys()) == {"D", "boundary"}
    assert isinstance(dm.test_datasets["D"], _ConditionSubset)
    assert isinstance(dm.test_datasets["boundary"], _ConditionSubset)


@pytest.mark.parametrize("batch_size", [None, 1, 5, 20])
def test_dataloader_tensor(batch_size):
    input_tensor, output_tensor = _create_tensor_data()
    problem = SupervisedProblem(input_=input_tensor, output_=output_tensor)
    trainer = Trainer(
        solver=SupervisedSolver(problem=problem, model=torch.nn.Linear(10, 10)),
        batch_size=batch_size,
        train_size=0.7,
        val_size=0.2,
        test_size=0.1,
    )
    dm = trainer.data_module
    dm.setup()
    dataloader = dm.train_dataloader()
    assert isinstance(dataloader, _Aggregator)
    data = next(iter(dataloader))
    assert isinstance(data, dict)
    assert isinstance(data["data"]["input"], torch.Tensor)
    assert isinstance(data["data"]["target"], torch.Tensor)
    assert (
        len(data["data"]["input"]) == batch_size
        if batch_size is not None
        else 70
    )

    dataloader = dm.val_dataloader()
    assert isinstance(dataloader, _Aggregator)
    data = next(iter(dataloader))
    assert isinstance(data, dict)
    assert isinstance(data["data"]["input"], torch.Tensor)
    assert isinstance(data["data"]["target"], torch.Tensor)
    assert (
        len(data["data"]["input"]) == batch_size
        if batch_size is not None
        else 10
    )


@pytest.mark.parametrize("batch_size", [None, 1, 5, 20])
def test_dataloader_graph(batch_size):
    input_graph, output_graph = _create_graph_data()
    problem = SupervisedProblem(input_=input_graph, output_=output_graph)
    trainer = Trainer(
        solver=SupervisedSolver(problem=problem, model=torch.nn.Linear(10, 10)),
        train_size=0.7,
        val_size=0.2,
        test_size=0.1,
        batch_size=batch_size,
    )
    dm = trainer.data_module
    dm.setup()
    dataloader = dm.train_dataloader()
    assert isinstance(dataloader, _Aggregator)
    data = next(iter(dataloader))
    assert isinstance(data, dict)
    assert isinstance(data["data"]["input"], Batch)
    assert isinstance(data["data"]["target"], torch.Tensor)
    assert (
        len(data["data"]["input"]) == batch_size
        if batch_size is not None
        else 70
    )

    dataloader = dm.val_dataloader()
    assert isinstance(dataloader, _Aggregator)
    data = next(iter(dataloader))
    assert isinstance(data, dict)
    assert isinstance(data["data"]["input"], Batch)
    assert isinstance(data["data"]["target"], torch.Tensor)
    assert (
        len(data["data"]["input"]) == batch_size
        if batch_size is not None
        else 10
    )


@pytest.mark.parametrize("batch_size", [None, 1, 5, 20])
def test_dataloader_poisson_cbs(batch_size):
    problem = Poisson2DSquareProblem()
    problem.discretise_domain(n=10, mode="grid")
    trainer = Trainer(
        solver=PINN(problem=problem, model=torch.nn.Linear(10, 10)),
        batch_size=batch_size,
        val_size=0.1,
        test_size=0.2,
        train_size=0.7,
        batching_mode="common_batch_size",
    )
    dm = trainer.data_module
    dm.setup()

    dataloader = dm.train_dataloader()
    assert isinstance(dataloader, _Aggregator)
    data = next(iter(dataloader))
    assert isinstance(data, dict)
    assert isinstance(data["D"]["input"], torch.Tensor)
    assert isinstance(data["D"]["input"], torch.Tensor)
    assert isinstance(data["boundary"]["input"], torch.Tensor)
    assert isinstance(data["boundary"]["input"], torch.Tensor)
    assert (
        len(data["D"]["input"]) == batch_size if batch_size is not None else 70
    )
    assert (
        len(data["boundary"]["input"]) == min(batch_size, 7)
        if batch_size is not None
        else 7
    )

    dataloader = dm.val_dataloader()
    assert isinstance(dataloader, _Aggregator)
    data = next(iter(dataloader))
    assert isinstance(data, dict)
    assert isinstance(data["D"]["input"], torch.Tensor)
    assert isinstance(data["D"]["input"], torch.Tensor)
    assert isinstance(data["boundary"]["input"], torch.Tensor)
    assert isinstance(data["boundary"]["input"], torch.Tensor)
    assert (
        len(data["D"]["input"]) == min(batch_size, 10)
        if batch_size is not None
        else 10
    )
    assert (
        len(data["boundary"]["input"]) == min(batch_size, 1)
        if batch_size is not None
        else 1
    )


@pytest.mark.parametrize("batch_size", [None, 5, 20])
def test_dataloader_poisson_proportional(batch_size):
    problem = Poisson2DSquareProblem()
    problem.discretise_domain(n=10, mode="grid")
    trainer = Trainer(
        solver=PINN(problem=problem, model=torch.nn.Linear(10, 10)),
        batch_size=batch_size,
        val_size=0.1,
        test_size=0.2,
        train_size=0.7,
        batching_mode="proportional",
    )
    dm = trainer.data_module
    dm.setup()

    dataloader = dm.train_dataloader()
    assert isinstance(dataloader, _Aggregator)
    data = next(iter(dataloader))
    assert isinstance(data, dict)
    assert isinstance(data["D"]["input"], torch.Tensor)
    assert isinstance(data["D"]["input"], torch.Tensor)
    assert isinstance(data["boundary"]["input"], torch.Tensor)
    assert isinstance(data["boundary"]["input"], torch.Tensor)
    assert (
        len(data["D"]["input"]) == batch_size - 1
        if batch_size is not None
        else 70
    )
    assert len(data["boundary"]["input"]) == 1 if batch_size is not None else 7
