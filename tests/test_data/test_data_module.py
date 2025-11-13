import torch
import pytest
from pina.data import PinaDataModule
from pina.data.dataset import PinaDataset
from pina.problem.zoo import SupervisedProblem
from pina.graph import RadiusGraph

from pina.data.dataloader import DummyDataloader, PinaDataLoader
from pina import Trainer
from pina.solver import SupervisedSolver
from torch_geometric.data import Batch
from torch.utils.data import DataLoader

input_tensor = torch.rand((100, 10))
output_tensor = torch.rand((100, 2))

x = torch.rand((100, 50, 10))
pos = torch.rand((100, 50, 2))
input_graph = [
    RadiusGraph(x=x_, pos=pos_, radius=0.2) for x_, pos_, in zip(x, pos)
]
output_graph = torch.rand((100, 50, 10))


@pytest.mark.parametrize(
    "input_, output_",
    [(input_tensor, output_tensor), (input_graph, output_graph)],
)
def test_constructor(input_, output_):
    problem = SupervisedProblem(input_=input_, output_=output_)
    PinaDataModule(problem)


@pytest.mark.parametrize(
    "input_, output_",
    [(input_tensor, output_tensor), (input_graph, output_graph)],
)
@pytest.mark.parametrize(
    "train_size, val_size, test_size", [(0.7, 0.2, 0.1), (0.7, 0.3, 0)]
)
def test_setup_train(input_, output_, train_size, val_size, test_size):
    problem = SupervisedProblem(input_=input_, output_=output_)
    dm = PinaDataModule(
        problem, train_size=train_size, val_size=val_size, test_size=test_size
    )
    dm.setup()
    assert hasattr(dm, "train_dataset")
    assert isinstance(dm.train_dataset, dict)
    assert all(
        isinstance(dm.train_dataset[cond], PinaDataset)
        for cond in dm.train_dataset
    )
    assert all(
        dm.train_dataset[cond].is_graph_dataset == isinstance(input_, list)
        for cond in dm.train_dataset
    )
    assert all(
        len(dm.train_dataset[cond]) == int(len(input_) * train_size)
        for cond in dm.train_dataset
    )
    if test_size > 0:
        assert hasattr(dm, "test_dataset")
        assert dm.test_dataset is None
    else:
        assert not hasattr(dm, "test_dataset")
    assert hasattr(dm, "val_dataset")

    assert isinstance(dm.val_dataset, dict)
    assert all(
        isinstance(dm.val_dataset[cond], PinaDataset) for cond in dm.val_dataset
    )
    assert all(
        isinstance(dm.val_dataset[cond], PinaDataset) for cond in dm.val_dataset
    )


@pytest.mark.parametrize(
    "input_, output_",
    [(input_tensor, output_tensor), (input_graph, output_graph)],
)
@pytest.mark.parametrize(
    "train_size, val_size, test_size", [(0.7, 0.2, 0.1), (0.0, 0.0, 1.0)]
)
def test_setup_test(input_, output_, train_size, val_size, test_size):
    problem = SupervisedProblem(input_=input_, output_=output_)
    dm = PinaDataModule(
        problem, train_size=train_size, val_size=val_size, test_size=test_size
    )
    dm.setup(stage="test")
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
    assert all(
        isinstance(dm.test_dataset[cond], PinaDataset)
        for cond in dm.test_dataset
    )
    assert all(
        dm.test_dataset[cond].is_graph_dataset == isinstance(input_, list)
        for cond in dm.test_dataset
    )
    assert all(
        len(dm.test_dataset[cond]) == int(len(input_) * test_size)
        for cond in dm.test_dataset
    )


# @pytest.mark.parametrize(
#     "input_, output_",
#     [(input_tensor, output_tensor), (input_graph, output_graph)],
# )
# def test_dummy_dataloader(input_, output_):
#     problem = SupervisedProblem(input_=input_, output_=output_)
#     solver = SupervisedSolver(problem=problem, model=torch.nn.Linear(10, 10))
#     trainer = Trainer(
#         solver, batch_size=None, train_size=0.7, val_size=0.3, test_size=0.0
#     )
#     dm = trainer.data_module
#     dm.setup()
#     dm.trainer = trainer
#     dataloader = dm.train_dataloader()
#     assert isinstance(dataloader, PinaDataLoader)
#     print(dataloader.dataloaders)
#     assert all([isinstance(ds, DummyDataloader) for ds in dataloader.dataloaders.values()])

#     data = next(iter(dataloader))
#     assert isinstance(data, list)
#     assert isinstance(data[0], tuple)
#     if isinstance(input_, list):
#         assert isinstance(data[0][1]["input"], Batch)
#     else:
#         assert isinstance(data[0][1]["input"], torch.Tensor)
#     assert isinstance(data[0][1]["target"], torch.Tensor)


# dataloader = dm.val_dataloader()
# assert isinstance(dataloader, DummyDataloader)
# assert len(dataloader) == 1
# data = next(dataloader)
# assert isinstance(data, list)
# assert isinstance(data[0], tuple)
# if isinstance(input_, list):
#     assert isinstance(data[0][1]["input"], Batch)
# else:
#     assert isinstance(data[0][1]["input"], torch.Tensor)
# assert isinstance(data[0][1]["target"], torch.Tensor)


@pytest.mark.parametrize(
    "input_, output_",
    [(input_tensor, output_tensor), (input_graph, output_graph)],
)
@pytest.mark.parametrize("automatic_batching", [True, False])
def test_dataloader(input_, output_, automatic_batching):
    problem = SupervisedProblem(input_=input_, output_=output_)
    solver = SupervisedSolver(problem=problem, model=torch.nn.Linear(10, 10))
    trainer = Trainer(
        solver,
        batch_size=10,
        train_size=0.7,
        val_size=0.3,
        test_size=0.0,
        automatic_batching=automatic_batching,
        common_batch_size=True,
    )
    dm = trainer.data_module
    dm.setup()
    dm.trainer = trainer
    dataloader = dm.train_dataloader()
    assert isinstance(dataloader, PinaDataLoader)
    assert len(dataloader) == 7
    data = next(iter(dataloader))
    assert isinstance(data, dict)
    if isinstance(input_, list):
        assert isinstance(data["data"]["input"], Batch)
    else:
        assert isinstance(data["data"]["input"], torch.Tensor)
    assert isinstance(data["data"]["target"], torch.Tensor)

    dataloader = dm.val_dataloader()
    assert isinstance(dataloader, PinaDataLoader)
    assert len(dataloader) == 3
    data = next(iter(dataloader))
    assert isinstance(data, dict)
    if isinstance(input_, list):
        assert isinstance(data["data"]["input"], Batch)
    else:
        assert isinstance(data["data"]["input"], torch.Tensor)
    assert isinstance(data["data"]["target"], torch.Tensor)


from pina import LabelTensor

input_tensor = LabelTensor(torch.rand((100, 3)), ["u", "v", "w"])
output_tensor = LabelTensor(torch.rand((100, 3)), ["u", "v", "w"])

x = LabelTensor(torch.rand((100, 50, 3)), ["u", "v", "w"])
pos = LabelTensor(torch.rand((100, 50, 2)), ["x", "y"])
input_graph = [
    RadiusGraph(x=x[i], pos=pos[i], radius=0.1) for i in range(len(x))
]
output_graph = LabelTensor(torch.rand((100, 50, 3)), ["u", "v", "w"])


@pytest.mark.parametrize(
    "input_, output_",
    [(input_tensor, output_tensor), (input_graph, output_graph)],
)
@pytest.mark.parametrize("automatic_batching", [True, False])
def test_dataloader_labels(input_, output_, automatic_batching):
    problem = SupervisedProblem(input_=input_, output_=output_)
    solver = SupervisedSolver(problem=problem, model=torch.nn.Linear(10, 10))
    trainer = Trainer(
        solver,
        batch_size=10,
        train_size=0.7,
        val_size=0.3,
        test_size=0.0,
        automatic_batching=automatic_batching,
        common_batch_size=True,
    )
    dm = trainer.data_module
    dm.setup()
    dm.trainer = trainer
    dataloader = dm.train_dataloader()
    assert isinstance(dataloader, PinaDataLoader)
    assert len(dataloader) == 7
    data = next(iter(dataloader))
    assert isinstance(data, dict)
    if isinstance(input_, list):
        assert isinstance(data["data"]["input"], Batch)
        assert isinstance(data["data"]["input"].x, LabelTensor)
        assert data["data"]["input"].x.labels == ["u", "v", "w"]
        assert data["data"]["input"].pos.labels == ["x", "y"]
    else:
        assert isinstance(data["data"]["input"], LabelTensor)
        assert data["data"]["input"].labels == ["u", "v", "w"]
    assert isinstance(data["data"]["target"], LabelTensor)
    assert data["data"]["target"].labels == ["u", "v", "w"]

    dataloader = dm.val_dataloader()
    assert isinstance(dataloader, PinaDataLoader)
    assert len(dataloader) == 3
    data = next(iter(dataloader))
    assert isinstance(data, dict)
    if isinstance(input_, list):
        assert isinstance(data["data"]["input"], Batch)
        assert isinstance(data["data"]["input"].x, LabelTensor)
        assert data["data"]["input"].x.labels == ["u", "v", "w"]
        assert data["data"]["input"].pos.labels == ["x", "y"]
    else:
        assert isinstance(data["data"]["input"], torch.Tensor)
        assert isinstance(data["data"]["input"], LabelTensor)
        assert data["data"]["input"].labels == ["u", "v", "w"]
    assert isinstance(data["data"]["target"], torch.Tensor)
    assert data["data"]["target"].labels == ["u", "v", "w"]


def test_input_propery_tensor():
    input = torch.stack([torch.zeros((1,)) + i for i in range(1000)])
    target = input

    problem = SupervisedProblem(input, target)
    datamodule = PinaDataModule(
        problem,
        train_size=0.7,
        test_size=0.2,
        val_size=0.1,
        batch_size=64,
        shuffle=False,
        automatic_batching=None,
        num_workers=0,
        pin_memory=False,
    )
    datamodule.setup("fit")
    datamodule.setup("test")
    input_ = datamodule.input
    assert isinstance(input_, dict)
    assert isinstance(input_["train"], dict)
    assert isinstance(input_["val"], dict)
    assert isinstance(input_["test"], dict)
    assert torch.isclose(input_["train"]["data"], input[:700]).all()
    assert torch.isclose(input_["val"]["data"], input[900:]).all()
    assert torch.isclose(input_["test"]["data"], input[700:900]).all()


def test_input_propery_graph():
    problem = SupervisedProblem(input_graph, output_graph)
    datamodule = PinaDataModule(
        problem,
        train_size=0.7,
        test_size=0.2,
        val_size=0.1,
        batch_size=64,
        shuffle=False,
        automatic_batching=None,
        num_workers=0,
        pin_memory=False,
    )
    datamodule.setup("fit")
    datamodule.setup("test")
    input_ = datamodule.input
    assert isinstance(input_, dict)
    assert isinstance(input_["train"], dict)
    assert isinstance(input_["val"], dict)
    assert isinstance(input_["test"], dict)
    assert isinstance(input_["train"]["data"], list)
    assert isinstance(input_["val"]["data"], list)
    assert isinstance(input_["test"]["data"], list)
    assert len(input_["train"]["data"]) == 70
    assert len(input_["val"]["data"]) == 10
    assert len(input_["test"]["data"]) == 20
