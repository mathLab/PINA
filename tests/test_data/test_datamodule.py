import torch
import pytest
from pina.data import PinaDataModule
from pina.problem.zoo import SupervisedProblem
from pina.graph import RadiusGraph

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
    ]
)
def test_setup(input_, output_):
    problem = SupervisedProblem(input_=input_, output_=output_)
    dm = PinaDataModule(problem)
    dm.setup()
    assert hasattr(dm, "train_dataset")
    assert hasattr(dm, "test_dataset")
    assert hasattr(dm, "val_dataset")
    

