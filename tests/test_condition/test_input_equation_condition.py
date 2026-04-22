import torch
import pytest
from pina.equation import Equation
from pina import LabelTensor, Condition
from pina.graph import RadiusGraph, Graph
from pina.condition import (
    InputEquationCondition,
    _TensorDataManager,
    _GraphDataManager,
    _BatchManager,
)

# Generate input and equation data for testing - tensor case
input_tensor = LabelTensor(torch.rand((10, 2)), ["x", "y"])
equation_tensor = Equation(lambda pts: pts["x"] ** 2 + pts["y"] ** 2 - 1)

# Generate input and equation data for testing - graph case
input_graph_list = [
    RadiusGraph(
        x=LabelTensor(torch.rand(10, 2), labels=["u", "v"]),
        pos=LabelTensor(torch.rand(10, 2), labels=["x", "y"]),
        radius=0.1,
        edge_attr=True,
    )
    for _ in range(3)
]
equation_graph = Equation(lambda pts: pts.x["u"] ** 2 + pts.x["v"] ** 2 - 1)


@pytest.mark.parametrize("case", ["tensor", "graph"])
def test_constructor(case):

    # Tensor case
    if case == "tensor":
        input_, equation = input_tensor, equation_tensor

    # Graph case
    elif case == "graph":
        input_, equation = input_graph_list, equation_graph

    # Define the condition
    condition = Condition(input=input_, equation=equation)

    # Assert correct types
    assert isinstance(condition, InputEquationCondition)

    # Assert that the equation is stored correctly
    assert condition.equation is equation

    # Assert correct input type
    if case == "tensor":
        assert isinstance(condition.input, LabelTensor)
    elif case == "graph":
        assert isinstance(condition.input, list)
        for graph in condition.input:
            assert isinstance(graph, Graph)

    # Should fail if input is not an instance of LabelTensor or Graph
    with pytest.raises(ValueError):
        Condition(input=torch.rand(10, 2), equation=equation)

    # Should fail if equation is not an instance of BaseEquation
    with pytest.raises(ValueError):
        Condition(input=input_, equation="not_an_equation")

    # Should fail if input is a list with wrong elements
    with pytest.raises(ValueError):
        Condition(
            input=[LabelTensor(torch.rand(10, 2), ["x", "y"])],
            equation=equation,
        )


@pytest.mark.parametrize("case", ["tensor", "graph"])
def test_get_item(case):

    # Tensor case
    if case == "tensor":
        input_, equation = input_tensor, equation_tensor

    # Graph case
    elif case == "graph":
        input_, equation = input_graph_list, equation_graph

    # Define the condition
    condition = Condition(input=input_, equation=equation)

    # Extract item using __getitem__
    index = 0
    item = condition[index]

    # Assert correct types and numerical parity
    if case == "tensor":
        assert isinstance(item, _TensorDataManager)
        assert isinstance(item.input, LabelTensor)
        assert torch.allclose(item.input, input_[index])

    elif case == "graph":
        assert isinstance(item, _GraphDataManager)
        assert isinstance(item.input, Graph)
        assert torch.allclose(item.input.x, input_[index].x)


@pytest.mark.parametrize("case", ["tensor", "graph"])
def test_create_batch(case):

    # Tensor case
    if case == "tensor":
        input_, equation = input_tensor, equation_tensor

    # Graph case
    elif case == "graph":
        input_, equation = input_graph_list, equation_graph

    # Define the condition
    condition = Condition(input=input_, equation=equation)

    # Create batches using automatic batching or condition's collate_fn
    idx = [0, 2]
    data_to_collate = [condition.data[i] for i in idx]
    batch_auto = condition.automatic_batching_collate_fn(data_to_collate)
    batch_collate = condition.collate_fn(idx, condition)

    # Check that the automatic batch has been properly created
    assert isinstance(batch_auto, (_BatchManager))
    assert hasattr(batch_auto, "input")

    # Check that the collate_fn batch has been properly created
    assert isinstance(batch_collate, dict)
    assert hasattr(batch_collate, "input")

    # Validate batch contents for tensor case
    if case == "tensor":

        # Create expected input batch
        expected_input = LabelTensor.stack([input_[i] for i in idx])

        # Assert that the automatic batch input is correct
        assert torch.allclose(batch_auto.input, expected_input)
        assert batch_auto.input.shape == expected_input.shape

        # Assert that the collate_fn batch input is correct
        assert torch.allclose(batch_collate.input, expected_input)
        assert batch_collate.input.shape == expected_input.shape

    # Validate batch contents for graph case
    elif case == "graph":

        # Create expected input batch
        expected_input = [condition.data[i].input for i in idx]

        # Assert that the automatic batch input is correct
        for i, graph in enumerate(expected_input):
            assert torch.allclose(batch_auto.input[i].x, graph.x)
        assert batch_auto.input.num_graphs == len(idx)

        # Assert that the collate_fn batch input is correct
        for i, graph in enumerate(expected_input):
            assert torch.allclose(batch_collate.input[i].x, graph.x)
        assert batch_collate.input.num_graphs == len(idx)
