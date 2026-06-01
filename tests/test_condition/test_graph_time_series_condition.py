import pytest
import torch
from pina.data.manager import _TensorDataManager, _BatchManager
from pina._src.core.utils import labelize_forward
from pina.condition import TimeSeriesCondition
from pina import LabelTensor, Condition
from pina._src.condition.graph_time_series_condition import GraphTimeSeriesCondition
from pina.graph import RadiusGraph

# Number of samples and time steps for testing
n_samples = 5
n_graphs = 10
n_nodes = 20
time_steps = 10


# Helper function to check tensor types
def _assert_tensor_type(t, use_lt):
    if use_lt:
        assert isinstance(t, LabelTensor)
    else:
        assert isinstance(t, torch.Tensor) and not isinstance(t, LabelTensor)


# Helper function to compute expected unroll windows
def _expected_unroll(data, n_windows, unroll_length, randomize):

    # Compute valid starting indices
    last_idx = data.shape[1] - unroll_length
    start_indices = torch.arange(last_idx + 1)

    # Randomize indices if required
    if randomize:
        start_indices = start_indices[torch.randperm(len(start_indices))]

    # Limit the number of windows
    if n_windows is not None and n_windows < len(start_indices):
        start_indices = start_indices[:n_windows]

    # Build expected windows
    windows = [data[:, s : s + unroll_length] for s in start_indices]

    return torch.stack(windows, dim=1)

# Helper function to create graph data 
def _create_graph_data(is_input, use_lt):

    # If LabelTensor is used, create graph data with LabelTensors
    if use_lt:
        x = LabelTensor(torch.rand(n_graphs, n_nodes, 2), ["u", "v"])
        pos = LabelTensor(torch.rand(n_graphs, n_nodes, 2), ["x", "y"])
        tensor = LabelTensor(torch.rand(n_graphs, n_nodes, 2), ["f", "g"])

    # Standard torch.Tensor without labels
    else:
        x = torch.rand(n_graphs, n_nodes, 2)
        pos = torch.rand(n_graphs, n_nodes, 2)
        tensor = torch.rand(n_graphs, n_nodes, 2)

    # Create a list of Graphs
    graph = [
        RadiusGraph(
            pos=pos[i],
            radius=0.1,
            x=x[i] if is_input else None,
            y=x[i] if not is_input else None,
        )
        for i in range(len(x))
    ]

    return graph, tensor


# Define a dummy solver for testing
class DummySolver:

    def __init__(self, use_lt, input_vars):
        if use_lt:
            self.forward = labelize_forward(
                forward=self.forward,
                input_variables=input_vars,
                output_variables=input_vars,
            )

        self._params = None
        self._kwargs = {}
        self.aggregation_strategy = torch.mean

    def forward(self, samples):
        return samples

    def preprocess_step(self, current_state, **kwargs):
        return current_state

    def postprocess_step(self, predicted_state, **kwargs):
        return predicted_state

    def _get_weights(self, condition_name, step_losses):
        return 1.0


@pytest.mark.parametrize("use_lt", [True, False])
@pytest.mark.parametrize("n_windows", [4, 6])
@pytest.mark.parametrize("unroll_length", [3, 5])
@pytest.mark.parametrize("randomize", [True, False])
def test_constructor(use_lt, n_windows, unroll_length, randomize):

    # Define the condition
    input_tensor, _ = _create_graph_data(is_input=True, use_lt=use_lt)
    condition = GraphTimeSeriesCondition(
        input=input_tensor,
        n_windows=n_windows,
        unroll_length=unroll_length,
        randomize=randomize,
    )

    # Assert correct types
    assert isinstance(condition, TimeSeriesCondition)
    # _assert_tensor_type(condition.input, use_lt)

    # Assert numerical parity
    if not randomize:
        expected_tensor = _expected_unroll(
            input_tensor, n_windows, unroll_length, randomize
        )
        assert torch.allclose(condition.input, expected_tensor)

    # Assert labels if LabelTensor is used
    if use_lt:
        assert condition.input.labels == ["u", "v"]

    # Should fail if unroll_length is not a positive integer
    with pytest.raises(AssertionError):
        Condition(
            input=input_tensor,
            n_windows=n_windows,
            unroll_length=0,
            randomize=randomize,
        )

    # Should fail if n_windows is not a positive integer
    with pytest.raises(AssertionError):
        Condition(
            input=input_tensor,
            n_windows=0,
            unroll_length=unroll_length,
            randomize=randomize,
        )

    # Should fail if randomize is not a boolean value
    with pytest.raises(ValueError):
        Condition(
            input=input_tensor,
            n_windows=n_windows,
            unroll_length=unroll_length,
            randomize="not_a_boolean",
        )

    # Should fail if the input tensor has less than 3 dimensions
    with pytest.raises(ValueError):
        Condition(
            input=torch.rand(n_samples, 2),
            n_windows=n_windows,
            unroll_length=unroll_length,
            randomize=randomize,
        )

    # Should fail if unroll_length is not greater than 1
    with pytest.raises(ValueError):
        Condition(
            input=input_tensor,
            n_windows=n_windows,
            unroll_length=1,
            randomize=randomize,
        )

    # Should fail if unroll_length is greater than the number of time steps
    with pytest.raises(ValueError):
        Condition(
            input=input_tensor,
            n_windows=n_windows,
            unroll_length=time_steps + 1,
            randomize=randomize,
        )

    # Should fail if n_windows is greater than the number of valid windows
    with pytest.raises(ValueError):
        Condition(
            input=input_tensor,
            n_windows=10,
            unroll_length=unroll_length,
            randomize=randomize,
        )


@pytest.mark.parametrize("use_lt", [True, False])
@pytest.mark.parametrize("n_windows", [4, 6])
@pytest.mark.parametrize("unroll_length", [3, 5])
@pytest.mark.parametrize("randomize", [True, False])
def test_get_item(use_lt, n_windows, unroll_length, randomize):

    # Define the condition
    input_tensor = _create_tensor_data(use_lt)
    condition = Condition(
        input=input_tensor,
        n_windows=n_windows,
        unroll_length=unroll_length,
        randomize=randomize,
    )

    # Extract item using __getitem__
    index = 0
    item = condition[index]

    # Assert correct types
    assert isinstance(item, _TensorDataManager)
    _assert_tensor_type(item.input, use_lt)

    # Assert correct shapes
    expected_shape = torch.Size([n_windows, unroll_length, 2])
    assert item.input.shape == expected_shape

    # Assert numerical parity
    if not randomize:
        expected_tensor = _expected_unroll(
            input_tensor, n_windows, unroll_length, randomize
        )
        assert torch.allclose(item.input, expected_tensor[index])


@pytest.mark.parametrize("use_lt", [True, False])
@pytest.mark.parametrize("n_windows", [4, 6])
@pytest.mark.parametrize("unroll_length", [3, 5])
@pytest.mark.parametrize("randomize", [True, False])
def test_create_batch(use_lt, n_windows, unroll_length, randomize):

    # Define the condition
    input_tensor = _create_tensor_data(use_lt)
    condition = Condition(
        input=input_tensor,
        n_windows=n_windows,
        unroll_length=unroll_length,
        randomize=randomize,
    )

    # Create batches using automatic batching or condition's collate_fn
    idx = [0, 2]
    data_to_collate = [condition.data[i] for i in idx]
    batch_auto = condition.automatic_batching_collate_fn(data_to_collate)
    batch_collate = condition.collate_fn(idx, condition)

    # Check that the automatic batch has been properly created
    assert isinstance(batch_auto, _BatchManager)
    assert hasattr(batch_auto, "input")

    # Check that the collate_fn batch has been properly created
    assert isinstance(batch_collate, dict)
    assert hasattr(batch_collate, "input")

    # Assert that the automatic batch input is correct
    expected_shape = torch.Size([len(idx), n_windows, unroll_length, 2])
    assert batch_auto.input.shape == expected_shape

    # Assert that the collate_fn batch input is correct
    expected_shape = torch.Size([len(idx), n_windows, unroll_length, 2])
    assert batch_collate.input.shape == expected_shape

    # Create input values
    if not randomize:
        expected_tensor = _expected_unroll(
            input_tensor, n_windows, unroll_length, randomize
        )
        assert torch.allclose(batch_collate.input, expected_tensor[idx])
        assert torch.allclose(batch_auto.input, expected_tensor[idx])


@pytest.mark.parametrize("use_lt", [True, False])
@pytest.mark.parametrize("n_windows", [4, 6])
@pytest.mark.parametrize("unroll_length", [3, 5])
@pytest.mark.parametrize("randomize", [True, False])
def test_evaluate(use_lt, n_windows, unroll_length, randomize):

    # Define the input tensor
    input_tensor = _create_tensor_data(use_lt)
    input_vars = input_tensor.labels if use_lt else None

    # Define the condition and the solver
    condition = Condition(
        input=input_tensor,
        n_windows=n_windows,
        unroll_length=unroll_length,
        randomize=randomize,
    )
    solver = DummySolver(use_lt, input_vars)
    loss_fn = torch.nn.MSELoss(reduction="none")

    # Extract the batch
    batch = {"input": condition.input}

    # Evaluate the condition and compute the expected residuals
    residuals = condition.evaluate(batch, solver)

    # Compute expected autoregressive step residuals
    step_residuals = []
    current_state = batch["input"][:, :, 0]

    for step in range(1, batch["input"].shape[2]):
        predicted_state = current_state
        target_state = batch["input"][:, :, step]

        step_residual = predicted_state - target_state
        step_residuals.append(step_residual)

        current_state = predicted_state

    expected = torch.stack(step_residuals).as_subclass(torch.Tensor)

    # Assert that the evaluated residuals are correct
    assert torch.allclose(residuals, expected)
