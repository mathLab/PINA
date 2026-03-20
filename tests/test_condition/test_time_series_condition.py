import pytest
import torch

from pina.condition import TimeSeriesCondition


class DummySolver:
    def __init__(self):
        self.weight_calls = []

    def preprocess_step(self, current_state, **kwargs):
        return current_state

    def forward(self, x):
        return x + 1.0

    def postprocess_step(self, predicted_state, **kwargs):
        return predicted_state

    def _get_weights(self, condition_name, step_losses, eps):
        self.weight_calls.append((condition_name, eps, step_losses.shape))
        return torch.ones_like(step_losses)


def test_evaluate_time_series_condition_mean_aggregation():
    input_tensor = torch.tensor([[[[0.0], [1.0], [2.0]]]])
    condition = TimeSeriesCondition(input=input_tensor, eps=0.1)
    solver = DummySolver()
    loss = torch.nn.MSELoss(reduction="none")

    value = condition.evaluate(
        {"input": input_tensor},
        solver,
        loss,
        condition_name="autoregressive",
    )

    torch.testing.assert_close(value, torch.tensor(0.0))
    assert solver.weight_calls == [
        ("autoregressive", 0.1, torch.Size([2, 1, 1, 1]))
    ]


def test_evaluate_time_series_condition_invalid_shape():
    input_tensor = torch.randn(2, 3, 4)
    condition = TimeSeriesCondition(input=input_tensor)
    solver = DummySolver()
    loss = torch.nn.MSELoss(reduction="none")

    with pytest.raises(ValueError, match="at least 4 dimensions"):
        condition.evaluate({"input": input_tensor}, solver, loss)
