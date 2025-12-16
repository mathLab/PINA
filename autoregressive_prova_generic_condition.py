import torch
import matplotlib.pyplot as plt

from pina import Trainer
from pina.optim import TorchOptimizer
from pina.problem import AbstractProblem
from pina.condition.data_condition import DataCondition
from pina.solver import AutoregressiveSolver

NUM_TIMESTEPS = 100
NUM_FEATURES = 15
USE_TEST_MODEL = False

# ============================================================================
# DATA
# ============================================================================

torch.manual_seed(42)

y = torch.zeros(NUM_TIMESTEPS, NUM_FEATURES)
y[0] = torch.rand(NUM_FEATURES)  # Random initial state

for t in range(NUM_TIMESTEPS - 1):
    y[t + 1] = 0.95 * y[t]  # + 0.05 * torch.sin(y[t].sum())

# ============================================================================
# TRAINING
# ============================================================================

class SimpleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(y.shape[1], 20),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(20, y.shape[1]),
        )

    def forward(self, x):
        return x + self.layers(x)


class TestModel(torch.nn.Module):
    """
    Debug model that implements the EXACT transformation rule.
    y[t+1] = 0.95 * y[t]
    Expected loss is zero
    """

    def __init__(self, data_series=None):
        super().__init__()
        self.dummy_param = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x):
        next_state = 0.95 * x  # + 0.05 * torch.sin(x.sum(dim=1, keepdim=True))
        return next_state + 0.0 * self.dummy_param


class Problem(AbstractProblem):
    output_variables = None
    input_variables = None
    conditions = {
        "data_condition_0":DataCondition(input=y),
        "data_condition_1":DataCondition(input=y),
    }

problem = Problem()

#for each condition, define unroll instructions with these keys:
#   - unroll_length: length of each unroll window
#   - num_unrolls: number of unroll windows to create (if None, use all possible)
#   - randomize: whether to randomize the starting indices of the unroll windows
unroll_instructions = {
    "data_condition_0": {
        "unroll_length": 10,
        "num_unrolls": 89,
        "randomize": True,
        "eps": 5.0
    },
    "data_condition_1": {
        "unroll_length": 20,
        "num_unrolls": 79,
        "randomize": True,
        "eps": 10.0
    },
}

solver = AutoregressiveSolver(
    unroll_instructions=unroll_instructions,
    problem=problem,
    model=TestModel() if USE_TEST_MODEL else SimpleModel(),
    optimizer= TorchOptimizer(torch.optim.AdamW, lr=0.01),
    eps=10.0,
)

trainer = Trainer(
    solver, max_epochs=2000, accelerator="cpu", enable_model_summary=False, shuffle=False
)
trainer.train()

# ============================================================================
# VISUALIZATION
# ============================================================================

test_start_idx = 50
num_prediction_steps = 30

initial_state = y[test_start_idx]  # Shape: [features]
predictions = solver.predict(initial_state, num_prediction_steps)
actual = y[test_start_idx : test_start_idx + num_prediction_steps + 1]

total_mse = torch.nn.functional.mse_loss(predictions[1:], actual[1:])
print(f"\nOverall MSE (all {num_prediction_steps} steps): {total_mse:.6f}")

# viauzlize single dof
dof_to_plot = [0, 3, 6, 9, 12]
colors = [
    "r",
    "g",
    "b",
    "c",
    "m",
    "y",
    "k",
]
plt.figure(figsize=(10, 6))
for dof, color in zip(dof_to_plot, colors):
    plt.plot(
        range(test_start_idx, test_start_idx + num_prediction_steps + 1),
        actual[:, dof].numpy(),
        label="Actual",
        marker="o",
        color=color,
        markerfacecolor="none",
    )
    plt.plot(
        range(test_start_idx, test_start_idx + num_prediction_steps + 1),
        predictions[:, dof].numpy(),
        label="Predicted",
        marker="x",
        color=color,
    )

plt.title(f"Autoregressive Predictions vs Actual, MRSE: {total_mse:.6f}")
plt.legend()
plt.xlabel("Timestep")
plt.savefig(f"autoregressive_predictions.png")
plt.close()
