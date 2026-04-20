import shutil
import pytest
import torch
from torch._dynamo.eval_frame import OptimizedModule

from pina import Condition, Trainer, LabelTensor
from pina.solver import AutoregressiveSolver
from pina.condition import DataCondition
from pina.problem import BaseProblem
from pina.model import FeedForward


# Hyperparameters and settings
n_traj = 5
t_steps = 10
n_feats = 2
unroll_length = 3
n_unrolls = 4


# TODO: test this in AutoregressiveCondition once it's implemented
# Utility function to create synthetic data for testing
def create_data(n_traj, t_steps, n_feats, unroll_length, n_unrolls, use_lt):

    init_state = torch.rand(n_traj, n_feats)
    traj = torch.stack([0.95**i * init_state for i in range(t_steps)], dim=1)

    data = AutoregressiveSolver.unroll(
        data=traj,
        unroll_length=unroll_length,
        n_unrolls=n_unrolls,
        randomize=True,
    )
    labels = [f"feat_{i}" for i in range(n_feats)]
    return LabelTensor(data, labels=labels)


# Data
data = create_data(
    n_traj=n_traj,
    t_steps=t_steps,
    n_feats=n_feats,
    unroll_length=unroll_length,
    n_unrolls=n_unrolls,
    use_lt=True,
)


# Problem
class Problem(BaseProblem):

    input_variables = [f"feat_{i}" for i in range(n_feats)]
    output_variables = [f"feat_{i}" for i in range(n_feats)]
    conditions = {}

    def __init__(self, data):
        super().__init__()
        self.data = data
        self.conditions = {"autoregressive": Condition(input=self.data)}
        self.conditions_settings = {
            "autoregressive": {"eps": 0.1}
        }  # TODO: remove once the autoregressive condition is implemented


problem = Problem(data)
model = FeedForward(n_feats, n_feats, 128, 2)


@pytest.mark.parametrize("use_lt", [True, False])
@pytest.mark.parametrize("bool_value", [True, False])
def test_constructor(use_lt, bool_value):

    solver = AutoregressiveSolver(
        problem=problem,
        model=model,
        reset_weights_at_epoch_start=bool_value,
        use_lt=use_lt,
    )

    assert solver.accepted_conditions_types == (
        DataCondition,
    )  # TODO: update once the AutoregressiveCondition is implemented


@pytest.mark.parametrize("use_lt", [True, False])
@pytest.mark.parametrize("batch_size", [None, 1, 2, 5])
@pytest.mark.parametrize("compile", [True, False])
@pytest.mark.parametrize("bool_value", [True, False])
def test_solver_train(use_lt, batch_size, compile, bool_value):
    solver = AutoregressiveSolver(
        model=model,
        problem=problem,
        reset_weights_at_epoch_start=bool_value,
        use_lt=use_lt,
    )
    trainer = Trainer(
        solver=solver,
        max_epochs=2,
        accelerator="cpu",
        batch_size=batch_size,
        train_size=1.0,
        val_size=0.0,
        test_size=0.0,
        compile=compile,
    )
    trainer.train()


@pytest.mark.parametrize("use_lt", [True, False])
@pytest.mark.parametrize("batch_size", [None, 1, 2, 5])
@pytest.mark.parametrize("compile", [True, False])
@pytest.mark.parametrize("bool_value", [True, False])
def test_solver_validation(use_lt, batch_size, compile, bool_value):
    solver = AutoregressiveSolver(
        model=model,
        problem=problem,
        reset_weights_at_epoch_start=bool_value,
        use_lt=use_lt,
    )
    trainer = Trainer(
        solver=solver,
        max_epochs=2,
        accelerator="cpu",
        batch_size=batch_size,
        train_size=0.9,
        val_size=0.1,
        test_size=0.0,
        compile=compile,
    )
    trainer.train()
    if trainer.compile:
        assert isinstance(solver.model, OptimizedModule)


@pytest.mark.parametrize("use_lt", [True, False])
@pytest.mark.parametrize("batch_size", [None, 1, 2, 5])
@pytest.mark.parametrize("compile", [True, False])
@pytest.mark.parametrize("bool_value", [True, False])
def test_solver_test(use_lt, batch_size, compile, bool_value):
    solver = AutoregressiveSolver(
        model=model,
        problem=problem,
        reset_weights_at_epoch_start=bool_value,
        use_lt=use_lt,
    )
    trainer = Trainer(
        solver=solver,
        max_epochs=2,
        accelerator="cpu",
        batch_size=batch_size,
        train_size=0.6,
        val_size=0.2,
        test_size=0.2,
        compile=compile,
    )
    trainer.test()


@pytest.mark.parametrize("use_lt", [True, False])
def test_train_load_restore(use_lt):
    dir = "tests/test_solver/tmp"
    solver = AutoregressiveSolver(
        model=model,
        problem=problem,
        reset_weights_at_epoch_start=False,
        use_lt=use_lt,
    )
    trainer = Trainer(
        solver=solver,
        max_epochs=5,
        accelerator="cpu",
        batch_size=None,
        train_size=0.7,
        val_size=0.2,
        test_size=0.1,
        default_root_dir=dir,
    )
    trainer.train()

    # restore
    new_trainer = Trainer(solver=solver, max_epochs=5, accelerator="cpu")
    new_trainer.train(
        ckpt_path=f"{dir}/lightning_logs/version_0/checkpoints/"
        + "epoch=4-step=5.ckpt"
    )

    # loading
    new_solver = AutoregressiveSolver.load_from_checkpoint(
        f"{dir}/lightning_logs/version_0/checkpoints/epoch=4-step=5.ckpt",
        problem=problem,
        model=model,
    )

    test_pts = LabelTensor(
        torch.rand(n_traj, t_steps, n_feats), problem.input_variables
    )
    assert new_solver.forward(test_pts).shape == (n_traj, t_steps, n_feats)
    assert new_solver.forward(test_pts).shape == solver.forward(test_pts).shape
    torch.testing.assert_close(
        new_solver.forward(test_pts), solver.forward(test_pts)
    )

    shutil.rmtree("tests/test_solver/tmp")
