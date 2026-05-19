import pytest
import torch
from pina.problem.zoo import Poisson2DSquareProblem as Poisson
from torch._dynamo.eval_frame import OptimizedModule
from pina import LabelTensor, Trainer, Condition
from pina.solver import EnsemblePINN
from pina.model import FeedForward
from pina.condition import (
    InputTargetCondition,
    InputEquationCondition,
    DomainEquationCondition,
)

# Initialize and discretise the problem
problem = Poisson()
problem.discretise_domain(10)

# Save input and output variables for convenience
input_vars = problem.input_variables
output_vars = problem.output_variables

# Add a data condition to the problem
input_ = LabelTensor(torch.rand(10, len(input_vars)), input_vars)
target_ = LabelTensor(torch.rand(10, len(output_vars)), output_vars)
problem.conditions["data"] = Condition(input=input_, target=target_)

# Initialize ensemble of models
N = 5
models = [FeedForward(len(input_vars), len(output_vars)) for _ in range(N)]


def test_constructor():

    # Define the solver
    solver = EnsemblePINN(problem=problem, models=models)

    # Assert accepted conditions types and number of ensemble members
    assert solver.accepted_conditions_types == (
        InputTargetCondition,
        InputEquationCondition,
        DomainEquationCondition,
    )
    assert solver.num_ensemble == N


@pytest.mark.parametrize("batch_size", [None, 5])
@pytest.mark.parametrize("compile", [True, False])
def test_solver_train(batch_size, compile):

    # Define the solver
    solver = EnsemblePINN(problem=problem, models=models)

    # Training procedure
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

    # Check if models are compiled when compile is True
    if trainer.compile:
        assert all(
            [isinstance(model, OptimizedModule) for model in solver.models]
        )


@pytest.mark.parametrize("batch_size", [None, 5])
@pytest.mark.parametrize("compile", [True, False])
def test_solver_validation(batch_size, compile):

    # Define the solver
    solver = EnsemblePINN(problem=problem, models=models)

    # Training procedure
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

    # Check if models are compiled when compile is True
    if trainer.compile:
        assert all(
            [isinstance(model, OptimizedModule) for model in solver.models]
        )


@pytest.mark.parametrize("batch_size", [None, 5])
@pytest.mark.parametrize("compile", [True, False])
def test_solver_test(batch_size, compile):

    # Define the solver
    solver = EnsemblePINN(problem=problem, models=models)

    # Training procedure
    trainer = Trainer(
        solver=solver,
        max_epochs=2,
        accelerator="cpu",
        batch_size=batch_size,
        train_size=0.7,
        val_size=0.1,
        test_size=0.2,
        compile=compile,
    )
    trainer.test()

    # Check if models are compiled when compile is True
    if trainer.compile:
        assert all(
            [isinstance(model, OptimizedModule) for model in solver.models]
        )


def test_train_load_restore(clean_tmp_dir):

    # Initialize the directory to store the checkpoints
    dir = clean_tmp_dir

    # Define the solver
    solver = EnsemblePINN(models=models, problem=problem)

    # Training procedure
    trainer = Trainer(
        solver=solver,
        max_epochs=5,
        accelerator="cpu",
        batch_size=None,
        train_size=0.7,
        val_size=0.1,
        test_size=0.2,
        default_root_dir=dir,
    )
    trainer.train()

    # Restore the training from a checkpoint
    new_trainer = Trainer(solver=solver, max_epochs=5, accelerator="cpu")
    new_trainer.train(
        ckpt_path=f"{dir}/lightning_logs/version_0/checkpoints/"
        + "epoch=4-step=5.ckpt"
    )

    # Load the solver from a checkpoint
    new_solver = EnsemblePINN.load_from_checkpoint(
        f"{dir}/lightning_logs/version_0/checkpoints/epoch=4-step=5.ckpt",
        problem=problem,
        models=models,
    )

    # Create input data for testing the forward pass
    test_pts = LabelTensor(torch.rand(20, 2), problem.input_variables)

    # Assert the loaded solver behaves as the original one
    assert new_solver.forward(test_pts).shape == solver.forward(test_pts).shape
    torch.testing.assert_close(
        new_solver.forward(test_pts), solver.forward(test_pts)
    )
