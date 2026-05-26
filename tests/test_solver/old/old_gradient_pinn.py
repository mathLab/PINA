import pytest
import torch
from torch._dynamo.eval_frame import OptimizedModule
from pina import LabelTensor, Condition, Trainer
from pina.problem import BaseProblem
from pina.solver import GradientPINN
from pina.model import FeedForward
from pina.problem.zoo import (
    InversePoisson2DSquareProblem as InversePoisson,
    Poisson2DSquareProblem as Poisson,
)
from pina.condition import (
    InputTargetCondition,
    InputEquationCondition,
    DomainEquationCondition,
)

# Initialize forward and inverse problems
problem = Poisson()
problem.discretise_domain(10)
inverse_problem = InversePoisson(load=True, data_size=0.01)
inverse_problem.discretise_domain(10)

# Add input-output condition to test supervised learning
input_pts = torch.rand(10, len(problem.input_variables))
input_pts = LabelTensor(input_pts, problem.input_variables)
output_pts = torch.rand(10, len(problem.output_variables))
output_pts = LabelTensor(output_pts, problem.output_variables)
problem.conditions["data"] = Condition(input=input_pts, target=output_pts)

# Initialize the model
model = FeedForward(len(problem.input_variables), len(problem.output_variables))


# Define a dummy problem for testing error handling
class DummyProblem(BaseProblem):
    input_variables = ["x", "y"]
    output_variables = ["u"]
    conditions = {}


@pytest.mark.parametrize("problem", [problem, inverse_problem])
@pytest.mark.parametrize("gradient_conditions", [None, ["D"]])
def test_constructor(problem, gradient_conditions):

    # Initialize the solver
    solver = GradientPINN(
        model=model, problem=problem, gradient_conditions=gradient_conditions
    )

    # Check accepted condition types
    assert solver.accepted_conditions_types == (
        InputTargetCondition,
        InputEquationCondition,
        DomainEquationCondition,
    )

    # Should raise error if problem is not a SpatialProblem
    with pytest.raises(ValueError):
        GradientPINN(model=model, problem=DummyProblem())

    # Should raise error if any of the provided conditions is not defined
    with pytest.raises(ValueError):
        GradientPINN(
            model=model,
            problem=problem,
            gradient_conditions=["non_existent_condition"],
        )

    # Should raise error if any of the provided conditions is not equation-based
    with pytest.raises(ValueError):
        GradientPINN(
            model=model,
            problem=problem,
            gradient_conditions=["data"],
        )


@pytest.mark.parametrize("problem", [problem, inverse_problem])
@pytest.mark.parametrize("gradient_conditions", [None, ["D"]])
@pytest.mark.parametrize("batch_size", [None, 5])
@pytest.mark.parametrize("compile", [True, False])
def test_solver_train(problem, batch_size, compile, gradient_conditions):

    # Initialize the solver and the trainer
    solver = GradientPINN(
        model=model, problem=problem, gradient_conditions=gradient_conditions
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

    # Check if model is compiled when compile=True
    if trainer.compile:
        assert isinstance(solver.model, OptimizedModule)


@pytest.mark.parametrize("problem", [problem, inverse_problem])
@pytest.mark.parametrize("batch_size", [None, 5])
@pytest.mark.parametrize("compile", [True, False])
@pytest.mark.parametrize("gradient_conditions", [None, ["D"]])
def test_solver_validation(problem, batch_size, compile, gradient_conditions):

    # Initialize the solver and the trainer
    solver = GradientPINN(
        model=model, problem=problem, gradient_conditions=gradient_conditions
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

    # Check if model is compiled when compile=True
    if trainer.compile:
        assert isinstance(solver.model, OptimizedModule)


@pytest.mark.parametrize("problem", [problem, inverse_problem])
@pytest.mark.parametrize("batch_size", [None, 5])
@pytest.mark.parametrize("compile", [True, False])
@pytest.mark.parametrize("gradient_conditions", [None, ["D"]])
def test_solver_test(problem, batch_size, compile, gradient_conditions):

    # Initialize the solver and the trainer
    solver = GradientPINN(
        model=model, problem=problem, gradient_conditions=gradient_conditions
    )
    trainer = Trainer(
        solver=solver,
        max_epochs=2,
        accelerator="cpu",
        batch_size=batch_size,
        train_size=0.7,
        val_size=0.2,
        test_size=0.1,
        compile=compile,
    )
    trainer.test()

    # Check if model is compiled when compile=True
    if trainer.compile:
        assert isinstance(solver.model, OptimizedModule)


@pytest.mark.parametrize("problem", [problem, inverse_problem])
@pytest.mark.parametrize("gradient_conditions", [None, ["D"]])
def test_train_load_restore(clean_tmp_dir, problem, gradient_conditions):

    # Set the temporary directory for saving checkpoints
    dir = clean_tmp_dir

    # Initialize the solver and the trainer
    solver = GradientPINN(
        model=model, problem=problem, gradient_conditions=gradient_conditions
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

    # Restore the trainer from a checkpoint
    new_trainer = Trainer(solver=solver, max_epochs=5, accelerator="cpu")
    new_trainer.train(
        ckpt_path=f"{dir}/lightning_logs/version_0/checkpoints/"
        + "epoch=4-step=5.ckpt"
    )

    # Load the solver from the checkpoint
    new_solver = GradientPINN.load_from_checkpoint(
        f"{dir}/lightning_logs/version_0/checkpoints/epoch=4-step=5.ckpt",
        problem=problem,
        model=model,
    )

    # Test points
    n_pts = 20
    test_pts = problem.domains["D"].sample(n=n_pts)

    # Assert that the predictions from the loaded solver match original ones
    assert new_solver.forward(test_pts).shape == (n_pts, 1)
    assert new_solver.forward(test_pts).shape == solver.forward(test_pts).shape
    torch.testing.assert_close(
        new_solver.forward(test_pts), solver.forward(test_pts)
    )
