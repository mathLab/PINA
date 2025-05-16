import pytest
import torch

from pina import LabelTensor, Condition
from pina.problem import TimeDependentProblem
from pina.solver import GradientPINN
from pina.model import FeedForward
from pina.trainer import Trainer
from pina.problem.zoo import (
    Poisson2DSquareProblem as Poisson,
    InversePoisson2DSquareProblem as InversePoisson,
)
from pina.condition import (
    InputTargetCondition,
    InputEquationCondition,
    DomainEquationCondition,
)
from torch._dynamo.eval_frame import OptimizedModule


class DummyTimeProblem(TimeDependentProblem):
    """
    A mock time-dependent problem for testing purposes.
    """

    output_variables = ["u"]
    temporal_domain = None
    conditions = {}


# define problems
problem = Poisson()
problem.discretise_domain(10)
inverse_problem = InversePoisson()
inverse_problem.discretise_domain(10)

# reduce the number of data points to speed up testing
if hasattr(inverse_problem.conditions, "data"):
    data_condition = inverse_problem.conditions["data"]
    data_condition.input = data_condition.input[:10]
    data_condition.target = data_condition.target[:10]

# add input-output condition to test supervised learning
input_pts = torch.rand(10, len(problem.input_variables))
input_pts = LabelTensor(input_pts, problem.input_variables)
output_pts = torch.rand(10, len(problem.output_variables))
output_pts = LabelTensor(output_pts, problem.output_variables)
problem.conditions["data"] = Condition(input=input_pts, target=output_pts)

# define model
model = FeedForward(len(problem.input_variables), len(problem.output_variables))


@pytest.mark.parametrize("problem", [problem, inverse_problem])
def test_constructor(problem):
    with pytest.raises(ValueError):
        GradientPINN(model=model, problem=DummyTimeProblem())
    solver = GradientPINN(model=model, problem=problem)

    assert solver.accepted_conditions_types == (
        InputTargetCondition,
        InputEquationCondition,
        DomainEquationCondition,
    )


@pytest.mark.parametrize("problem", [problem, inverse_problem])
@pytest.mark.parametrize("batch_size", [None, 1, 5, 20])
@pytest.mark.parametrize("compile", [True, False])
def test_solver_train(problem, batch_size, compile):
    solver = GradientPINN(model=model, problem=problem)
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
    if trainer.compile:
        assert isinstance(solver.model, OptimizedModule)


@pytest.mark.parametrize("problem", [problem, inverse_problem])
@pytest.mark.parametrize("batch_size", [None, 1, 5, 20])
@pytest.mark.parametrize("compile", [True, False])
def test_solver_validation(problem, batch_size, compile):
    solver = GradientPINN(model=model, problem=problem)
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


@pytest.mark.parametrize("problem", [problem, inverse_problem])
@pytest.mark.parametrize("batch_size", [None, 1, 5, 20])
@pytest.mark.parametrize("compile", [True, False])
def test_solver_test(problem, batch_size, compile):
    solver = GradientPINN(model=model, problem=problem)
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
    if trainer.compile:
        assert isinstance(solver.model, OptimizedModule)


@pytest.mark.parametrize("problem", [problem, inverse_problem])
def test_train_load_restore(problem):
    dir = "tests/test_solver/tmp"
    problem = problem
    solver = GradientPINN(model=model, problem=problem)
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
    new_solver = GradientPINN.load_from_checkpoint(
        f"{dir}/lightning_logs/version_0/checkpoints/epoch=4-step=5.ckpt",
        problem=problem,
        model=model,
    )

    test_pts = LabelTensor(torch.rand(20, 2), problem.input_variables)
    assert new_solver.forward(test_pts).shape == (20, 1)
    assert new_solver.forward(test_pts).shape == (
        solver.forward(test_pts).shape
    )
    torch.testing.assert_close(
        new_solver.forward(test_pts), solver.forward(test_pts)
    )

    # rm directories
    import shutil

    shutil.rmtree("tests/test_solver/tmp")
