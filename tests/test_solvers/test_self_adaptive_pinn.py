import torch
import pytest

from pina import LabelTensor, Condition
from pina.solvers import SelfAdaptivePINN as SAPINN
from pina.trainer import Trainer
from pina.model import FeedForward
from pina.problem.zoo import (
    Poisson2DSquareProblem as Poisson,
    InversePoisson2DSquareProblem as InversePoisson
)
from pina.condition import (
    InputOutputPointsCondition,
    InputPointsEquationCondition,
    DomainEquationCondition
)
from torch._dynamo.eval_frame import OptimizedModule


# make the problem
problem = Poisson()
problem.discretise_domain(50)
inverse_problem = InversePoisson()
inverse_problem.discretise_domain(50)
model = FeedForward(
    len(problem.input_variables),
    len(problem.output_variables)
)

# add input-output condition to test supervised learning
input_pts = torch.rand(50, len(problem.input_variables))
input_pts = LabelTensor(input_pts, problem.input_variables)
output_pts = torch.rand(50, len(problem.output_variables))
output_pts = LabelTensor(output_pts, problem.output_variables)
problem.conditions['data'] = Condition(
    input_points=input_pts,
    output_points=output_pts
)


@pytest.mark.parametrize("problem", [problem, inverse_problem])
@pytest.mark.parametrize("weight_fn", [torch.nn.Sigmoid(), torch.nn.Tanh()])
def test_constructor(problem, weight_fn):
    with pytest.raises(ValueError):
        SAPINN(model=model, problem=problem, weight_function=1)
    solver = SAPINN(problem=problem, model=model, weight_function=weight_fn)

    assert solver.accepted_conditions_types == (
        InputOutputPointsCondition,
        InputPointsEquationCondition,
        DomainEquationCondition
    )


@pytest.mark.parametrize("problem", [problem, inverse_problem])
def test_wrong_batch(problem):
    with pytest.raises(NotImplementedError):
        solver = SAPINN(model=model, problem=problem)
        trainer = Trainer(solver=solver,
                          max_epochs=2,
                          accelerator='cpu',
                          batch_size=10,
                          train_size=1.,
                          val_size=0.,
                          test_size=0.)
        trainer.train()


@pytest.mark.parametrize("problem", [problem, inverse_problem])
@pytest.mark.parametrize("compile", [True, False])
def test_solver_train(problem, compile):
    solver = SAPINN(model=model, problem=problem)
    trainer = Trainer(solver=solver,
                      max_epochs=2,
                      accelerator='cpu',
                      batch_size=None,
                      train_size=1.,
                      val_size=0.,
                      test_size=0.,
                      compile=compile)
    trainer.train()
    if trainer.compile:
        assert (all([isinstance(model, (OptimizedModule, torch.nn.ModuleDict))
                for model in solver.models]))


@pytest.mark.parametrize("problem", [problem, inverse_problem])
@pytest.mark.parametrize("compile", [True, False])
def test_solver_validation(problem, compile):
    solver = SAPINN(model=model, problem=problem)
    trainer = Trainer(solver=solver,
                      max_epochs=2,
                      accelerator='cpu',
                      batch_size=None,
                      train_size=0.9,
                      val_size=0.1,
                      test_size=0.,
                      compile=compile)
    trainer.train()
    if trainer.compile:
        assert (all([isinstance(model, (OptimizedModule, torch.nn.ModuleDict))
                for model in solver.models]))


@pytest.mark.parametrize("problem", [problem, inverse_problem])
@pytest.mark.parametrize("compile", [True, False])
def test_solver_test(problem, compile):
    solver = SAPINN(model=model, problem=problem)
    trainer = Trainer(solver=solver,
                      max_epochs=2,
                      accelerator='cpu',
                      batch_size=None,
                      train_size=0.7,
                      val_size=0.2,
                      test_size=0.1,
                      compile=compile)
    trainer.test()
    if trainer.compile:
        assert (all([isinstance(model, (OptimizedModule, torch.nn.ModuleDict))
                for model in solver.models]))


@pytest.mark.parametrize("problem", [problem, inverse_problem])
def test_train_load_restore(problem):
    dir = "tests/test_solvers/tmp"
    problem = problem
    solver = SAPINN(model=model, problem=problem)
    trainer = Trainer(solver=solver,
                      max_epochs=5,
                      accelerator='cpu',
                      batch_size=None,
                      train_size=0.7,
                      val_size=0.2,
                      test_size=0.1,
                      default_root_dir=dir)
    trainer.train()
    # restore
    new_trainer = Trainer(solver=solver, max_epochs=5, accelerator='cpu')
    new_trainer.train(
        ckpt_path=f'{dir}/lightning_logs/version_0/checkpoints/' +
        'epoch=4-step=5.ckpt')

    # loading
    new_solver = SAPINN.load_from_checkpoint(
        f'{dir}/lightning_logs/version_0/checkpoints/epoch=4-step=5.ckpt',
        problem=problem, model=model)

    test_pts = LabelTensor(torch.rand(20, 2), problem.input_variables)
    assert new_solver.forward(test_pts).shape == (20, 1)
    assert new_solver.forward(test_pts).shape == (
        solver.forward(test_pts).shape
    )
    torch.testing.assert_close(
        new_solver.forward(test_pts),
        solver.forward(test_pts))

    # rm directories
    import shutil
    shutil.rmtree('tests/test_solvers/tmp')
