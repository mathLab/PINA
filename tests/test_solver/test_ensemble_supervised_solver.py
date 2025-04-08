import torch
import pytest
from torch._dynamo.eval_frame import OptimizedModule
from torch_geometric.nn import GCNConv
from pina import Condition, LabelTensor
from pina.condition import InputTargetCondition
from pina.problem import AbstractProblem
from pina.solver import DeepEnsembleSupervisedSolver
from pina.model import FeedForward
from pina.trainer import Trainer
from pina.graph import KNNGraph


class LabelTensorProblem(AbstractProblem):
    input_variables = ["u_0", "u_1"]
    output_variables = ["u"]
    conditions = {
        "data": Condition(
            input=LabelTensor(torch.randn(20, 2), ["u_0", "u_1"]),
            target=LabelTensor(torch.randn(20, 1), ["u"]),
        ),
    }


class TensorProblem(AbstractProblem):
    input_variables = ["u_0", "u_1"]
    output_variables = ["u"]
    conditions = {
        "data": Condition(input=torch.randn(20, 2), target=torch.randn(20, 1))
    }


x = torch.rand((100, 20, 5))
pos = torch.rand((100, 20, 2))
output_ = torch.rand((100, 20, 1))
input_ = [
    KNNGraph(x=x_, pos=pos_, neighbours=3, edge_attr=True)
    for x_, pos_ in zip(x, pos)
]


class GraphProblem(AbstractProblem):
    output_variables = None
    conditions = {"data": Condition(input=input_, target=output_)}


x = LabelTensor(torch.rand((100, 20, 5)), ["a", "b", "c", "d", "e"])
pos = LabelTensor(torch.rand((100, 20, 2)), ["x", "y"])
output_ = LabelTensor(torch.rand((100, 20, 1)), ["u"])
input_ = [
    KNNGraph(x=x[i], pos=pos[i], neighbours=3, edge_attr=True)
    for i in range(len(x))
]


class GraphProblemLT(AbstractProblem):
    output_variables = ["u"]
    input_variables = ["a", "b", "c", "d", "e"]
    conditions = {"data": Condition(input=input_, target=output_)}


models = [FeedForward(2, 1) for i in range(10)]


class Models(torch.nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lift = torch.nn.Linear(5, 10)
        self.activation = torch.nn.Tanh()
        self.output = torch.nn.Linear(10, 1)

        self.conv = GCNConv(10, 10)

    def forward(self, batch):

        x = batch.x
        edge_index = batch.edge_index
        for _ in range(1):
            y = self.lift(x)
            y = self.activation(y)
            y = self.conv(y, edge_index)
            y = self.activation(y)
            y = self.output(y)
        return y


graph_models = [Models() for i in range(10)]


def test_constructor():
    solver = DeepEnsembleSupervisedSolver(
        problem=TensorProblem(), models=models
    )
    DeepEnsembleSupervisedSolver(problem=LabelTensorProblem(), models=models)
    assert DeepEnsembleSupervisedSolver.accepted_conditions_types == (
        InputTargetCondition
    )
    assert solver.num_ensembles == 10


@pytest.mark.parametrize("batch_size", [None, 1, 5, 20])
@pytest.mark.parametrize("use_lt", [True, False])
@pytest.mark.parametrize("compile", [True, False])
def test_solver_train(use_lt, batch_size, compile):
    problem = LabelTensorProblem() if use_lt else TensorProblem()
    solver = DeepEnsembleSupervisedSolver(
        problem=problem, models=models, use_lt=use_lt
    )
    trainer = Trainer(
        solver=solver,
        max_epochs=2,
        accelerator="cpu",
        batch_size=batch_size,
        train_size=1.0,
        test_size=0.0,
        val_size=0.0,
        compile=compile,
    )

    trainer.train()
    if trainer.compile:
        assert all(
            [isinstance(model, OptimizedModule) for model in solver.models]
        )


@pytest.mark.parametrize("batch_size", [None, 1, 5, 20])
@pytest.mark.parametrize("use_lt", [True, False])
def test_solver_train_graph(batch_size, use_lt):
    problem = GraphProblemLT() if use_lt else GraphProblem()
    solver = DeepEnsembleSupervisedSolver(
        problem=problem, models=graph_models, use_lt=use_lt
    )
    trainer = Trainer(
        solver=solver,
        max_epochs=2,
        accelerator="cpu",
        batch_size=batch_size,
        train_size=1.0,
        test_size=0.0,
        val_size=0.0,
    )

    trainer.train()


@pytest.mark.parametrize("use_lt", [True, False])
@pytest.mark.parametrize("compile", [True, False])
def test_solver_validation(use_lt, compile):
    problem = LabelTensorProblem() if use_lt else TensorProblem()
    solver = DeepEnsembleSupervisedSolver(
        problem=problem, models=models, use_lt=use_lt
    )
    trainer = Trainer(
        solver=solver,
        max_epochs=2,
        accelerator="cpu",
        batch_size=None,
        train_size=0.9,
        val_size=0.1,
        test_size=0.0,
        compile=compile,
    )
    trainer.train()
    if trainer.compile:
        assert all(
            [isinstance(model, OptimizedModule) for model in solver.models]
        )


@pytest.mark.parametrize("batch_size", [None, 1, 5, 20])
@pytest.mark.parametrize("use_lt", [True, False])
def test_solver_validation_graph(batch_size, use_lt):
    problem = GraphProblemLT() if use_lt else GraphProblem()
    solver = DeepEnsembleSupervisedSolver(
        problem=problem, models=graph_models, use_lt=use_lt
    )
    trainer = Trainer(
        solver=solver,
        max_epochs=2,
        accelerator="cpu",
        batch_size=batch_size,
        train_size=0.9,
        val_size=0.1,
        test_size=0.0,
    )

    trainer.train()


@pytest.mark.parametrize("use_lt", [True, False])
@pytest.mark.parametrize("compile", [True, False])
def test_solver_test(use_lt, compile):
    problem = LabelTensorProblem() if use_lt else TensorProblem()
    solver = DeepEnsembleSupervisedSolver(
        problem=problem, models=models, use_lt=use_lt
    )
    trainer = Trainer(
        solver=solver,
        max_epochs=2,
        accelerator="cpu",
        batch_size=None,
        train_size=0.8,
        val_size=0.1,
        test_size=0.1,
        compile=compile,
    )
    trainer.test()
    if trainer.compile:
        assert all(
            [isinstance(model, OptimizedModule) for model in solver.models]
        )


@pytest.mark.parametrize("batch_size", [None, 1, 5, 20])
@pytest.mark.parametrize("use_lt", [True, False])
def test_solver_test_graph(batch_size, use_lt):
    problem = GraphProblemLT() if use_lt else GraphProblem()
    solver = DeepEnsembleSupervisedSolver(
        problem=problem, models=graph_models, use_lt=use_lt
    )
    trainer = Trainer(
        solver=solver,
        max_epochs=2,
        accelerator="cpu",
        batch_size=batch_size,
        train_size=0.8,
        val_size=0.1,
        test_size=0.1,
    )

    trainer.test()


def test_train_load_restore():
    dir = "tests/test_solver/tmp/"
    problem = LabelTensorProblem()
    solver = DeepEnsembleSupervisedSolver(problem=problem, models=models)
    trainer = Trainer(
        solver=solver,
        max_epochs=5,
        accelerator="cpu",
        batch_size=None,
        train_size=0.9,
        test_size=0.1,
        val_size=0.0,
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
    new_solver = DeepEnsembleSupervisedSolver.load_from_checkpoint(
        f"{dir}/lightning_logs/version_0/checkpoints/epoch=4-step=5.ckpt",
        problem=problem,
        models=models,
    )

    test_pts = LabelTensor(torch.rand(20, 2), problem.input_variables)
    assert new_solver.forward(test_pts).shape == solver.forward(test_pts).shape
    torch.testing.assert_close(
        new_solver.forward(test_pts), solver.forward(test_pts)
    )

    # rm directories
    import shutil

    shutil.rmtree("tests/test_solver/tmp")
