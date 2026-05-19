import torch
import pytest
from torch._dynamo.eval_frame import OptimizedModule
from pina import Condition, LabelTensor, Trainer
from pina.solver import EnsembleSimpleSolver
from pina.problem import BaseProblem
from pina.graph import KNNGraph
from pina.condition import (
    InputTargetCondition,
    InputEquationCondition,
    DomainEquationCondition,
)


# Helper class for Tensor problems
class TensorProblem(BaseProblem):

    # Input and output variables
    input_variables = ["u_0", "u_1"]
    output_variables = ["u"]

    # Input and target
    input_ = torch.rand(20, 2)
    target_ = torch.rand(20, 1)

    # Condition
    conditions = {}

    def __init__(self, use_lt):
        super().__init__()

        # Add labels if use_lt is True
        if use_lt:
            self.input_ = LabelTensor(self.input_, self.input_variables)
            self.target_ = LabelTensor(self.target_, self.output_variables)

        # Initialize conditions
        self.conditions["data"] = Condition(
            input=self.input_, target=self.target_
        )


# Helper class for Graph problems
class GraphProblem(BaseProblem):

    # Input and output variables
    input_variables = ["a", "b", "c"]
    output_variables = ["u"]

    # Graph attributes and target
    x = torch.rand(10, 20, 3)
    pos = torch.rand(10, 20, 2)
    target_ = torch.rand(10, 20, 1)

    # Condition
    conditions = {}

    def __init__(self, use_lt):
        super().__init__()

        # Add labels if use_lt is True
        if use_lt:
            self.x = LabelTensor(self.x, self.input_variables)
            self.pos = LabelTensor(self.pos, ["x", "y"])
            self.target_ = LabelTensor(self.target_, self.output_variables)

        # Initialize the input graphs
        input_ = [
            KNNGraph(x=self.x[i], pos=self.pos[i], neighbours=3, edge_attr=True)
            for i in range(len(self.x))
        ]

        # Initialize conditions
        self.conditions["data"] = Condition(input=input_, target=self.target_)


# Helper class for Graph-consistent architecture
class DummyGraphModel(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(3, 1)

    def forward(self, batch):
        return self.layer(batch.x)


tensor_models = [torch.nn.Linear(2, 1) for _ in range(10)]
graph_models = [DummyGraphModel() for _ in range(10)]


@pytest.mark.parametrize("case", ["tensor", "graph"])
@pytest.mark.parametrize("use_lt", [True, False])
def test_constructor(case, use_lt):

    # Initialize problems and models based on the case
    if case == "tensor":
        problem = TensorProblem(use_lt=use_lt)
        models = tensor_models
    else:
        problem = GraphProblem(use_lt=use_lt)
        models = graph_models

    # Define the solver
    solver = EnsembleSimpleSolver(problem=problem, models=models)

    # Assert accepted conditions types and number of ensemble members
    assert solver.accepted_conditions_types == (
        InputTargetCondition,
        InputEquationCondition,
        DomainEquationCondition,
    )
    assert solver.num_ensemble == 10


@pytest.mark.parametrize("case", ["tensor", "graph"])
@pytest.mark.parametrize("batch_size", [None, 5])
@pytest.mark.parametrize("use_lt", [True, False])
@pytest.mark.parametrize("compile", [True, False])
def test_solver_train(batch_size, compile, case, use_lt):

    # Initialize problems and models based on the case
    if case == "tensor":
        problem = TensorProblem(use_lt=use_lt)
        models = tensor_models
    else:
        problem = GraphProblem(use_lt=use_lt)
        models = graph_models

    # Define the solver
    solver = EnsembleSimpleSolver(problem=problem, models=models, use_lt=use_lt)

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


@pytest.mark.parametrize("case", ["tensor", "graph"])
@pytest.mark.parametrize("batch_size", [None, 5])
@pytest.mark.parametrize("use_lt", [True, False])
@pytest.mark.parametrize("compile", [True, False])
def test_solver_validation(batch_size, compile, case, use_lt):

    # Initialize problems and models based on the case
    if case == "tensor":
        problem = TensorProblem(use_lt=use_lt)
        models = tensor_models
    else:
        problem = GraphProblem(use_lt=use_lt)
        models = graph_models

    # Define the solver
    solver = EnsembleSimpleSolver(problem=problem, models=models, use_lt=use_lt)

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


@pytest.mark.parametrize("case", ["tensor", "graph"])
@pytest.mark.parametrize("batch_size", [None, 5])
@pytest.mark.parametrize("use_lt", [True, False])
@pytest.mark.parametrize("compile", [True, False])
def test_solver_test(batch_size, compile, case, use_lt):

    # Initialize problems and models based on the case
    if case == "tensor":
        problem = TensorProblem(use_lt=use_lt)
        models = tensor_models
    else:
        problem = GraphProblem(use_lt=use_lt)
        models = graph_models

    # Define the solver
    solver = EnsembleSimpleSolver(problem=problem, models=models, use_lt=use_lt)

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


@pytest.mark.parametrize("case", ["tensor", "graph"])
@pytest.mark.parametrize("use_lt", [True, False])
def test_train_load_restore(clean_tmp_dir, case, use_lt):

    # Initialize the directory to store the checkpoints
    dir = clean_tmp_dir

    # Initialize problems and models based on the case
    if case == "tensor":
        problem = TensorProblem(use_lt=use_lt)
        models = tensor_models
    else:
        problem = GraphProblem(use_lt=use_lt)
        models = graph_models

    # Define the solver
    solver = EnsembleSimpleSolver(problem=problem, models=models, use_lt=use_lt)

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
    new_solver = EnsembleSimpleSolver.load_from_checkpoint(
        f"{dir}/lightning_logs/version_0/checkpoints/epoch=4-step=5.ckpt",
        problem=problem,
        models=models,
    )

    # Create input data for testing the forward pass
    if case == "tensor":
        test_pts = LabelTensor(torch.rand(20, 2), problem.input_variables)
    else:
        test_pts = KNNGraph(
            x=LabelTensor(torch.rand(20, 3), ["a", "b", "c"]),
            pos=LabelTensor(torch.rand(20, 2), ["x", "y"]),
            neighbours=3,
            edge_attr=True,
        )

    # Assert the loaded solver behaves as the original one
    assert new_solver.forward(test_pts).shape == solver.forward(test_pts).shape
    torch.testing.assert_close(
        new_solver.forward(test_pts), solver.forward(test_pts)
    )
