import torch
import torch.nn as nn

import pytest
from pina import Condition, LabelTensor
from pina.solver import GAROM
from pina.condition import InputTargetCondition
from pina.problem import AbstractProblem
from pina.model import FeedForward
from pina.trainer import Trainer
from torch._dynamo.eval_frame import OptimizedModule


class TensorProblem(AbstractProblem):
    input_variables = ["u_0", "u_1"]
    output_variables = ["u"]
    conditions = {
        "data": Condition(target=torch.randn(50, 2), input=torch.randn(50, 1))
    }


# simple Generator Network
class Generator(nn.Module):

    def __init__(
        self,
        input_dimension=2,
        parameters_dimension=1,
        noise_dimension=2,
        activation=torch.nn.SiLU,
    ):
        super().__init__()

        self._noise_dimension = noise_dimension
        self._activation = activation
        self.model = FeedForward(6 * noise_dimension, input_dimension)
        self.condition = FeedForward(parameters_dimension, 5 * noise_dimension)

    def forward(self, param):
        # uniform sampling in [-1, 1]
        z = (
            2
            * torch.rand(
                size=(param.shape[0], self._noise_dimension),
                device=param.device,
                dtype=param.dtype,
                requires_grad=True,
            )
            - 1
        )
        return self.model(torch.cat((z, self.condition(param)), dim=-1))


# Simple Discriminator Network


class Discriminator(nn.Module):

    def __init__(
        self,
        input_dimension=2,
        parameter_dimension=1,
        hidden_dimension=2,
        activation=torch.nn.ReLU,
    ):
        super().__init__()

        self._activation = activation
        self.encoding = FeedForward(input_dimension, hidden_dimension)
        self.decoding = FeedForward(2 * hidden_dimension, input_dimension)
        self.condition = FeedForward(parameter_dimension, hidden_dimension)

    def forward(self, data):
        x, condition = data
        encoding = self.encoding(x)
        conditioning = torch.cat((encoding, self.condition(condition)), dim=-1)
        decoding = self.decoding(conditioning)
        return decoding


def test_constructor():
    GAROM(
        problem=TensorProblem(),
        generator=Generator(),
        discriminator=Discriminator(),
    )
    assert GAROM.accepted_conditions_types == (InputTargetCondition)


@pytest.mark.parametrize("batch_size", [None, 1, 5, 20])
@pytest.mark.parametrize("compile", [True, False])
def test_solver_train(batch_size, compile):
    solver = GAROM(
        problem=TensorProblem(),
        generator=Generator(),
        discriminator=Discriminator(),
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
@pytest.mark.parametrize("compile", [True, False])
def test_solver_validation(batch_size, compile):
    solver = GAROM(
        problem=TensorProblem(),
        generator=Generator(),
        discriminator=Discriminator(),
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
        assert all(
            [isinstance(model, OptimizedModule) for model in solver.models]
        )


@pytest.mark.parametrize("batch_size", [None, 1, 5, 20])
@pytest.mark.parametrize("compile", [True, False])
def test_solver_test(batch_size, compile):
    solver = GAROM(
        problem=TensorProblem(),
        generator=Generator(),
        discriminator=Discriminator(),
    )
    trainer = Trainer(
        solver=solver,
        max_epochs=2,
        accelerator="cpu",
        batch_size=batch_size,
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


def test_train_load_restore():
    dir = "tests/test_solver/tmp/"
    problem = TensorProblem()
    solver = GAROM(
        problem=TensorProblem(),
        generator=Generator(),
        discriminator=Discriminator(),
    )
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
    new_solver = GAROM.load_from_checkpoint(
        f"{dir}/lightning_logs/version_0/checkpoints/epoch=4-step=5.ckpt",
        problem=TensorProblem(),
        generator=Generator(),
        discriminator=Discriminator(),
    )

    test_pts = torch.rand(20, 1)
    assert new_solver.forward(test_pts).shape == (20, 2)
    assert new_solver.forward(test_pts).shape == solver.forward(test_pts).shape

    # rm directories
    import shutil

    shutil.rmtree("tests/test_solver/tmp")
