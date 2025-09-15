import torch
import pytest
from copy import deepcopy

from pina import Trainer, LabelTensor, Condition
from pina.solver import SupervisedSolver
from pina.model import FeedForward
from pina.callback import NormalizerDataCallback
from pina.problem import AbstractProblem
from pina.problem.zoo import Poisson2DSquareProblem as Poisson
from pina.condition.input_target_condition import InputTargetCondition
from pina.solver import PINN

# for checking normalization
stage_map = {
    "train": ["train_dataset"],
    "validate": ["val_dataset"],
    "test": ["test_dataset"],
    "all": ["train_dataset", "val_dataset", "test_dataset"],
}

input_1 = torch.rand(20, 2) * 10
target_1 = torch.rand(20, 1) * 10
input_2 = torch.rand(20, 2) * 5
target_2 = torch.rand(20, 1) * 5


class LabelTensorProblem(AbstractProblem):
    input_variables = ["u_0", "u_1"]
    output_variables = ["u"]
    conditions = {
        "data1": Condition(
            input=LabelTensor(input_1, ["u_0", "u_1"]),
            target=LabelTensor(target_1, ["u"]),
        ),
        "data2": Condition(
            input=LabelTensor(input_2, ["u_0", "u_1"]),
            target=LabelTensor(target_2, ["u"]),
        ),
    }


class TensorProblem(AbstractProblem):
    input_variables = ["u_0", "u_1"]
    output_variables = ["u"]
    conditions = {
        "data1": Condition(input=input_1, target=target_1),
        "data2": Condition(input=input_2, target=target_2),
    }


supervised_solver_no_lt = SupervisedSolver(
    problem=TensorProblem(), model=FeedForward(2, 1), use_lt=False
)
supervised_solver_lt = SupervisedSolver(
    problem=LabelTensorProblem(), model=FeedForward(2, 1), use_lt=True
)

poisson_problem = Poisson()
poisson_problem.conditions["data"] = Condition(
    input=LabelTensor(torch.rand(20, 2) * 10, ["x", "y"]),
    target=LabelTensor(torch.rand(20, 1) * 10, ["u"]),
)


@pytest.mark.parametrize("scale_fn", [torch.std, torch.var])
@pytest.mark.parametrize("shift_fn", [torch.mean, torch.median])
@pytest.mark.parametrize("apply_to", ["input", "target"])
@pytest.mark.parametrize("stage", ["train", "validate", "test", "all"])
def test_init(scale_fn, shift_fn, apply_to, stage):
    normalizer = NormalizerDataCallback(
        scale_fn=scale_fn, shift_fn=shift_fn, apply_to=apply_to, stage=stage
    )
    assert normalizer.scale_fn == scale_fn
    assert normalizer.shift_fn == shift_fn
    assert normalizer.apply_to == apply_to
    assert normalizer.stage == stage


def test_init_invalid_scale():
    with pytest.raises(ValueError):
        NormalizerDataCallback(scale_fn=1)


def test_init_invalid_shift():
    with pytest.raises(ValueError):
        NormalizerDataCallback(shift_fn=1)


@pytest.mark.parametrize("invalid_apply_to", ["inputt", "targett", 1])
def test_init_invalid_apply_to(invalid_apply_to):
    with pytest.raises(ValueError):
        NormalizerDataCallback(apply_to=invalid_apply_to)


@pytest.mark.parametrize("invalid_stage", ["trainn", "validatee", 1])
def test_init_invalid_stage(invalid_stage):
    with pytest.raises(ValueError):
        NormalizerDataCallback(stage=invalid_stage)


@pytest.mark.parametrize(
    "solver", [supervised_solver_lt, supervised_solver_no_lt]
)
@pytest.mark.parametrize("scale_fn", [torch.std, torch.var])
@pytest.mark.parametrize("shift_fn", [torch.mean, torch.median])
@pytest.mark.parametrize("apply_to", ["input", "target"])
@pytest.mark.parametrize("stage", ["all", "train", "validate", "test"])
def test_setup(solver, scale_fn, shift_fn, stage, apply_to):
    trainer = Trainer(
        solver=solver,
        callbacks=NormalizerDataCallback(
            scale_fn=scale_fn, shift_fn=shift_fn, stage=stage, apply_to=apply_to
        ),
        max_epochs=1,
        train_size=0.4,
        val_size=0.3,
        test_size=0.3,
        shuffle=False,
    )
    trainer_copy = deepcopy(trainer)
    trainer_copy.data_module.setup("fit")
    trainer_copy.data_module.setup("test")
    trainer.train()
    trainer.test()

    normalizer = trainer.callbacks[0].normalizer

    for cond in ["data1", "data2"]:
        scale = scale_fn(
            trainer_copy.data_module.train_dataset.conditions_dict[cond][
                apply_to
            ]
        )
        shift = shift_fn(
            trainer_copy.data_module.train_dataset.conditions_dict[cond][
                apply_to
            ]
        )
        assert "scale" in normalizer[cond]
        assert "shift" in normalizer[cond]
        assert normalizer[cond]["scale"] - scale < 1e-5
        assert normalizer[cond]["shift"] - shift < 1e-5
        for ds_name in stage_map[stage]:
            dataset = getattr(trainer.data_module, ds_name, None)
            old_dataset = getattr(trainer_copy.data_module, ds_name, None)
            current_points = dataset.conditions_dict[cond][apply_to]
            old_points = old_dataset.conditions_dict[cond][apply_to]
            expected = (old_points - shift) / scale
            assert torch.allclose(current_points, expected)


@pytest.mark.parametrize("scale_fn", [torch.std, torch.var])
@pytest.mark.parametrize("shift_fn", [torch.mean, torch.median])
@pytest.mark.parametrize("apply_to", ["input"])
@pytest.mark.parametrize("stage", ["all", "train", "validate", "test"])
def test_setup_pinn(scale_fn, shift_fn, stage, apply_to):
    pinn = PINN(
        problem=poisson_problem,
        model=FeedForward(2, 1),
    )
    poisson_problem.discretise_domain(n=10)
    trainer = Trainer(
        solver=pinn,
        callbacks=NormalizerDataCallback(
            scale_fn=scale_fn,
            shift_fn=shift_fn,
            stage=stage,
            apply_to=apply_to,
        ),
        max_epochs=1,
        train_size=0.4,
        val_size=0.3,
        test_size=0.3,
        shuffle=False,
    )

    trainer_copy = deepcopy(trainer)
    trainer_copy.data_module.setup("fit")
    trainer_copy.data_module.setup("test")
    trainer.train()
    trainer.test()

    conditions = trainer.callbacks[0].normalizer.keys()
    assert "data" in conditions
    assert len(conditions) == 1
    normalizer = trainer.callbacks[0].normalizer
    cond = "data"

    scale = scale_fn(
        trainer_copy.data_module.train_dataset.conditions_dict[cond][apply_to]
    )
    shift = shift_fn(
        trainer_copy.data_module.train_dataset.conditions_dict[cond][apply_to]
    )
    assert "scale" in normalizer[cond]
    assert "shift" in normalizer[cond]
    assert normalizer[cond]["scale"] - scale < 1e-5
    assert normalizer[cond]["shift"] - shift < 1e-5
    for ds_name in stage_map[stage]:
        dataset = getattr(trainer.data_module, ds_name, None)
        old_dataset = getattr(trainer_copy.data_module, ds_name, None)
        current_points = dataset.conditions_dict[cond][apply_to]
        old_points = old_dataset.conditions_dict[cond][apply_to]
        expected = (old_points - shift) / scale
        assert torch.allclose(current_points, expected)
