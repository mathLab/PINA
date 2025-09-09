import torch
import pytest

from copy import deepcopy

from pina import Trainer, LabelTensor, Condition
from pina.solver import PINN, SupervisedSolver
from pina.model import FeedForward
from pina.callback import NormalizerDataCallback
from pina.problem import AbstractProblem
from pina.problem.zoo import Poisson2DSquareProblem as Poisson


# for checking normalization
stage_map = {
    "train": ["train_dataset"],
    "validate": ["val_dataset"],
    "test": ["test_dataset"],
    "all": ["train_dataset", "val_dataset", "test_dataset"],
}

# pinn solver
problem = Poisson()
problem.discretise_domain(10)
model = FeedForward(len(problem.input_variables), len(problem.output_variables))
pinn_solver = PINN(problem=problem, model=model)


class LabelTensorProblem(AbstractProblem):
    input_variables = ["u_0", "u_1"]
    output_variables = ["u"]
    conditions = {
        "data1": Condition(
            input=LabelTensor(torch.randn(20, 2), ["u_0", "u_1"]),
            target=LabelTensor(torch.randn(20, 1), ["u"]),
        ),
        "data2": Condition(
            input=LabelTensor(torch.randn(20, 2), ["u_0", "u_1"]),
            target=LabelTensor(torch.randn(20, 1), ["u"]),
        ),
    }


class TensorProblem(AbstractProblem):
    input_variables = ["u_0", "u_1"]
    output_variables = ["u"]
    conditions = {
        "data1": Condition(input=torch.randn(20, 2), target=torch.randn(20, 1)),
        "data2": Condition(input=torch.randn(20, 2), target=torch.randn(20, 1)),
    }


supervised_solver_no_lt = SupervisedSolver(
    problem=TensorProblem(), model=FeedForward(2, 1), use_lt=False
)
supervised_solver_lt = SupervisedSolver(
    problem=LabelTensorProblem(), model=FeedForward(2, 1), use_lt=False
)


# Test constructor
@pytest.mark.parametrize(
    "normalizer",
    [
        {"scale": 2.1, "shift": 1},
        {"scale": 2, "shift": torch.randn(2)},
        {"scale": 2, "shift": 1.1},
        {"a": {"scale": 2, "shift": 1}, "b": {"scale": 3, "shift": 0.5}},
    ],
)
def test_constructor_valid_normalizers(normalizer):
    NormalizerDataCallback(normalizer=normalizer)


@pytest.mark.parametrize(
    "invalid_normalizer",
    [
        {"scale": 1},  # missing shift
        {"shift": 1},  # missing scale
        {"a": {"scale": 1}},  # dict of dicts, inner missing shift
        [1, 2, 3],  # wrong type
        "invalid",  # wrong type
    ],
)
def test_constructor_invalid_normalizer_raises(invalid_normalizer):
    with pytest.raises(ValueError):
        NormalizerDataCallback(normalizer=invalid_normalizer)


@pytest.mark.parametrize("apply_to", ["input", "target"])
def test_constructor_valid_apply_to(apply_to):
    cb = NormalizerDataCallback(apply_to=apply_to)
    assert cb.apply_to == apply_to


@pytest.mark.parametrize("apply_to", ["invalid", "", None, 123])
def test_constructor_invalid_apply_to_raises(apply_to):
    with pytest.raises(ValueError):
        NormalizerDataCallback(apply_to=apply_to)


@pytest.mark.parametrize("stage", ["train", "validate", "test", "all"])
def test_constructor_valid_stage(stage):
    cb = NormalizerDataCallback(stage=stage)
    assert cb.stage == stage


@pytest.mark.parametrize("stage", ["invalid", "", None, 123])
def test_constructor_invalid_stage_raises(stage):
    with pytest.raises(ValueError):
        NormalizerDataCallback(stage=stage)


# Test setup
@pytest.mark.parametrize(
    "normalizer",
    [
        {"scale": 0.5, "shift": 1},
    ],
)
def test_invalid_setup(normalizer):
    trainer = Trainer(
        solver=pinn_solver,
        callbacks=NormalizerDataCallback(normalizer, apply_to="target"),
        max_epochs=1,
        train_size=0.4,
        val_size=0.3,
        test_size=0.3,
    )
    # trigger setup
    with pytest.raises(RuntimeError):
        trainer.train()
    with pytest.raises(RuntimeError):
        trainer.test()


@pytest.mark.parametrize("apply_to", ["input", "target"])
@pytest.mark.parametrize(
    "solver", [supervised_solver_lt, supervised_solver_no_lt]
)
@pytest.mark.parametrize("stage", ["train", "validate", "test", "all"])
def test_setup(apply_to, solver, stage):
    shift = torch.tensor([1, 1]) if apply_to == "input" else torch.tensor([1])

    # Helper function to run trainer and check normalization
    def check_normalization(normalizer_spec, check_cond=None):
        trainer = Trainer(
            solver=solver,
            callbacks=NormalizerDataCallback(
                normalizer=normalizer_spec, stage=stage, apply_to=apply_to
            ),
            max_epochs=1,
            train_size=0.4,
            val_size=0.3,
            test_size=0.3,
            shuffle=False,
        )
        # save a copy of the old trainer datamodule
        trainer_copy = deepcopy(trainer)
        # trigger setup
        trainer_copy.data_module.setup("fit")
        trainer_copy.data_module.setup("test")
        trainer.train()
        trainer.test()
        normalizer_spec = trainer.callbacks[0].normalizer
        for ds_name in stage_map[stage]:
            dataset = getattr(trainer.data_module, ds_name, None)
            old_dataset = getattr(trainer_copy.data_module, ds_name, None)
            for cond in ["data1", "data2"]:
                current_points = dataset.conditions_dict[cond][apply_to]
                old_points = old_dataset.conditions_dict[cond][apply_to]
                if check_cond is None or cond in check_cond:
                    scale = normalizer_spec[cond]["scale"]
                    shift_val = normalizer_spec[cond]["shift"]
                    expected = (old_points - shift_val) / scale
                else:
                    expected = old_points
                print(torch.allclose(current_points, expected))
                assert torch.allclose(current_points, expected)

    # Test full normalizer applied to all conditions
    full_normalizer = {"scale": 0.5, "shift": shift}
    check_normalization(full_normalizer)

    # Test partial normalizer applied to some conditions
    partial_normalizer = {"data1": {"scale": 0.5, "shift": shift}}
    check_normalization(partial_normalizer, check_cond=["data1"])
