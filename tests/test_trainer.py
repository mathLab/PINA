import pytest
from pina import Trainer
from pina.solver import PINN
from pina.model import FeedForward
from pina.problem.zoo import Poisson2DSquareProblem


# Define the problem, the model and the solver for testing purposes
problem = Poisson2DSquareProblem()
problem.discretise_domain(n=10, mode="random")
model = FeedForward(len(problem.input_variables), len(problem.output_variables))
solver = PINN(model=model, problem=problem)


@pytest.mark.parametrize("batching_mode", Trainer._AVAIL_BATCHING_MODES)
@pytest.mark.parametrize("automatic_batching", [True, False])
@pytest.mark.parametrize("pin_memory", [True, False])
@pytest.mark.parametrize("shuffle", [True, False])
@pytest.mark.parametrize("compile", [True, False])
@pytest.mark.parametrize("batch_size", [None, 5])
@pytest.mark.parametrize(
    "train_size, test_size, val_size", [(0.8, 0.1, 0.1), (0.7, 0.2, 0.1)]
)
def test_constructor(
    batch_size,
    train_size,
    test_size,
    val_size,
    compile,
    batching_mode,
    automatic_batching,
    pin_memory,
    shuffle,
):

    Trainer(
        solver=solver,
        batch_size=batch_size,
        train_size=train_size,
        test_size=test_size,
        val_size=val_size,
        compile=compile,
        batching_mode=batching_mode if batch_size else "common_batch_size",
        automatic_batching=automatic_batching,
        num_workers=0,
        pin_memory=pin_memory if batch_size else False,
        shuffle=shuffle,
    )

    # Should raise ValueError if solver is not an instance of SolverInterface
    with pytest.raises(ValueError):
        Trainer(
            solver="not_a_solver",
            batch_size=batch_size,
            train_size=train_size,
            test_size=test_size,
            val_size=val_size,
            compile=compile,
            batching_mode=batching_mode if batch_size else "common_batch_size",
            automatic_batching=automatic_batching,
            num_workers=0,
            pin_memory=pin_memory if batch_size else False,
            shuffle=shuffle,
        )

    # Should raise ValueError if train_size + test_size + val_size != 1.0
    with pytest.raises(ValueError):
        Trainer(
            solver=solver,
            batch_size=batch_size,
            train_size=0.5,
            test_size=0.3,
            val_size=0.3,
            compile=compile,
            batching_mode=batching_mode if batch_size else "common_batch_size",
            automatic_batching=automatic_batching,
            num_workers=0,
            pin_memory=pin_memory if batch_size else False,
            shuffle=shuffle,
        )

    # Should raise ValueError if compile is not a boolean
    with pytest.raises(ValueError):
        Trainer(
            solver=solver,
            batch_size=batch_size,
            train_size=train_size,
            test_size=test_size,
            val_size=val_size,
            compile="not_a_boolean",
            batching_mode=batching_mode if batch_size else "common_batch_size",
            automatic_batching=automatic_batching,
            num_workers=0,
            pin_memory=pin_memory if batch_size else False,
            shuffle=shuffle,
        )

    # Should raise ValueError if automatic_batching is not a boolean
    with pytest.raises(ValueError):
        Trainer(
            solver=solver,
            batch_size=batch_size,
            train_size=train_size,
            test_size=test_size,
            val_size=val_size,
            compile=compile,
            batching_mode=batching_mode if batch_size else "common_batch_size",
            automatic_batching="not_a_boolean",
            num_workers=0,
            pin_memory=pin_memory if batch_size else False,
            shuffle=shuffle,
        )

    # Should raise ValueError if shuffle is not a boolean
    with pytest.raises(ValueError):
        Trainer(
            solver=solver,
            batch_size=batch_size,
            train_size=train_size,
            test_size=test_size,
            val_size=val_size,
            compile=compile,
            batching_mode=batching_mode if batch_size else "common_batch_size",
            automatic_batching=automatic_batching,
            num_workers=0,
            pin_memory=pin_memory if batch_size else False,
            shuffle="not_a_boolean",
        )

    # Should raise ValueError if pin_memory is not a boolean
    with pytest.raises(ValueError):
        Trainer(
            solver=solver,
            batch_size=batch_size,
            train_size=train_size,
            test_size=test_size,
            val_size=val_size,
            compile=compile,
            batching_mode=batching_mode if batch_size else "common_batch_size",
            automatic_batching=automatic_batching,
            num_workers=0,
            pin_memory="not_a_boolean",
            shuffle=shuffle,
        )

    # Should raise ValueError if num_workers is negative
    with pytest.raises(AssertionError):
        Trainer(
            solver=solver,
            batch_size=batch_size,
            train_size=train_size,
            test_size=test_size,
            val_size=val_size,
            compile=compile,
            batching_mode=batching_mode if batch_size else "common_batch_size",
            automatic_batching=automatic_batching,
            num_workers=-1,
            pin_memory=pin_memory if batch_size else False,
            shuffle=shuffle,
        )

    # Should raise ValueError if batch_size is not a positive integer
    with pytest.raises(AssertionError):
        Trainer(
            solver=solver,
            batch_size=-1,
            train_size=train_size,
            test_size=test_size,
            val_size=val_size,
            compile=compile,
            batching_mode=batching_mode if batch_size else "common_batch_size",
            automatic_batching=automatic_batching,
            num_workers=0,
            pin_memory=pin_memory if batch_size else False,
            shuffle=shuffle,
        )

    # Should raise ValueError if an invalid batching mode is provided
    with pytest.raises(ValueError):
        Trainer(
            solver=solver,
            batch_size=batch_size,
            train_size=train_size,
            test_size=test_size,
            val_size=val_size,
            compile=compile,
            batching_mode="invalid_mode",
            automatic_batching=automatic_batching,
            num_workers=0,
            pin_memory=pin_memory if batch_size else False,
            shuffle=shuffle,
        )

    # Should raise RuntimeError if any domain has not been discretised
    with pytest.raises(RuntimeError):

        # Create a new problem without discretising the domain
        new_problem = Poisson2DSquareProblem()
        new_solver = PINN(model=model, problem=new_problem)

        Trainer(
            solver=new_solver,
            batch_size=batch_size,
            train_size=train_size,
            test_size=test_size,
            val_size=val_size,
            compile=compile,
            batching_mode=batching_mode if batch_size else "common_batch_size",
            automatic_batching=automatic_batching,
            num_workers=0,
            pin_memory=pin_memory if batch_size else False,
            shuffle=shuffle,
        )
