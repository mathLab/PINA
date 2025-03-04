import pytest
import math
from pina.solver import PINN
from pina.loss import ScalarWeighting
from pina.trainer import Trainer
from pina.model import FeedForward
from pina.problem.zoo import Poisson2DSquareProblem as Poisson
from pina.callback import LinearWeightUpdate


# Define the problem
poisson_problem = Poisson()
poisson_problem.discretise_domain(50, "grid")
cond_name = list(poisson_problem.conditions.keys())[0]

# Define the model
model = FeedForward(
    input_dimensions=len(poisson_problem.input_variables),
    output_dimensions=len(poisson_problem.output_variables),
    layers=[32, 32],
)

# Define the weighting schema
weights_dict = {key: 1 for key in poisson_problem.conditions.keys()}
weighting = ScalarWeighting(weights=weights_dict)

# Define the solver
solver = PINN(problem=poisson_problem, model=model, weighting=weighting)

# Value used for testing
epochs = 10


@pytest.mark.parametrize("initial_value", [1, 5.5])
@pytest.mark.parametrize("target_value", [10, 25.5])
def test_constructor(initial_value, target_value):
    LinearWeightUpdate(
        target_epoch=epochs,
        condition_name=cond_name,
        initial_value=initial_value,
        target_value=target_value,
    )

    # Target_epoch must be int
    with pytest.raises(ValueError):
        LinearWeightUpdate(
            target_epoch=10.0,
            condition_name=cond_name,
            initial_value=0,
            target_value=1,
        )

    # Condition_name must be str
    with pytest.raises(ValueError):
        LinearWeightUpdate(
            target_epoch=epochs,
            condition_name=100,
            initial_value=0,
            target_value=1,
        )

    # Initial_value must be float or int
    with pytest.raises(ValueError):
        LinearWeightUpdate(
            target_epoch=epochs,
            condition_name=cond_name,
            initial_value="0",
            target_value=1,
        )

    # Target_value must be float or int
    with pytest.raises(ValueError):
        LinearWeightUpdate(
            target_epoch=epochs,
            condition_name=cond_name,
            initial_value=0,
            target_value="1",
        )


@pytest.mark.parametrize("initial_value, target_value", [(1, 10), (10, 1)])
def test_training(initial_value, target_value):
    callback = LinearWeightUpdate(
        target_epoch=epochs,
        condition_name=cond_name,
        initial_value=initial_value,
        target_value=target_value,
    )
    trainer = Trainer(
        solver=solver,
        callbacks=[callback],
        accelerator="cpu",
        max_epochs=epochs,
    )
    trainer.train()

    # Check that the final weight value matches the target value
    final_value = solver.weighting.weights[cond_name]
    assert math.isclose(final_value, target_value)

    # Target_epoch must be greater than 0
    with pytest.raises(ValueError):
        callback = LinearWeightUpdate(
            target_epoch=0,
            condition_name=cond_name,
            initial_value=0,
            target_value=1,
        )
        trainer = Trainer(
            solver=solver,
            callbacks=[callback],
            accelerator="cpu",
            max_epochs=5,
        )
        trainer.train()

    # Target_epoch must be less than or equal to max_epochs
    with pytest.raises(ValueError):
        callback = LinearWeightUpdate(
            target_epoch=epochs,
            condition_name=cond_name,
            initial_value=0,
            target_value=1,
        )
        trainer = Trainer(
            solver=solver,
            callbacks=[callback],
            accelerator="cpu",
            max_epochs=epochs - 1,
        )
        trainer.train()

    # Condition_name must be a problem condition
    with pytest.raises(ValueError):
        callback = LinearWeightUpdate(
            target_epoch=epochs,
            condition_name="not_a_condition",
            initial_value=0,
            target_value=1,
        )
        trainer = Trainer(
            solver=solver,
            callbacks=[callback],
            accelerator="cpu",
            max_epochs=epochs,
        )
        trainer.train()

    # Weighting schema must be ScalarWeighting
    with pytest.raises(ValueError):
        callback = LinearWeightUpdate(
            target_epoch=epochs,
            condition_name=cond_name,
            initial_value=0,
            target_value=1,
        )
        unweighted_solver = PINN(problem=poisson_problem, model=model)
        trainer = Trainer(
            solver=unweighted_solver,
            callbacks=[callback],
            accelerator="cpu",
            max_epochs=epochs,
        )
        trainer.train()
