from pina.solver import PINN
from pina.trainer import Trainer
from pina.model import FeedForward
from pina.callback import PINAProgressBar
from pina.problem.zoo import Poisson2DSquareProblem as Poisson


# make the problem
poisson_problem = Poisson()
n = 10
condition_names = list(poisson_problem.conditions.keys())
poisson_problem.discretise_domain(n, "grid", domains="boundary")
poisson_problem.discretise_domain(n, "grid", domains="D")
model = FeedForward(
    len(poisson_problem.input_variables), len(poisson_problem.output_variables)
)

# make the solver
solver = PINN(problem=poisson_problem, model=model)


def test_progress_bar_constructor():
    PINAProgressBar()


def test_progress_bar_routine():
    # make the trainer
    trainer = Trainer(
        solver=solver,
        callbacks=[PINAProgressBar(["val", condition_names[0]])],
        accelerator="cpu",
        max_epochs=5,
    )
    trainer.train()
    # TODO there should be a check that the correct metrics are displayed
