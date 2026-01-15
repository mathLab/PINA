from pina.solver import PINN
from pina.trainer import Trainer
from pina.model import FeedForward
from pina.callback import MetricTracker
from pina.problem.zoo import Poisson2DSquareProblem as Poisson


# make the problem
poisson_problem = Poisson()
n = 10
poisson_problem.discretise_domain(n, "grid", domains="boundary")
poisson_problem.discretise_domain(n, "grid", domains="D")
model = FeedForward(
    len(poisson_problem.input_variables), len(poisson_problem.output_variables)
)

# make the solver
solver = PINN(problem=poisson_problem, model=model)


def test_metric_tracker_constructor():
    MetricTracker()


def test_metric_tracker_routine():
    # make the trainer
    trainer = Trainer(
        solver=solver,
        callbacks=[MetricTracker()],
        accelerator="cpu",
        max_epochs=5,
        log_every_n_steps=1,
    )
    trainer.train()
    # get the tracked metrics
    metrics = trainer.callbacks[0].metrics
    # assert the logged metrics are correct
    logged_metrics = sorted(list(metrics.keys()))
    assert logged_metrics == ["train_loss"]
