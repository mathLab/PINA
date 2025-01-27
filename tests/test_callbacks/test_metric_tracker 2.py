from pina.solvers import PINN
from pina.trainer import Trainer
from pina.model import FeedForward
from pina.callbacks import MetricTracker
from pina.problem.zoo import Poisson2DSquareProblem as Poisson


# make the problem
poisson_problem = Poisson()
boundaries = ['nil_g1', 'nil_g2', 'nil_g3', 'nil_g4']
n = 10
poisson_problem.discretise_domain(n, 'grid', locations=boundaries)
model = FeedForward(len(poisson_problem.input_variables),
                    len(poisson_problem.output_variables))

# make the solver
solver = PINN(problem=poisson_problem, model=model)


def test_metric_tracker_constructor():
    MetricTracker()

def test_metric_tracker_routine():
    # make the trainer
    trainer = Trainer(solver=solver,
                      callbacks=[
                          MetricTracker()
                      ],
                      accelerator='cpu',
                      max_epochs=5)
    trainer.train()
    # get the tracked metrics
    metrics = trainer.callbacks[0].metrics
    # assert the logged metrics are correct
    logged_metrics = sorted(list(metrics.keys()))
    total_metrics = sorted(
        list([key + '_loss' for key in poisson_problem.conditions.keys()])
        + ['mean_loss'])
    assert logged_metrics == total_metrics


