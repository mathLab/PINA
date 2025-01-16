from pina.solvers import PINN
from pina.trainer import Trainer
from pina.model import FeedForward
from pina.callbacks.processing_callbacks import PINAProgressBar
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


def test_progress_bar_constructor():
    PINAProgressBar(['mean_loss'])

def test_progress_bar_routine():
    # make the trainer
    trainer = Trainer(solver=solver,
                      callbacks=[
                          PINAProgressBar(['mean', 'D'])
                      ],
                      accelerator='cpu',
                      max_epochs=5)
    trainer.train()
    # TODO there should be a check that the correct metrics are displayed