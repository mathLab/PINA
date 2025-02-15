from pina.solvers import PINN
from pina.trainer import Trainer
from pina.model import FeedForward
from pina.problem.zoo import Poisson2DSquareProblem as Poisson
from pina.callbacks import R3Refinement


# make the problem
poisson_problem = Poisson()
boundaries = ['g1', 'g2', 'g3', 'g4']
n = 10
poisson_problem.discretise_domain(n, 'grid', domains=boundaries)
poisson_problem.discretise_domain(n, 'grid', domains='D')
model = FeedForward(len(poisson_problem.input_variables),
                    len(poisson_problem.output_variables))

# make the solver
solver = PINN(problem=poisson_problem, model=model)


# def test_r3constructor():
#     R3Refinement(sample_every=10)


# def test_r3refinment_routine():
#     # make the trainer
#     trainer = Trainer(solver=solver,
#                       callbacks=[R3Refinement(sample_every=1)],
#                       accelerator='cpu',
#                       max_epochs=5)
#     trainer.train()

# def test_r3refinment_routine():
#     model = FeedForward(len(poisson_problem.input_variables),
#                     len(poisson_problem.output_variables))
#     solver = PINN(problem=poisson_problem, model=model)
#     trainer = Trainer(solver=solver,
#                       callbacks=[R3Refinement(sample_every=1)],
#                       accelerator='cpu',
#                       max_epochs=5)
#     before_n_points = {loc : len(pts) for loc, pts in trainer.solver.problem.input_pts.items()}
#     trainer.train()
#     after_n_points = {loc : len(pts) for loc, pts in trainer.solver.problem.input_pts.items()}
#     assert before_n_points == after_n_points
