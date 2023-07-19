import matplotlib.pyplot as plt
from pina.plotter import Plotter
from pina import Trainer
from pina.model import FeedForward
from pina import PINN
from pina.problem import SpatialProblem
from pina.operators import grad
from pina import Condition, CartesianDomain
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from pina.equation.equation import Equation


import torch


class SimpleODE(SpatialProblem):

    output_variables = ['u']
    spatial_domain = CartesianDomain({'x': [0, 1]})

    # defining the ode equation
    def ode_equation(input_, output_):

        # computing the derivative
        u_x = grad(output_, input_, components=['u'], d=['x'])

        # extracting the u input variable
        u = output_.extract(['u'])

        # calculate the residual and return it
        return u_x - u

    # defining the initial condition
    def initial_condition(input_, output_):

        # setting the initial value
        value = 1.0

        # extracting the u input variable
        u = output_.extract(['u'])

        # calculate the residual and return it
        return u - value

    # conditions to hold
    conditions = {
        'x0': Condition(location=CartesianDomain({'x': 0.}), equation=Equation(initial_condition)),
        'D': Condition(location=CartesianDomain({'x': [0, 1]}), equation=Equation(ode_equation)),
    }

    # sampled points (see below)
    input_pts = None

    # defining the true solution
    def truth_solution(self, pts):
        return torch.exp(pts.extract(['x']))


# initialize the problem
problem = SimpleODE()

# build the model
model = FeedForward(
    layers=[10, 10],
    func=torch.nn.Tanh,
    output_variables=1,
    input_variables=1
)

# create the PINN object
pinn = PINN(problem, model)


# sampling 20 points in [0, 1] through discretization
pinn.problem.discretise_domain(n=20, mode='grid', variables=['x'])

# sampling 20 points in (0, 1) through latin hypercube samping
pinn.problem.discretise_domain(n=20, mode='latin', variables=['x'])

# sampling 20 points in (0, 1) randomly
pinn.problem.discretise_domain(n=20, mode='random', variables=['x'])


# initialize trainer with logger
# trainer = Trainer(pinn)
trainer = Trainer(
    pinn, kwargs={'logger': TensorBoardLogger(save_dir='./ciao')})

# train the model
trainer.train()


# 1D plotting
# if len(pinn.problem.spatial_domain.variables) == 1:
#     pts = pinn.problem.domain.sample(
#         100, 'grid', variables=['x'])  # Generate grid for plotting
#     predicted_output = pinn(pts)

#     # Plotting the predicted output
#     plt.plot(pts.detach(), predicted_output.detach(), label='Predicted')

#     # Plotting the true solution if available
#     if pinn.problem.truth_solution is not None:
#         truth_solution = pinn.problem.truth_solution(pts)
#         plt.plot(pts.detach(), truth_solution.detach(), label='True')

#     plt.xlabel('x')
#     plt.ylabel('u')
#     plt.legend()
#     plt.show()

# # 2D plotting
# elif len(pinn.problem.spatial_domain.variables) == 2:
#     res = 100  # Resolution of the grid
#     v = pinn.problem.spatial_domain.variables

#     # Generate a grid of points for plotting
#     x = torch.linspace(
#         pinn.problem.spatial_domain[v[0]][0], pinn.problem.spatial_domain[v[0]][1], res)
#     y = torch.linspace(
#         pinn.problem.spatial_domain[v[1]][0], pinn.problem.spatial_domain[v[1]][1], res)
#     xx, yy = torch.meshgrid(x, y)
#     pts = torch.stack([xx.flatten(), yy.flatten()], dim=1)

#     # Evaluate the predicted output
#     predicted_output = pinn(pts)

#     # Reshape the output for plotting
#     pred_output_grid = predicted_output.reshape(res, res)

#     # Plotting the predicted output
#     plt.contourf(xx, yy, pred_output_grid.detach(), cmap='viridis')
#     plt.colorbar()
#     plt.xlabel(v[0])
#     plt.ylabel(v[1])
#     plt.show()

# # # TODO plot_loss

print('Done!')
