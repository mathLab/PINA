import matplotlib.pyplot as plt

# from pina.plotter import Plotter
from pina import Trainer
from pina.model import FeedForward
from pina import PINN
from pina.problem import SpatialProblem
from pina.operators import grad
from pina import Condition, CartesianDomain
from pina.equation.equation import Equation
import lightning.pytorch as pl
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger

import torch


class SimpleODE(SpatialProblem):
    output_variables = ["u"]
    spatial_domain = CartesianDomain({"x": [0, 1]})

    # defining the ode equation
    def ode_equation(input_, output_):
        # computing the derivative
        u_x = grad(output_, input_, components=["u"], d=["x"])

        # extracting the u input variable
        u = output_.extract(["u"])

        # calculate the residual and return it
        return u_x - u

    # defining the initial condition
    def initial_condition(input_, output_):
        # setting the initial value
        value = 1.0

        # extracting the u input variable
        u = output_.extract(["u"])

        # calculate the residual and return it
        return u - value

    # conditions to hold
    conditions = {
        "x0": Condition(
            location=CartesianDomain({"x": 0.0}), equation=Equation(initial_condition)
        ),
        "D": Condition(
            location=CartesianDomain({"x": [0, 1]}), equation=Equation(ode_equation)
        ),
    }

    # sampled points (see below)
    input_pts = None

    # defining the true solution
    def truth_solution(self, pts):
        return torch.exp(pts.extract(["x"]))


# initialize the problem
problem = SimpleODE()

# build the model
model = FeedForward(
    layers=[10, 10], func=torch.nn.Tanh, output_variables=1, input_variables=1
)
print(model)

# create the PINN object
pinn = PINN(problem, model)


# sampling 20 points in [0, 1] through discretization
pinn.problem.discretise_domain(n=20, mode="grid", variables=["x"])

# sampling 20 points in (0, 1) through latin hypercube samping
pinn.problem.discretise_domain(n=20, mode="latin", variables=["x"])

# sampling 20 points in (0, 1) randomly
pinn.problem.discretise_domain(n=20, mode="random", variables=["x"])


# initialize trainer with logger
# trainer = Trainer(pinn)
trainer = Trainer(solver=pinn, kwargs={"default_root_dir": "../checkpoints/"})

# train the model
trainer.train()

checkpoint = torch.load(
    "../checkpoints/lightning_logs/version_0/checkpoints/epoch=999-step=1000.ckpt"
)
print(checkpoint.keys())

# FOR THIS TO WORK, NEED self.save_parameters() in __init__ of solver.py
# print('hyperparams', checkpoint['hyper_parameters']) 

classifier = "_neural_net._model.model."
# state_dict = {
#     key[len(classifier):] if key.startswith(classifier) else key: value
#     for key, value in checkpoint["state_dict"].items()
# }

# checkpoint["state_dict"] = state_dict

torch.save(checkpoint, "../checkpoints/path.ckpt")

# HAVE TO MANUALLY EDIT CHECKPOINT FILES
#checkpoint["state_dict"] = state_dict
#model2 = pl.LightningModule.load_from_checkpoint("../checkpoints/lightning_logs/version_0/checkpoints/epoch=999-step=1000.ckpt")
model2 = PINN.load_from_checkpoint(checkpoint_path="../checkpoints/path.ckpt", problem=problem, model=model)
print(model2)
# model2 = FeedForward(
#     layers=[10, 10], func=torch.nn.Tanh, output_variables=1, input_variables=1
# )

# model2.load_state_dict(state_dict=state_dict)

print("Done!")