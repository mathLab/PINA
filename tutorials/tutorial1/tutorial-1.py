#!/usr/bin/env python
# coding: utf-8

# This tutorial presents how to solve with Physics-Informed Neural Networks a 2-D Poisson problem with Dirichlet boundary conditions.
# We consider a Poisson problem with a sinusoidal forcing term, in the square domain D = [0, 1]*[0, 1], with boundaries gamma1, gamma2, gamma3, gamma4.
# First of all, some useful imports.

import os
import numpy as np
import argparse
import sys
import torch
from torch.nn import ReLU, Tanh, Softplus
from pina.problem import SpatialProblem
from pina.operators import nabla
from pina.model import FeedForward
from pina.adaptive_functions import AdaptiveSin, AdaptiveCos, AdaptiveTanh
from pina import Condition, Span, PINN, LabelTensor, Plotter

# Now, the Poisson problem is written in PINA code as a class. The equations are written as that should be satisfied in the corresponding domains. truth_solution is the exact solution which will be compared with the predicted one.

class Poisson(SpatialProblem):
    output_variables = ['u']
    spatial_domain = Span({'x': [0, 1], 'y': [0, 1]})

    def laplace_equation(input_, output_):
        force_term = (torch.sin(input_.extract(['x'])*torch.pi) *
                      torch.sin(input_.extract(['y'])*torch.pi))
        nabla_u = nabla(output_.extract(['u']), input_)
        return nabla_u - force_term

    def nil_dirichlet(input_, output_):
        value = 0.0
        return output_.extract(['u']) - value

    conditions = {
        'gamma1': Condition(Span({'x': [0, 1], 'y':  1}), nil_dirichlet),
        'gamma2': Condition(Span({'x': [0, 1], 'y': 0}), nil_dirichlet),
        'gamma3': Condition(Span({'x':  1, 'y': [0, 1]}), nil_dirichlet),
        'gamma4': Condition(Span({'x': 0, 'y': [0, 1]}), nil_dirichlet),
        'D': Condition(Span({'x': [0, 1], 'y': [0, 1]}), laplace_equation),
    }

    def poisson_sol(self, pts):
        return -(
            torch.sin(pts.extract(['x'])*torch.pi)*
            torch.sin(pts.extract(['y'])*torch.pi)
        )/(2*torch.pi**2)
        #return -(np.sin(x*np.pi)*np.sin(y*np.pi))/(2*np.pi**2)

    truth_solution = poisson_sol

# Then, a feed-forward neural network is defined, through the class FeedForward. A 2-D grid is instantiated inside the square domain and on the boundaries. This neural network takes as input the coordinates of the points which compose the grid and gives as output the solution of the Poisson problem. The residual of the equations are evaluated at each point of the grid and the loss minimized by the neural network is the sum of the residuals.
# In this tutorial, the neural network is composed by two hidden layers of 10 neurons each, and it is trained for 5000 epochs with a learning rate of 0.003. These parameters can be modified as desired.
# The output of the cell below is the final loss of the training phase of the PINN.

poisson_problem = Poisson()

model = FeedForward(layers=[10, 10],
                    output_variables=poisson_problem.output_variables,
                    input_variables=poisson_problem.input_variables)

pinn = PINN(poisson_problem, model, lr=0.003, regularizer=1e-8)
pinn.span_pts(20, 'grid', ['D'])
pinn.span_pts(20, 'grid', ['gamma1', 'gamma2', 'gamma3', 'gamma4'])
pinn.train(5000, 100)


# # The loss trend is saved in a dedicated txt file located in 'tutorial1_files'.

os.mkdir('tutorial1_files')
with open('tutorial1_files/poisson_history.txt', 'w') as file_:
    for i, losses in enumerate(pinn.history):
        file_.write('{} {}\n'.format(i, sum(losses)))
pinn.save_state('tutorial1_files/pina.poisson')


# # Now the Plotter class is used to plot the results.
# # The solution predicted by the neural network is plotted on the left, the exact one is represented at the center and on the right the error between the exact and the predicted solutions is showed. 


plotter = Plotter()
plotter.plot(pinn)

# Now, the same problem is solved in a different way.
# A new neural network is now defined, with an additional input variable, named extra-feature, which coincides with the forcing term in the Laplace equation. 
# The set of input variables to the neural network is:
# [x, y, k(x,y)],
# where x and y are the coordinates of the points of the grid and k(x, y) is the forcing term evaluated at the grid points. 
# This forcing term is initialized in the class 'myFeature', the output of the cell below is also in this case the final loss of PINN.


class myFeature(torch.nn.Module):
    """
    Feature: sin(x)
    """

    def __init__(self):
        super(myFeature, self).__init__()

    def forward(self, x):
        t = (torch.sin(x.extract(['x'])*torch.pi) *
             torch.sin(x.extract(['y'])*torch.pi))
        return LabelTensor(t, ['sin(x)sin(y)'])
        
feat = [myFeature()]# if args.features else []

poisson_problem = Poisson()
model_feat = FeedForward(
        layers=[20, 20],
        output_variables=poisson_problem.output_variables,
        input_variables=poisson_problem.input_variables,
        func=Softplus,
        extra_features=feat
    )

pinn_feat = PINN(
        poisson_problem,
        model_feat,
        lr=0.03,
        error_norm='mse',
        regularizer=1e-8)

pinn_feat.span_pts(20, 'grid', locations=['gamma1', 'gamma2', 'gamma3', 'gamma4'])
pinn_feat.span_pts(20, 'grid', locations=['D'])
pinn_feat.train(5000, 100)
pinn_feat.save_state('pina.poisson')


# The losses are saved in a txt file as for the basic Poisson case.

with open('tutorial1_files/poisson_history_feat.txt', 'w') as file_:
        for i, losses in enumerate(pinn_feat.history):
            file_.write('{} {}\n'.format(i, sum(losses)))
pinn_feat.save_state('tutorial1_files/pina.poisson_feat')


# The predicted and exact solutions and the error between them are represented below.


plotter_feat = Plotter()
plotter_feat.plot(pinn_feat)


# Another way to predict the solution is to add a parametric forcing term of the Laplace equation as an extra-feature. The parameters added in the expression of the extra-feature are learned during the training phase of the neural network.
# The new Poisson problem is defined in the dedicated class 'ParametricPoisson', where the domain is no more only spatial, but includes the parameters' space. In our case, the parameters' bounds are 0 and 30. 


from pina.problem import ParametricProblem

class ParametricPoisson(SpatialProblem, ParametricProblem):
    bounds_x = [0, 1]
    bounds_y = [0, 1]
    bounds_alpha = [0, 30]
    bounds_beta = [0, 30]
    spatial_variables = ['x', 'y']
    parameters = ['alpha', 'beta']
    output_variables = ['u']
    spatial_domain = Span({'x': bounds_x, 'y': bounds_y})
    parameter_domain = Span({'alpha': bounds_alpha, 'beta': bounds_beta})

    def laplace_equation(input_, output_):
        force_term = (torch.sin(input_.extract(['x'])*torch.pi) *
                      torch.sin(input_.extract(['y'])*torch.pi))
        nabla_u = nabla(output_.extract(['u']), input_)
        return nabla_u - force_term

    def nil_dirichlet(input_, output_):
        value = 0.0
        return output_.extract(['u']) - value

    conditions = {
        'gamma1': Condition(
            Span({'x': bounds_x, 'y': bounds_y[1], 'alpha': bounds_alpha, 'beta': bounds_beta}),
            nil_dirichlet),
        'gamma2': Condition(
            Span({'x': bounds_x, 'y': bounds_y[0], 'alpha': bounds_alpha, 'beta': bounds_beta}),
            nil_dirichlet),
        'gamma3': Condition(
            Span({'x': bounds_x[1], 'y': bounds_y, 'alpha': bounds_alpha, 'beta': bounds_beta}),
            nil_dirichlet),
        'gamma4': Condition(
            Span({'x': bounds_x[0], 'y': bounds_y, 'alpha': bounds_alpha, 'beta': bounds_beta}),
            nil_dirichlet),
        'D': Condition(
            Span({'x': bounds_x, 'y': bounds_y, 'alpha': bounds_alpha, 'beta': bounds_beta}),
            laplace_equation),
    }

    def poisson_sol(self, pts):
        return -(
            torch.sin(pts.extract(['x'])*torch.pi)*
            torch.sin(pts.extract(['y'])*torch.pi)
        )/(2*torch.pi**2)


# # Here, as done for the other cases, the new parametric feature is defined and the neural network is re-initialized and trained, considering as two additional parameters alpha and beta. 


param_poisson_problem = ParametricPoisson()


class myFeature(torch.nn.Module):
    """
    """
    def __init__(self):
        super(myFeature, self).__init__()

    def forward(self, x):
        t =  (x.extract(['beta'])*torch.sin(x.extract(['alpha'])*x.extract(['x'])*torch.pi)*
               torch.sin(x.extract(['alpha'])*x.extract(['y'])*torch.pi))
        return LabelTensor(t, ['b*sin(a*x)sin(a*y)'])

feat = [myFeature()]
model_learn = FeedForward(layers=[10, 10],
                    output_variables=param_poisson_problem.output_variables,
                    input_variables=param_poisson_problem.input_variables,
                    extra_features=feat)

pinn_learn = PINN(poisson_problem, model_feat, lr=0.003, regularizer=1e-8)
pinn_learn.span_pts(20, 'grid', ['D'])
pinn_learn.span_pts(20, 'grid', ['gamma1', 'gamma2', 'gamma3', 'gamma4'])
pinn_learn.train(5000, 100)


# The losses are saved as for the other two cases trained above.

with open('tutorial1_files/poisson_history_learn_feat.txt', 'w') as file_:
    for i, losses in enumerate(pinn_learn.history):
        file_.write('{} {}\n'.format(i, sum(losses)))
pinn_learn.save_state('tutorial1_files/pina.poisson_learn_feat')


# Here the plots for the prediction error (below on the right) shows that the prediction coming from the parametric PINN is more accurate than the one of the basic version of PINN.

plotter_learn = Plotter()
plotter_learn.plot(pinn_learn)


# Now the files containing the loss trends for the three cases are read. The loss histories are compared; we can see that the loss decreases faster in the cases of PINN with extra-feature.


import pandas as pd

df = pd.read_csv("tutorial1_files/poisson_history.txt", sep=" ", header=None)
epochs = df[0]
poisson_data = epochs.to_numpy()*100
basic = df[1].to_numpy()

df_feat = pd.read_csv("tutorial1_files/poisson_history_feat.txt", sep=" ", header=None)
feat = df_feat[1].to_numpy()

df_learn = pd.read_csv("tutorial1_files/poisson_history_learn_feat.txt", sep=" ", header=None)
learn_feat = df_learn[1].to_numpy()

import matplotlib.pyplot as plt
plt.semilogy(epochs, basic, label='Basic PINN')
plt.semilogy(epochs, feat, label='PINN with extra-feature')
plt.semilogy(epochs, learn_feat, label='PINN with learnable extra-feature')
plt.legend()
plt.grid()
plt.show()

