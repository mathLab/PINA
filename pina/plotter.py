""" Module for plotting. """
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from pina import LabelTensor
from pina import PINN
from .problem import Problem2D, Problem1D, TimeDependentProblem
#from pina.tdproblem1d import TimeDepProblem1D


class Plotter:

    def _plot_2D(self, obj, method='contourf'):
        """
        """
        if not isinstance(obj, PINN):
            raise RuntimeError

        res = 256
        pts = obj.problem.spatial_domain.discretize(res, 'grid')
        grids_container = [
            pts[:, 0].reshape(res, res),
            pts[:, 1].reshape(res, res),
        ]
        pts = LabelTensor(torch.tensor(pts), obj.problem.input_variables)
        predicted_output = obj.model(pts.tensor)

        if hasattr(obj.problem, 'truth_solution'):
            truth_output = obj.problem.truth_solution(*pts.tensor.T).float()
            fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(16, 6))

            cb = getattr(axes[0], method)(*grids_container, predicted_output.tensor.reshape(res, res).detach())
            fig.colorbar(cb, ax=axes[0])
            cb = getattr(axes[1], method)(*grids_container, truth_output.reshape(res, res).detach())
            fig.colorbar(cb, ax=axes[1])
            cb = getattr(axes[2], method)(*grids_container, (truth_output-predicted_output.tensor.float().flatten()).detach().reshape(res, res))
            fig.colorbar(cb, ax=axes[2])
        else:
            fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(8, 6))
            cb = getattr(axes, method)(*grids_container, predicted_output.tensor.reshape(res, res).detach())
            fig.colorbar(cb, ax=axes)


    def _plot_1D_TimeDep(self, obj, method='contourf'):
        """
        """
        if not isinstance(obj, PINN):
            raise RuntimeError

        res = 256
        grids_container = np.meshgrid(
            obj.problem.spatial_domain.discretize(res, 'grid'),
            obj.problem.temporal_domain.discretize(res, 'grid'),
        )
        pts = np.hstack([
            grids_container[0].reshape(-1, 1),
            grids_container[1].reshape(-1, 1),
        ])
        pts = LabelTensor(torch.tensor(pts), obj.problem.input_variables)
        predicted_output = obj.model(pts.tensor)

        if hasattr(obj.problem, 'truth_solution'):
            truth_output = obj.problem.truth_solution(*pts.tensor.T).float()
            fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(16, 6))

            cb = getattr(axes[0], method)(*grids_container, predicted_output.tensor.reshape(res, res).detach())
            fig.colorbar(cb, ax=axes[0])
            cb = getattr(axes[1], method)(*grids_container, truth_output.reshape(res, res).detach())
            fig.colorbar(cb, ax=axes[1])
            cb = getattr(axes[2], method)(*grids_container, (truth_output-predicted_output.tensor.float().flatten()).detach().reshape(res, res))
            fig.colorbar(cb, ax=axes[2])
        else:
            fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(8, 6))
            cb = getattr(axes, method)(*grids_container, predicted_output.tensor.reshape(res, res).detach())
            fig.colorbar(cb, ax=axes)



    def plot(self, obj, filename=None):
        """
        """
        if isinstance(obj.problem, (TimeDependentProblem, Problem1D)): #  time-dep 1D
            self._plot_1D_TimeDep(obj, method='pcolor')
        elif isinstance(obj.problem, Problem2D):            #  2D
            self._plot_2D(obj, method='pcolor')

        if filename:
            plt.savefig(filename)
        else:
            plt.show()
