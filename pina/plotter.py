""" Module for plotting. """
import matplotlib
#matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from pina import LabelTensor
from pina import PINN
from .problem import SpatialProblem, TimeDependentProblem
#from pina.tdproblem1d import TimeDepProblem1D


class Plotter:

    def plot_samples(self, pinn, variables=None):

        if variables is None:
            variables = pinn.problem.domain.variables
        elif variables == 'spatial':
            variables = pinn.problem.spatial_domain.variables
        elif variables == 'temporal':
            variables = pinn.problem.temporal_domain.variables

        if len(variables) not in [1, 2, 3]:
            raise ValueError

        fig = plt.figure()
        proj = '3d' if len(variables) == 3 else None
        ax = fig.add_subplot(projection=proj)
        for location in pinn.input_pts:
            coords = pinn.input_pts[location].extract(variables).T.detach()
            if coords.shape[0] == 1:  # 1D samples
                ax.plot(coords[0], torch.zeros(coords[0].shape), '.',
                        label=location)
            else:
                ax.plot(*coords, '.', label=location)

        ax.set_xlabel(variables[0])
        try:
            ax.set_ylabel(variables[1])
        except:
            pass

        try:
            ax.set_zlabel(variables[2])
        except:
            pass

        plt.legend()
        plt.show()

    def _1d_plot(self, pts, pred, method, truth_solution=None, **kwargs):
        """
        """
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))

        ax.plot(pts, pred.detach(), **kwargs)

        if truth_solution:
            truth_output = truth_solution(pts).float()
            ax.plot(pts, truth_output.detach(), **kwargs)

        plt.xlabel(pts.labels[0])
        plt.ylabel(pred.labels[0])
        plt.show()

    def _2d_plot(self, pts, pred, v, res, method, truth_solution=None,
                 **kwargs):
        """
        """

        grids = [p_.reshape(res, res) for p_ in pts.extract(v).cpu().T]

        pred_output = pred.reshape(res, res)
        if truth_solution:
            truth_output = truth_solution(pts).float().reshape(res, res)
            fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(16, 6))

            cb = getattr(ax[0], method)(*grids, pred_output.cpu().detach(), **kwargs)
            fig.colorbar(cb, ax=ax[0])
            cb = getattr(ax[1], method)(*grids, truth_output.cpu().detach(), **kwargs)
            fig.colorbar(cb, ax=ax[1])
            cb = getattr(ax[2], method)(*grids,
                                        (truth_output-pred_output).cpu().detach(),
                                        **kwargs)
            fig.colorbar(cb, ax=ax[2])
        else:
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 6))
            cb = getattr(ax, method)(*grids, pred_output.cpu().detach(), **kwargs)
            fig.colorbar(cb, ax=ax)


    def plot(self, pinn, components=None, fixed_variables={}, method='contourf',
             res=256, filename=None, **kwargs):
        """
        """
        if components is None:
            components = [pinn.problem.output_variables]
        v = [
            var for var in pinn.problem.input_variables
            if var not in fixed_variables.keys()
        ]
        pts = pinn.problem.domain.sample(res, 'grid', variables=v)

        fixed_pts = torch.ones(pts.shape[0], len(fixed_variables))
        fixed_pts *= torch.tensor(list(fixed_variables.values()))
        fixed_pts = fixed_pts.as_subclass(LabelTensor)
        fixed_pts.labels = list(fixed_variables.keys())

        pts = pts.append(fixed_pts)
        pts = pts.to(device=pinn.device)

        predicted_output = pinn.model(pts)
        if isinstance(components, str):
            predicted_output = predicted_output.extract(components)
        elif callable(components):
            predicted_output = components(predicted_output)

        truth_solution = getattr(pinn.problem, 'truth_solution', None)
        if len(v) == 1:
            self._1d_plot(pts, predicted_output, method, truth_solution,
                          **kwargs)
        elif len(v) == 2:
            self._2d_plot(pts, predicted_output, v, res, method,
                          truth_solution, **kwargs)

        if filename:
            plt.title('Output {} with parameter {}'.format(components,
                                                           fixed_variables))
            plt.savefig(filename)
        else:
            plt.show()

    def plot_loss(self, pinn, label=None, log_scale=True):
        """
        Plot the loss trend

        TODO
        """

        if not label:
            label = str(pinn)

        epochs = list(pinn.history_loss.keys())
        loss = np.array(list(pinn.history_loss.values()))
        if loss.ndim != 1:
            loss = loss[:, 0]

        plt.plot(epochs, loss, label=label)
        if log_scale:
            plt.yscale('log')
