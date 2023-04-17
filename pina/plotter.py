""" Module for plotting. """
import matplotlib
# matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from pina import LabelTensor
from pina import PINN
from .problem import SpatialProblem, TimeDependentProblem
#from pina.tdproblem1d import TimeDepProblem1D


class Plotter:
    """
    Implementation of a plotter class, for easy visualizations.
    """

    def plot_samples(self, pinn, variables=None):
        """
            Plot a sample of solution.

        :param pinn: the PINN object.
        :type pinn: PINN
        :param variables: pinn variable domains: spatial or temporal,
            defaults to None.
        :type variables: str, optional

        :Example:
            >>> plotter = Plotter()
            >>> plotter.plot_samples(pinn=pinn, variables='spatial')
        """

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
        """Plot solution for one dimensional function

        :param pts: Points to plot the solution.
        :type pts: torch.Tensor
        :param pred: PINN solution evaluated at 'pts'.
        :type pred: torch.Tensor
        :param method: not used, kept for code compatibility
        :type method: None
        :param truth_solution: Real solution evaluated at 'pts',
            defaults to None.
        :type truth_solution: torch.Tensor, optional
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
        """Plot solution for two dimensional function

        :param pts: Points to plot the solution.
        :type pts: torch.Tensor
        :param pred: PINN solution evaluated at 'pts'.
        :type pred: torch.Tensor
        :param method: matplotlib method to plot 2-dimensional data,
            see https://matplotlib.org/stable/api/axes_api.html for
            reference.
        :type method: str
        :param truth_solution: Real solution evaluated at 'pts',
            defaults to None.
        :type truth_solution: torch.Tensor, optional
        """

        grids = [p_.reshape(res, res) for p_ in pts.extract(v).cpu().T]

        pred_output = pred.reshape(res, res)
        if truth_solution:
            truth_output = truth_solution(pts).float().reshape(res, res)
            fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(16, 6))

            cb = getattr(ax[0], method)(
                *grids, pred_output.cpu().detach(), **kwargs)
            fig.colorbar(cb, ax=ax[0])
            cb = getattr(ax[1], method)(
                *grids, truth_output.cpu().detach(), **kwargs)
            fig.colorbar(cb, ax=ax[1])
            cb = getattr(ax[2], method)(*grids,
                                        (truth_output-pred_output).cpu().detach(),
                                        **kwargs)
            fig.colorbar(cb, ax=ax[2])
        else:
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 6))
            cb = getattr(ax, method)(
                *grids, pred_output.cpu().detach(), **kwargs)
            fig.colorbar(cb, ax=ax)

    def plot(self, pinn, components=None, fixed_variables={}, method='contourf',
             res=256, filename=None, **kwargs):
        """
            Plot sample of PINN output.

        :param pinn: the PINN object.
        :type pinn: PINN
        :param components: function components to plot, defaults to None.
        :type components: list['str'], optional
        :param fixed_variables: function variables to be kept fixed during
            plotting passed as a dict where the dict-key is the variable
            and the dict-value is the value to be kept fixed, defaults to {}.
        :type fixed_variables: dict, optional
        :param method: matplotlib method to plot the solution,
            defaults to 'contourf'.
        :type method: str, optional
        :param res: number of points used for plotting in each axis,
            defaults to 256.
        :type res: int, optional
        :param filename: file name to save the plot, defaults to None
        :type filename: str, optional
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
            Plot the loss function values during traininig.

        :param pinn: the PINN object.
        :type pinn: PINN
        :param label: matplolib label, defaults to None
        :type label: str, optional
        :param log_scale: use of log scale in plotting, defaults to True.
        :type log_scale: bool, optional
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
