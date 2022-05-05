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

            cb = getattr(axes[0], method)(*grids_container, predicted_output.reshape(res, res).detach())
            fig.colorbar(cb, ax=axes[0])
            cb = getattr(axes[1], method)(*grids_container, truth_output.reshape(res, res).detach())
            fig.colorbar(cb, ax=axes[1])
            cb = getattr(axes[2], method)(*grids_container, (truth_output-predicted_output.float().flatten()).detach().reshape(res, res))
            fig.colorbar(cb, ax=axes[2])
        else:
            fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(8, 6))
            cb = getattr(axes, method)(*grids_container, predicted_output.reshape(res, res).detach())
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

            cb = getattr(axes[0], method)(*grids_container, predicted_output.reshape(res, res).detach())
            fig.colorbar(cb, ax=axes[0])
            cb = getattr(axes[1], method)(*grids_container, truth_output.reshape(res, res).detach())
            fig.colorbar(cb, ax=axes[1])
            cb = getattr(axes[2], method)(*grids_container, (truth_output-predicted_output.float().flatten()).detach().reshape(res, res))
            fig.colorbar(cb, ax=axes[2])
        else:
            fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(8, 6))
            cb = getattr(axes, method)(*grids_container, predicted_output.reshape(res, res).detach())
            fig.colorbar(cb, ax=axes)



    def plot(self, obj, method='contourf', component='u', parametric=False, params_value=1, filename=None):
        """
        """
        res = 256
        pts = obj.problem.domain.sample(res, 'grid')
        if parametric:
            pts_params = torch.ones(pts.shape[0], len(obj.problem.parameters), dtype=pts.dtype)*params_value
            pts_params = LabelTensor(pts_params, obj.problem.parameters)
            pts = pts.append(pts_params)
        grids_container = [
            pts[:, 0].reshape(res, res),
            pts[:, 1].reshape(res, res),
        ]
        ind_dict = {}
        all_locations = [condition for condition in obj.problem.conditions]
        for location in all_locations:
            if hasattr(obj.problem.conditions[location], 'location'):
                keys_range_ = obj.problem.conditions[location].location.range_.keys()
                if ('x' in keys_range_) and ('y' in keys_range_):
                    range_x = obj.problem.conditions[location].location.range_['x']
                    range_y = obj.problem.conditions[location].location.range_['y']
                    ind_x = np.where(np.logical_or(pts[:, 0]<range_x[0], pts[:, 0]>range_x[1]))
                    ind_y = np.where(np.logical_or(pts[:, 1]<range_y[0], pts[:, 1]>range_y[1]))
                    ind_to_exclude = np.union1d(ind_x, ind_y)
                    ind_dict[location] = ind_to_exclude
        import functools
        from functools import reduce
        final_inds = reduce(np.intersect1d, ind_dict.values())
        predicted_output = obj.model(pts)
        predicted_output = predicted_output.extract([component])
        predicted_output[final_inds] = np.nan
        if hasattr(obj.problem, 'truth_solution'):
            truth_output = obj.problem.truth_solution(*pts.T).float()
            fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(16, 6))

            cb = getattr(axes[0], method)(*grids_container, predicted_output.reshape(res, res).detach())
            fig.colorbar(cb, ax=axes[0])
            cb = getattr(axes[1], method)(*grids_container, truth_output.reshape(res, res).detach())
            fig.colorbar(cb, ax=axes[1])
            cb = getattr(axes[2], method)(*grids_container, (truth_output-predicted_output.float().flatten()).detach().reshape(res, res))
            fig.colorbar(cb, ax=axes[2])
        else:
            fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(8, 6))
            cb = getattr(axes, method)(*grids_container, predicted_output.reshape(res, res).detach())
            fig.colorbar(cb, ax=axes)

        if filename:
            plt.title('Output {} with parameter {}'.format(component, params_value))
            plt.savefig(filename)
        else:
            plt.show()

    def plot_samples(self, obj):
        for location in obj.input_pts:
            pts_x = obj.input_pts[location].extract(['x'])
            pts_y = obj.input_pts[location].extract(['y'])
            plt.plot(pts_x.detach(), pts_y.detach(), '.', label=location)

        plt.legend()
        plt.show()
