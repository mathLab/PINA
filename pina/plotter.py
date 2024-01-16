""" Module for plotting. """

import matplotlib.pyplot as plt
import torch
from pina.callbacks import MetricTracker
from pina import LabelTensor


class Plotter:
    """
    Implementation of a plotter class, for easy visualizations.
    """

    def plot_samples(self, problem, variables=None, filename=None, **kwargs):
        """
        Plot the training grid samples.

        :param AbstractProblem problem: The PINA problem from where to plot the domain.
        :param list(str) variables: Variables to plot. If None, all variables
            are plotted. If 'spatial', only spatial variables are plotted. If
            'temporal', only temporal variables are plotted. Defaults to None.
        :param str filename: The file name to save the plot. If None, the plot
            is shown using the setted matplotlib frontend. Default is None.

        .. todo::
            - Add support for 3D plots.
            - Fix support for more complex problems.

        :Example:
            >>> plotter = Plotter()
            >>> plotter.plot_samples(problem=problem, variables='spatial')
        """

        if variables is None:
            variables = problem.domain.variables
        elif variables == 'spatial':
            variables = problem.spatial_domain.variables
        elif variables == 'temporal':
            variables = problem.temporal_domain.variables

        if len(variables) not in [1, 2, 3]:
            raise ValueError

        fig = plt.figure()
        proj = '3d' if len(variables) == 3 else None
        ax = fig.add_subplot(projection=proj)
        for location in problem.input_pts:
            coords = problem.input_pts[location].extract(variables).T.detach()
            if coords.shape[0] == 1:  # 1D samples
                ax.plot(coords.flatten(),
                        torch.zeros(coords.flatten().shape),
                        '.',
                        label=location,
                        **kwargs)
            else:
                ax.plot(*coords, '.', label=location, **kwargs)

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
        if filename:
            plt.savefig(filename)
            plt.close()
        else:
            plt.show()

    def _1d_plot(self, pts, pred, v, method, truth_solution=None, **kwargs):
        """Plot solution for one dimensional function

        :param pts: Points to plot the solution.
        :type pts: torch.Tensor
        :param pred: SolverInterface solution evaluated at 'pts'.
        :type pred: torch.Tensor
        :param v: Fixed variables when plotting the solution.
        :type v: torch.Tensor
        :param method: Not used, kept for code compatibility
        :type method: None
        :param truth_solution: Real solution evaluated at 'pts',
            defaults to None.
        :type truth_solution: torch.Tensor, optional
        """
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))

        ax.plot(pts.extract(v), pred, label='Neural Network solution', **kwargs)

        if truth_solution:
            truth_output = truth_solution(pts).detach()
            ax.plot(pts.extract(v), truth_output, label='True solution', **kwargs)

        # TODO: pred is a torch.Tensor, so no labels is available
        #      extra variable for labels should be
        #      passed in the function arguments.
        # plt.ylabel(pred.labels[0]) 
        plt.legend()

    def _2d_plot(self,
                 pts,
                 pred,
                 v,
                 res,
                 method,
                 truth_solution=None,
                 **kwargs):
        """Plot solution for two dimensional function

        :param pts: Points to plot the solution.
        :type pts: torch.Tensor
        :param pred: ``SolverInterface`` solution evaluated at 'pts'.
        :type pred: torch.Tensor
        :param v: Fixed variables when plotting the solution.
        :type v: torch.Tensor
        :param method: Matplotlib method to plot 2-dimensional data,
            see https://matplotlib.org/stable/api/axes_api.html for
            reference.
        :type method: str
        :param truth_solution: Real solution evaluated at 'pts',
            defaults to None.
        :type truth_solution: torch.Tensor, optional
        """

        grids = [p_.reshape(res, res) for p_ in pts.extract(v).T]

        pred_output = pred.reshape(res, res)
        if truth_solution:
            truth_output = truth_solution(pts).float().reshape(res, res).as_subclass(torch.Tensor)
            fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(16, 6))

            cb = getattr(ax[0], method)(*grids, pred_output,
                                        **kwargs)
            fig.colorbar(cb, ax=ax[0])
            ax[0].title.set_text('Neural Network prediction')
            cb = getattr(ax[1], method)(*grids, truth_output,
                                        **kwargs)
            fig.colorbar(cb, ax=ax[1])
            ax[1].title.set_text('True solution')
            cb = getattr(ax[2],
                         method)(*grids,
                                 (truth_output - pred_output),
                                 **kwargs)
            fig.colorbar(cb, ax=ax[2])
            ax[2].title.set_text('Residual')
        else:
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 6))
            cb = getattr(ax, method)(*grids, pred_output,
                                     **kwargs)
            fig.colorbar(cb, ax=ax)
            ax.title.set_text('Neural Network prediction')

    def plot(self,
             solver,
             components=None,
             fixed_variables={},
             method='contourf',
             res=256,
             filename=None,
             **kwargs):
        """
        Plot sample of SolverInterface output.

        :param SolverInterface solver: The ``SolverInterface`` object instance.
        :param list(str) components: The output variable to plot. If None, all
            the output variables of the problem are selected. Default value is None.
        :param dict fixed_variables: A dictionary with all the variables that
            should be kept fixed during the plot. The keys of the dictionary
            are the variables name whereas the values are the corresponding
            values of the variables. Defaults is `dict()`.
        :param str method: The matplotlib method to use for
            plotting the solution. Available methods are {'contourf', 'pcolor'}.
            Default is 'contourf'.
        :param int res: The resolution, aka the number of points used for
            plotting in each axis. Default is 256.
        :param str filename: The file name to save the plot. If None, the plot
            is shown using the setted matplotlib frontend. Default is None.
        """

        if components is None:
            components = solver.problem.output_variables
        
        if len(components) > 1:
            raise NotImplementedError('Multidimensional plots are not implemented, '
                                      'set components to an available components of the problem.')
        v = [
            var for var in solver.problem.input_variables
            if var not in fixed_variables.keys()
        ]
        pts = solver.problem.domain.sample(res, 'grid', variables=v)

        fixed_pts = torch.ones(pts.shape[0], len(fixed_variables))
        fixed_pts *= torch.tensor(list(fixed_variables.values()))
        fixed_pts = fixed_pts.as_subclass(LabelTensor)
        fixed_pts.labels = list(fixed_variables.keys())

        pts = pts.append(fixed_pts)
        pts = pts.to(device=solver.device)

        # computing soluting and sending to cpu
        predicted_output = solver.forward(pts).extract(components).as_subclass(torch.Tensor).cpu().detach()
        pts = pts.cpu()
        truth_solution = getattr(solver.problem, 'truth_solution', None)

        if len(v) == 1:
            self._1d_plot(pts, predicted_output, v, method, truth_solution,
                          **kwargs)
        elif len(v) == 2:
            self._2d_plot(pts, predicted_output, v, res, method, truth_solution,
                          **kwargs)

        plt.tight_layout()
        if filename:
            plt.savefig(filename)
            plt.close()
        else:
            plt.show()

    def plot_loss(self,
                  trainer,
                  metrics=None,
                  logy=False,
                  logx=False,
                  filename=None,
                  **kwargs):
        """
        Plot the loss function values during traininig.

        :param trainer: the PINA Trainer object instance.
        :type trainer: Trainer
        :param str | list(str) metric: The metrics to use in the y axis. If None, the mean loss
            is plotted.
        :param bool logy: If True, the y axis is in log scale. Default is
            True.
        :param bool logx: If True, the x axis is in log scale. Default is
            True.
        :param str filename: The file name to save the plot. If None, the plot
            is shown using the setted matplotlib frontend. Default is None.
        """

        # check that MetricTracker has been used
        list_ = [
            idx for idx, s in enumerate(trainer.callbacks)
            if isinstance(s, MetricTracker)
        ]
        if not bool(list_):
            raise FileNotFoundError(
                'MetricTracker should be used as a callback during training to'
                ' use this method.')

        # extract trainer metrics
        trainer_metrics = trainer.callbacks[list_[0]].metrics
        if metrics is None:
            metrics = ['mean_loss']
        elif not isinstance(metrics, list):
            raise ValueError('metrics must be class list.')

        # loop over metrics to plot
        for metric in metrics:
            if metric not in trainer_metrics:
                raise ValueError(
                    f'{metric} not a valid metric. Available metrics are {list(trainer_metrics.keys())}.'
                )
            loss = trainer_metrics[metric]
            epochs = range(len(loss))
            plt.plot(epochs, loss.cpu(), **kwargs)

        # plotting
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend()

        # log axis
        if logy:
            plt.yscale('log')
        if logx:
            plt.xscale('log')

        # saving in file
        if filename:
            plt.savefig(filename)
            plt.close()
