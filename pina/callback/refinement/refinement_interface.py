"""
RefinementInterface class for handling the refinement of points in a neural
network training process.
"""

import torch
from abc import ABCMeta
from lightning.pytorch import Callback
from torch_geometric.data.feature_store import abstractmethod
from torch_geometric.nn.conv import point_transformer_conv
from ...condition.domain_equation_condition import DomainEquationCondition


class RefinementInterface(Callback, metaclass=ABCMeta):
    """
    Interface class of Refinement
    """

    def __init__(self, sample_every):
        """
        Initializes the RefinementInterface.

        :param int sample_every: The number of epochs between each refinement.
        """
        self.sample_every = sample_every
        self.conditions = None
        self.dataset = None
        self.solver = None

    def on_train_start(self, trainer, _):
        """
        Called when the training begins. It initializes the conditions and
        dataset.

        :param lightning.pytorch.Trainer trainer: The trainer object.
        :param _: Unused argument.
        """
        self.problem = trainer.solver.problem
        self.solver = trainer.solver
        self.conditions = {}
        for name, cond in self.problem.conditions.items():
            if isinstance(cond, DomainEquationCondition):
                self.conditions[name] = cond
        self.dataset = trainer.datamodule.train_dataset

    @property
    def points(self):
        """
        Returns the points of the dataset.
        """
        return self.dataset.conditions_dict

    def on_train_epoch_end(self, trainer, _):
        """
        Performs the refinement at the end of each training epoch (if needed).

        :param lightning.pytorch.Trainer trainer: The trainer object.
        :param _: Unused argument.
        """
        if trainer.current_epoch % self.sample_every == 0:
            self.update()

    def update(self):
        """
        Performs the refinement of the points.
        """
        new_points = {}
        for name, condition in self.conditions.items():
            new_points[name] = {"input": self.sample(name, condition)}
        self.dataset.update_data(new_points)

    def per_point_residual(self, conditions_name=None):
        """
        Computes the residuals for a PINN object.

        :return: the total loss, and pointwise loss.
        :rtype: tuple
        """
        # compute residual
        res_loss = {}
        tot_loss = []
        points = self.points
        if conditions_name is None:
            conditions_name = list(self.conditions.keys())
        for name in conditions_name:
            cond = self.conditions[name]
            cond_points = points[name]["input"]
            target = self._compute_residual(cond_points, cond.equation)
            res_loss[name] = torch.abs(target).as_subclass(torch.Tensor)
            tot_loss.append(torch.abs(target))
        return torch.vstack(tot_loss).tensor.mean(), res_loss

    def _compute_residual(self, pts, equation):
        pts.requires_grad_(True)
        pts.retain_grad()
        return equation.residual(pts, self.solver.forward(pts))

    @abstractmethod
    def sample(self, condition):
        """
        Samples new points based on the condition.
        """
        pass
