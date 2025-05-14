"""Module for the R3Refinement callback."""

import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss
from .refinement_interface import RefinementInterface
from ...label_tensor import LabelTensor
from ...utils import check_consistency
from ...loss import LossInterface


class R3Refinement(RefinementInterface):
    """
    PINA Implementation of an R3 Refinement Callback.
    """

    def __init__(
        self, sample_every, residual_loss=nn.L1Loss, condition_to_update=None
    ):
        """
        This callback implements the R3 (Retain-Resample-Release) routine for
        sampling new points based on adaptive search.
        The algorithm incrementally accumulates collocation points in regions
        of high PDE residuals, and releases those with low residuals.
        Points are sampled uniformly in all regions where sampling is needed.

        .. seealso::

            Original Reference: Daw, Arka, et al. *Mitigating Propagation
            Failures in Physics-informed Neural Networks
            using Retain-Resample-Release (R3) Sampling. (2023)*.
            DOI: `10.48550/arXiv.2207.02338
            <https://doi.org/10.48550/arXiv.2207.02338>`_

        :param int sample_every: Frequency for sampling.
        :raises ValueError: If `sample_every` is not an integer.

        Example:
            >>> r3_callback = R3Refinement(sample_every=5)
        """
        super().__init__(sample_every, condition_to_update)
        # check consistency loss
        check_consistency(residual_loss, (LossInterface, _Loss), subclass=True)
        self.loss_fn = residual_loss(reduction="none")

    def sample(self, current_points, condition_name, solver):
        # Compute residuals for the given condition (average over fields)
        condition = solver.problem.conditions[condition_name]
        target = solver.compute_residual(
            current_points.requires_grad_(True), condition.equation
        )
        residuals = self.loss_fn(target, torch.zeros_like(target)).mean(
            dim=tuple(range(1, target.ndim))
        )

        # Prepare new points
        labels = current_points.labels
        domain_name = solver.problem.conditions[condition_name].domain
        domain = solver.problem.domains[domain_name]
        num_old_points = self.initial_population_size[condition_name]
        mask = (residuals > residuals.mean()).flatten()

        if mask.any():  # Use high-residual points
            pts = current_points[mask]
            pts.labels = labels
            retain_pts = len(pts)
            samples = domain.sample(num_old_points - retain_pts, "random")
            return LabelTensor.cat([pts, samples])
        else:
            return domain.sample(num_old_points, "random")
