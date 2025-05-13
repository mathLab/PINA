"""Module for the R3Refinement callback."""

import torch
from .refinement_interface import RefinementInterface
from ...label_tensor import LabelTensor
from ...utils import check_consistency


class R3Refinement(RefinementInterface):
    """
    PINA Implementation of an R3 Refinement Callback.
    """

    def __init__(self, sample_every):
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

        super().__init__(sample_every=sample_every)
        self.const_pts = None

    def sample(self, condition_name, condition):
        avg_res, res = self.per_point_residual([condition_name])
        pts = self.dataset.conditions_dict[condition_name]["input"]
        domain = condition.domain
        labels = pts.labels
        pts = pts.cpu().detach().as_subclass(torch.Tensor)
        residuals = res[condition_name]
        mask = (residuals > avg_res).flatten()
        if any(mask):  # append residuals greater than average
            pts = (pts[mask]).as_subclass(LabelTensor)
            pts.labels = labels
            numb_pts = self.const_pts[condition_name] - len(pts)
        else:
            numb_pts = self.const_pts[condition_name]
            pts = None
        self.problem.discretise_domain(numb_pts, "random", domains=[domain])
        sampled_points = self.problem.discretised_domains[domain]
        tmp = (
            sampled_points
            if pts is None
            else LabelTensor.cat([pts, sampled_points])
        )
        return tmp

    def on_train_start(self, trainer, _):
        """
        Callback function called at the start of training.

        This method extracts the locations for sampling from the problem
        conditions and calculates the total population.

        :param trainer: The trainer object managing the training process.
        :type trainer: pytorch_lightning.Trainer
        :param _: Placeholder argument (not used).
        """
        super().on_train_start(trainer, _)
        self.const_pts = {}
        for condition in self.conditions:
            pts = self.dataset.conditions_dict[condition]["input"]
            self.const_pts[condition] = len(pts)
