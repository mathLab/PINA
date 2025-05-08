"""Module for the R3Refinement callback."""

import importlib.metadata
import torch
from lightning.pytorch.callbacks import Callback
from ..label_tensor import LabelTensor
from ..utils import check_consistency


class R3Refinement(Callback):
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

        super().__init__()

        # sample every
        check_consistency(sample_every, int)
        self._sample_every = sample_every
        self._const_pts = None
        self._domains = None

    def _compute_residual(self, trainer):
        """
        Computes the residuals for a PINN object.

        :return: the total loss, and pointwise loss.
        :rtype: tuple
        """

        # extract the solver and device from trainer
        solver = trainer.solver
        device = trainer._accelerator_connector._accelerator_flag
        precision = trainer.precision
        if precision == "64-true":
            precision = torch.float64
        elif precision == "32-true":
            precision = torch.float32
        else:
            raise RuntimeError(
                "Currently R3Refinement is only implemented "
                "for precision '32-true' and '64-true', set "
                "Trainer precision to match one of the "
                "available precisions."
            )

        # compute residual
        res_loss = {}
        tot_loss = []
        for condition in self._conditions:
            pts = trainer.datamodule.train_dataset.conditions_dict[condition][
                "input"
            ]
            equation = solver.problem.conditions[condition].equation
            # send points to correct device
            pts = pts.to(device=device, dtype=precision)
            pts = pts.requires_grad_(True)
            pts.retain_grad()
            # PINN loss: equation evaluated only for sampling locations
            target = equation.residual(pts, solver.forward(pts))
            res_loss[condition] = torch.abs(target).as_subclass(torch.Tensor)
            tot_loss.append(torch.abs(target))
        return torch.vstack(tot_loss), res_loss

    def _r3_routine(self, trainer):
        """
        R3 refinement main routine.

        :param Trainer trainer: PINA Trainer.
        """
        # compute residual (all device possible)
        tot_loss, res_loss = self._compute_residual(trainer)
        tot_loss = tot_loss.as_subclass(torch.Tensor)

        # !!!!!! From now everything is performed on CPU !!!!!!

        # average loss
        avg = (tot_loss.mean()).to("cpu")
        new_pts = {}

        dataset = trainer.datamodule.train_dataset
        problem = trainer.solver.problem
        for condition in self._conditions:
            pts = dataset.conditions_dict[condition]["input"]
            domain = problem.conditions[condition].domain
            if not isinstance(domain, str):
                domain = condition
            labels = pts.labels
            pts = pts.cpu().detach().as_subclass(torch.Tensor)
            residuals = res_loss[condition].cpu()
            mask = (residuals > avg).flatten()
            if any(mask):  # append residuals greater than average
                pts = (pts[mask]).as_subclass(LabelTensor)
                pts.labels = labels
                numb_pts = self._const_pts[condition] - len(pts)
            else:  # if no res greater than average, samples all uniformly
                numb_pts = self._const_pts[condition]
                pts = None
            problem.discretise_domain(numb_pts, "random", domains=[domain])
            sampled_points = problem.discretised_domains[domain]
            tmp = (
                sampled_points
                if pts is None
                else LabelTensor.cat([pts, sampled_points])
            )
            new_pts[condition] = {"input": tmp}
        dataset.update_data(new_pts)

    def on_train_start(self, trainer, _):
        """
        Callback function called at the start of training.

        This method extracts the locations for sampling from the problem
        conditions and calculates the total population.

        :param trainer: The trainer object managing the training process.
        :type trainer: pytorch_lightning.Trainer
        :param _: Placeholder argument (not used).

        :return: None
        :rtype: None
        """
        problem = trainer.solver.problem
        if hasattr(problem, "domains"):
            domains = problem.domains
            self._domains = domains
        else:
            self._domains = {}
            for name, data in problem.conditions.items():
                if hasattr(data, "domain"):
                    self._domains[name] = data.domain
        self._conditions = []
        for name, data in problem.conditions.items():
            if hasattr(data, "domain"):
                self._conditions.append(name)

        # extract total population
        const_pts = {}  # for each location, store the  pts to keep constant
        for condition in self._conditions:
            pts = trainer.datamodule.train_dataset.conditions_dict[condition][
                "input"
            ]
            const_pts[condition] = len(pts)
        self._const_pts = const_pts

    def on_train_epoch_end(self, trainer, __):
        """
        Callback function called at the end of each training epoch.

        This method triggers the R3 routine for refinement if the current
        epoch is a multiple of `_sample_every`.

        :param trainer: The trainer object managing the training process.
        :type trainer: pytorch_lightning.Trainer
        :param __: Placeholder argument (not used).

        :return: None
        :rtype: None
        """
        if trainer.current_epoch % self._sample_every == 0:
            self._r3_routine(trainer)
