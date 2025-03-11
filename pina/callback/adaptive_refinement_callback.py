"""PINA Callbacks Implementations"""

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
        of high PDE residuals, and releases those
        with low residuals. Points are sampled uniformly in all regions
        where sampling is needed.

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
        raise NotImplementedError(
            "R3Refinement callbacks is being refactoring in the "
            f"pina {importlib.metadata.metadata('pina-mathlab')['Verison']} "
            "version. Please use version 0.1 if R3Refinement is needed."
        )

    #     super().__init__()

    #     # sample every
    #     check_consistency(sample_every, int)
    #     self._sample_every = sample_every
    #     self._const_pts = None

    # def _compute_residual(self, trainer):
    #     """
    #     Computes the residuals for a PINN object.

    #     :return: the total loss, and pointwise loss.
    #     :rtype: tuple
    #     """

    #     # extract the solver and device from trainer
    #     solver = trainer.solver
    #     device = trainer._accelerator_connector._accelerator_flag
    #     precision = trainer.precision
    #     if precision == "64-true":
    #         precision = torch.float64
    #     elif precision == "32-true":
    #         precision = torch.float32
    #     else:
    #         raise RuntimeError(
    #             "Currently R3Refinement is only implemented "
    #             "for precision '32-true' and '64-true', set "
    #             "Trainer precision to match one of the "
    #             "available precisions."
    #         )

    #     # compute residual
    #     res_loss = {}
    #     tot_loss = []
    #     for location in self._sampling_locations:
    #         condition = solver.problem.conditions[location]
    #         pts = solver.problem.input_pts[location]
    #         # send points to correct device
    #         pts = pts.to(device=device, dtype=precision)
    #         pts = pts.requires_grad_(True)
    #         pts.retain_grad()
    #         # PINN loss: equation evaluated only for sampling locations
    #         target = condition.equation.residual(pts, solver.forward(pts))
    #         res_loss[location] = torch.abs(target).as_subclass(torch.Tensor)
    #         tot_loss.append(torch.abs(target))

    #     print(tot_loss)

    #     return torch.vstack(tot_loss), res_loss

    # def _r3_routine(self, trainer):
    #     """
    #     R3 refinement main routine.

    #     :param Trainer trainer: PINA Trainer.
    #     """
    #     # compute residual (all device possible)
    #     tot_loss, res_loss = self._compute_residual(trainer)
    #     tot_loss = tot_loss.as_subclass(torch.Tensor)

    #     # !!!!!! From now everything is performed on CPU !!!!!!

    #     # average loss
    #     avg = (tot_loss.mean()).to("cpu")
    #     old_pts = {}  # points to be retained
    #     for location in self._sampling_locations:
    #         pts = trainer._model.problem.input_pts[location]
    #         labels = pts.labels
    #         pts = pts.cpu().detach().as_subclass(torch.Tensor)
    #         residuals = res_loss[location].cpu()
    #         mask = (residuals > avg).flatten()
    #         if any(mask):  # append residuals greater than average
    #             pts = (pts[mask]).as_subclass(LabelTensor)
    #             pts.labels = labels
    #             old_pts[location] = pts
    #             numb_pts = self._const_pts[location] - len(old_pts[location])
    #             # sample new points
    #             trainer._model.problem.discretise_domain(
    #                 numb_pts, "random", locations=[location]
    #             )

    #         else:  # if no res greater than average, samples all uniformly
    #             numb_pts = self._const_pts[location]
    #             # sample new points
    #             trainer._model.problem.discretise_domain(
    #                 numb_pts, "random", locations=[location]
    #             )
    #     # adding previous population points
    #     trainer._model.problem.add_points(old_pts)

    #     # update dataloader
    #     trainer._create_or_update_loader()

    # def on_train_start(self, trainer, _):
    #     """
    #     Callback function called at the start of training.

    #     This method extracts the locations for sampling from the problem
    #     conditions and calculates the total population.

    #     :param trainer: The trainer object managing the training process.
    #     :type trainer: pytorch_lightning.Trainer
    #     :param _: Placeholder argument (not used).

    #     :return: None
    #     :rtype: None
    #     """
    #     # extract locations for sampling
    #     problem = trainer.solver.problem
    #     locations = []
    #     for condition_name in problem.conditions:
    #         condition = problem.conditions[condition_name]
    #         if hasattr(condition, "location"):
    #             locations.append(condition_name)
    #     self._sampling_locations = locations

    #     # extract total population
    #     const_pts = {}  # for each location, store the  pts to keep constant
    #     for location in self._sampling_locations:
    #         pts = trainer._model.problem.input_pts[location]
    #         const_pts[location] = len(pts)
    #     self._const_pts = const_pts

    # def on_train_epoch_end(self, trainer, __):
    #     """
    #     Callback function called at the end of each training epoch.

    #     This method triggers the R3 routine for refinement if the current
    #     epoch is a multiple of `_sample_every`.

    #     :param trainer: The trainer object managing the training process.
    #     :type trainer: pytorch_lightning.Trainer
    #     :param __: Placeholder argument (not used).

    #     :return: None
    #     :rtype: None
    #     """
    #     if trainer.current_epoch % self._sample_every == 0:
    #         self._r3_routine(trainer)
