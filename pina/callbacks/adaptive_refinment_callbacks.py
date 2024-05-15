"""PINA Callbacks Implementations"""

import torch
from pytorch_lightning.callbacks import Callback
from ..label_tensor import LabelTensor
from ..utils import check_consistency


class R3Refinement(Callback):

    def __init__(self, sample_every):
        """
        PINA Implementation of an R3 Refinement Callback.

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
        super().__init__()

        # sample every
        check_consistency(sample_every, int)
        self._sample_every = sample_every

    def _compute_residual(self, trainer):
        """
        Computes the residuals for a PINN object.

        :return: the total loss, and pointwise loss.
        :rtype: tuple
        """

        # extract the solver and device from trainer
        solver = trainer._model
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
        for location in self._sampling_locations:
            condition = solver.problem.conditions[location]
            pts = solver.problem.input_pts[location]
            # send points to correct device
            pts = pts.to(device=device, dtype=precision)
            pts = pts.requires_grad_(True)
            pts.retain_grad()
            # PINN loss: equation evaluated only for sampling locations
            target = condition.equation.residual(pts, solver.forward(pts))
            res_loss[location] = torch.abs(target).as_subclass(torch.Tensor)
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

        # points to keep
        old_pts = {}
        left_pts = {}
        for location in self._sampling_locations:
            tot_points = 0
            pts = trainer._model.problem.input_pts[location]
            Nloc = len(pts)
            labels = pts.labels
            pts = pts.cpu().detach().as_subclass(torch.Tensor)
            residuals = res_loss[location].cpu()
            mask = (residuals > avg).flatten()
            if any(
                mask
            ):  # if there are residuals greater than averge we append them
                # Fix the issue, masking remove labels
                pts = (pts[mask]).as_subclass(LabelTensor)
                pts.labels = labels
                old_pts[location] = pts
                tot_points += len(pts)
            left_pts[location] = Nloc - tot_points
        
            assert left_pts[location] + tot_points == Nloc


        # sample new points
        for  loc in self._sampling_locations:
            numb_pts = left_pts[loc]
            trainer._model.problem.discretise_domain(
                numb_pts, "random", locations=[loc]
            )
        # adding previous population points
        trainer._model.problem.add_points(old_pts)

        # update dataloader
        trainer._create_or_update_loader()

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
        # extract locations for sampling
        problem = trainer._model.problem
        locations = []
        for condition_name in problem.conditions:
            condition = problem.conditions[condition_name]
            if hasattr(condition, "location"):
                locations.append(condition_name)
        self._sampling_locations = locations

        # extract total population
        total_population = 0
        for location in self._sampling_locations:
            pts = trainer._model.problem.input_pts[location]
            total_population += len(pts)
        self._tot_pop_numb = total_population

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
