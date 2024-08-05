""" Trainer module. """

import torch
import pytorch_lightning
from .utils import check_consistency
from .data.dataset import SamplePointDataset, SamplePointLoader, DataPointDataset
from .solvers.solver import SolverInterface


class Trainer(pytorch_lightning.Trainer):

    def __init__(self, solver, batch_size=None, **kwargs):
        """
        PINA Trainer class for costumizing every aspect of training via flags.

        :param solver: A pina:class:`SolverInterface` solver for the differential problem.
        :type solver: SolverInterface
        :param batch_size: How many samples per batch to load. If ``batch_size=None`` all
            samples are loaded and data are not batched, defaults to None.
        :type batch_size: int | None

        :Keyword Arguments:
            The additional keyword arguments specify the training setup
            and can be choosen from the `pytorch-lightning
            Trainer API <https://lightning.ai/docs/pytorch/stable/common/trainer.html#trainer-class-api>`_
        """

        super().__init__(**kwargs)

        # check inheritance consistency for solver and batch size
        check_consistency(solver, SolverInterface)
        if batch_size is not None:
            check_consistency(batch_size, int)

        self._model = solver
        self.batch_size = batch_size

        # create dataloader
        if solver.problem.have_sampled_points is False:
            raise RuntimeError(
                f"Input points in {solver.problem.not_sampled_points} "
                "training are None. Please "
                "sample points in your problem by calling "
                "discretise_domain function before train "
                "in the provided locations."
            )

        self._create_or_update_loader()

    def _create_or_update_loader(self):
        """
        This method is used here because is resampling is needed
        during training, there is no need to define to touch the
        trainer dataloader, just call the method.
        """
        devices = self._accelerator_connector._parallel_devices

        if len(devices) > 1:
            raise RuntimeError("Parallel training is not supported yet.")

        device = devices[0]
        dataset_phys = SamplePointDataset(self._model.problem, device)
        dataset_data = DataPointDataset(self._model.problem, device)
        self._loader = SamplePointLoader(
            dataset_phys, dataset_data, batch_size=self.batch_size, shuffle=True
        )
        pb = self._model.problem
        if hasattr(pb, "unknown_parameters"):
            for key in pb.unknown_parameters:
                pb.unknown_parameters[key] = torch.nn.Parameter(
                    pb.unknown_parameters[key].data.to(device)
                )

    def train(self, **kwargs):
        """
        Train the solver method.
        """
        return super().fit(
            self._model, train_dataloaders=self._loader, **kwargs
        )

    @property
    def solver(self):
        """
        Returning trainer solver.
        """
        return self._model
