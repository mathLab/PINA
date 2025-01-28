""" Trainer module. """
import torch
import lightning
from .utils import check_consistency
from .data import PinaDataModule
from .solvers.solver import SolverInterface


class Trainer(lightning.pytorch.Trainer):

    def __init__(self,
                 solver,
                 batch_size=None,
                 train_size=.7,
                 test_size=.2,
                 val_size=.1,
                 predict_size=0.,
                 **kwargs):
        """
        PINA Trainer class for costumizing every aspect of training via flags.

        :param solver: A pina:class:`SolverInterface` solver for the
            differential problem.
        :type solver: SolverInterface
        :param batch_size: How many samples per batch to load.
            If ``batch_size=None`` all
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
        self.train_size = train_size
        self.test_size = test_size
        self.val_size = val_size
        self.predict_size = predict_size
        self.solver = solver
        self.batch_size = batch_size
        self._move_to_device()
        self.data_module = None
        self._create_loader()

    def _move_to_device(self):
        device = self._accelerator_connector._parallel_devices[0]
        # move parameters to device
        pb = self.solver.problem
        if hasattr(pb, "unknown_parameters"):
            for key in pb.unknown_parameters:
                pb.unknown_parameters[key] = torch.nn.Parameter(
                    pb.unknown_parameters[key].data.to(device))

    def _create_loader(self):
        """
        This method is used here because is resampling is needed
        during training, there is no need to define to touch the
        trainer dataloader, just call the method.
        """
        if not self.solver.problem.collector.full:
            error_message = '\n'.join([
                f"""{" " * 13} ---> Condition {key} {"sampled" if value else
                "not sampled"}""" for key, value in
                self._solver.problem.collector._is_conditions_ready.items()
            ])
            raise RuntimeError('Cannot create Trainer if not all conditions '
                               'are sampled. The Trainer got the following:\n'
                               f'{error_message}')
        automatic_batching = False
        self.data_module = PinaDataModule(collector=self.solver.problem.collector,
                                          train_size=self.train_size,
                                          test_size=self.test_size,
                                          val_size=self.val_size,
                                          predict_size=self.predict_size,
                                          batch_size=self.batch_size,
                                          automatic_batching=automatic_batching)

    def train(self, **kwargs):
        """
        Train the solver method.
        """
        return super().fit(self.solver, datamodule=self.data_module, **kwargs)

    def test(self, **kwargs):
        """
        Test the solver method.
        """
        return super().test(self.solver, datamodule=self.data_module, **kwargs)

    @property
    def solver(self):
        """
        Returning trainer solver.
        """
        return self._solver

    @solver.setter
    def solver(self, solver):
        self._solver = solver
