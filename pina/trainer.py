"""Trainer module."""

import sys
import torch
import lightning
from .utils import check_consistency
from .data import PinaDataModule
from .solver import SolverInterface, PINNInterface


class Trainer(lightning.pytorch.Trainer):

    def __init__(
        self,
        solver,
        batch_size=None,
        train_size=1.0,
        test_size=0.0,
        val_size=0.0,
        compile=None,
        automatic_batching=None,
        num_workers=None,
        pin_memory=None,
        shuffle=None,
        **kwargs,
    ):
        """
        PINA Trainer class for costumizing every aspect of training via flags.

        :param solver: A pina:class:`SolverInterface` solver for the
            differential problem.
        :type solver: SolverInterface
        :param batch_size: How many samples per batch to load.
            If ``batch_size=None`` all
            samples are loaded and data are not batched, defaults to None.
        :type batch_size: int | None
        :param train_size: Percentage of elements in the train dataset.
        :type train_size: float
        :param test_size: Percentage of elements in the test dataset.
        :type test_size: float
        :param val_size: Percentage of elements in the val dataset.
        :type val_size: float
        :param compile: if True model is compiled before training,
            default False. For Windows users compilation is always disabled.
        :type compile: bool
        :param automatic_batching: if True automatic PyTorch batching is
            performed. Please avoid using automatic batching when batch_size is
            large, default False.
        :type automatic_batching: bool
        :param num_workers: Number of worker threads for data loading.
            Default 0 (serial loading).
        :type num_workers: int
        :param pin_memory: Whether to use pinned memory for faster data
            transfer to GPU. Default False.
        :type pin_memory: bool
        :param shuffle: Whether to shuffle the data for training. Default True.
        :type pin_memory: bool

        :Keyword Arguments:
            The additional keyword arguments specify the training setup
            and can be choosen from the `pytorch-lightning
            Trainer API <https://lightning.ai/docs/pytorch/stable/common/trainer.html#trainer-class-api>`_
        """
        # check consistency for init types
        check_consistency(solver, SolverInterface)
        check_consistency(train_size, float)
        check_consistency(test_size, float)
        check_consistency(val_size, float)
        if automatic_batching is not None:
            check_consistency(automatic_batching, bool)
        if compile is not None:
            check_consistency(compile, bool)
        if pin_memory is not None:
            check_consistency(pin_memory, bool)
        else:
            pin_memory = False
        if num_workers is not None:
            check_consistency(pin_memory, int)
        else:
            num_workers = 0
        if shuffle is not None:
            check_consistency(shuffle, bool)
        else:
            shuffle = True
        if batch_size is not None:
            check_consistency(batch_size, int)

        # inference mode set to false when validating/testing PINNs otherwise
        # gradient is not tracked and optimization_cycle fails
        if isinstance(solver, PINNInterface):
            kwargs["inference_mode"] = False

        # Logging depends on the batch size, when batch_size is None then
        # log_every_n_steps should be zero
        if batch_size is None:
            kwargs["log_every_n_steps"] = 0
        else:
            kwargs.setdefault("log_every_n_steps", 50)  # default for lightning

        # Setting default kwargs, overriding lightning defaults
        kwargs.setdefault("enable_progress_bar", True)
        kwargs.setdefault("logger", None)

        super().__init__(**kwargs)

        # checking compilation and automatic batching
        if compile is None or sys.platform == "win32":
            compile = False

        self.automatic_batching = (
            automatic_batching if automatic_batching is not None else False
        )
        # set attributes
        self.compile = compile
        self.solver = solver
        self.batch_size = batch_size
        self._move_to_device()
        self.data_module = None
        self._create_datamodule(
            train_size,
            test_size,
            val_size,
            batch_size,
            automatic_batching,
            pin_memory,
            num_workers,
            shuffle,
        )

        # logging
        self.logging_kwargs = {
            "logger": bool(
                kwargs["logger"] is not None or kwargs["logger"] is True
            ),
            "sync_dist": bool(
                len(self._accelerator_connector._parallel_devices) > 1
            ),
            "on_step": bool(kwargs["log_every_n_steps"] > 0),
            "prog_bar": bool(kwargs["enable_progress_bar"]),
            "on_epoch": True,
        }

    def _move_to_device(self):
        device = self._accelerator_connector._parallel_devices[0]
        # move parameters to device
        pb = self.solver.problem
        if hasattr(pb, "unknown_parameters"):
            for key in pb.unknown_parameters:
                pb.unknown_parameters[key] = torch.nn.Parameter(
                    pb.unknown_parameters[key].data.to(device)
                )

    def _create_datamodule(
        self,
        train_size,
        test_size,
        val_size,
        batch_size,
        automatic_batching,
        pin_memory,
        num_workers,
        shuffle,
    ):
        """
        This method is used here because is resampling is needed
        during training, there is no need to define to touch the
        trainer dataloader, just call the method.
        """
        if not self.solver.problem.are_all_domains_discretised:
            error_message = "\n".join(
                [
                    f"""{" " * 13} ---> Domain {key} {
                    "sampled" if key in self.solver.problem.discretised_domains 
                    else
                    "not sampled"}"""
                    for key in self.solver.problem.domains.keys()
                ]
            )
            raise RuntimeError(
                "Cannot create Trainer if not all conditions "
                "are sampled. The Trainer got the following:\n"
                f"{error_message}"
            )
        self.data_module = PinaDataModule(
            self.solver.problem,
            train_size=train_size,
            test_size=test_size,
            val_size=val_size,
            batch_size=batch_size,
            automatic_batching=automatic_batching,
            num_workers=num_workers,
            pin_memory=pin_memory,
            shuffle=shuffle,
        )

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
