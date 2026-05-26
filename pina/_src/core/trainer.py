"""Trainer utilities built on top of the PyTorch Lightning Trainer class."""

import sys
import warnings
import torch
import lightning
from pina._src.solver.mixin.physics_informed_mixin import _PhysicsInformedMixin
from pina._src.solver.base_solver import BaseSolver
from pina._src.data.data_module import DataModule
from pina._src.core.utils import (
    check_consistency,
    custom_warning_format,
    check_positive_integer,
)

# set the warning for compile options
warnings.formatwarning = custom_warning_format
warnings.filterwarnings("always", category=UserWarning)


class Trainer(lightning.pytorch.Trainer):
    """
    PINA-specific extension of :class:`lightning.pytorch.Trainer`.

    The trainer configures solver execution, dataset splitting, batching,
    logging, compilation support, device placement for unknown parameters, and
    gradient tracking requirements for physics-informed solvers.
    """

    # Available batching modes
    _AVAIL_BATCHING_MODES = {
        "common_batch_size",
        "proportional",
        "separate_conditions",
    }

    def __init__(
        self,
        solver,
        batch_size=None,
        train_size=1.0,
        test_size=0.0,
        val_size=0.0,
        compile=False,
        batching_mode="common_batch_size",
        automatic_batching=False,
        num_workers=0,
        pin_memory=False,
        shuffle=True,
        **kwargs,
    ):
        """
        Initialization of the :class:`Trainer` class.

        :param BaseSolver solver: The solver used to train, validate, and test
            the associated problem.
        :param int batch_size: The number of samples per batch. If ``None``, the
            entire dataset is processed as a single batch. Default is ``None``.
        :param float train_size: The fraction of samples assigned to the
            training split. Must belong to the interval ``[0, 1]``.
            Default is ``1.0``.
        :param float val_size: The fraction of samples assigned to the
            validation split. Must belong to the interval ``[0, 1]``.
            Default is ``0.0``.
        :param float test_size: The fraction of samples assigned to the test
            split. Must belong to the interval ``[0, 1]``. Default is ``0.0``.
        :param bool compile: Whether to compile the model before training.
            Compilation is disabled on Windows and with Python 3.14 or later.
            Default is ``False``.
        :param str batching_mode: The strategy used to aggregate batches across
            dataloaders. Available options are ``"common_batch_size"`` for
            uniform batch sizes across conditions, ``"proportional"`` for batch
            sizes proportional to dataset sizes, and ``"separate_conditions"``
            for iterating through each condition separately.
            Default is ``"common_batch_size"``.
        :param bool automatic_batching: Whether PyTorch automatic batching
            should be enabled. If ``True``, dataset elements are retrieved
            individually and collated into batches by the dataloader.
            If ``False``, entire subsets are retrieved directly from the
            condition object. Default is ``False``.
        :param int num_workers: The number of worker processes used by
            dataloaders. Default is ``0`` for sequential loading.
        :param bool pin_memory: Whether pinned memory should be enabled during
            data loading. Default is ``False``.
        :param bool shuffle: Whether condition samples should be shuffled before
            splitting. Default is ``True``.
        :param dict kwargs: Additional keyword arguments forwarded to the
            Lightning trainer.
        :raises ValueError: If ``solver`` is not a PINA solver.
        :raises ValueError: If ``train_size``, ``val_size``, or ``test_size`` is
            not a float in the interval ``[0, 1]``.
        :raises ValueError: If the sum of ``train_size``, ``val_size``, and
            ``test_size`` is not equal to 1.
        :raises ValueError: If ``compile``, ``automatic_batching``,
            ``pin_memory``, or ``shuffle`` is not a boolean.
        :raises AssertionError: If ``num_workers`` is a negative integer.
        :raises ValueError: If ``batch_size``, when provided, is not a positive
            integer.
        :raises ValueError: If ``batching_mode`` is not one of the available
             options.
        :raises UserWarning: If compilation is requested on an unsupported
            platform or Python version.
        :raises UserWarning: If the provided ``batching_mode`` is incompatible
            with the ``batch_size``.
        :raises RuntimeError: If any domain in the problem has not been
            discretised.
        """
        # Check consistency
        check_consistency(solver, BaseSolver)
        check_consistency(train_size, float)
        check_consistency(test_size, float)
        check_consistency(val_size, float)
        check_consistency(compile, bool)
        check_consistency(automatic_batching, bool)
        check_consistency(pin_memory, bool)
        check_consistency(shuffle, bool)
        check_positive_integer(num_workers, strict=False)
        if batch_size is not None:
            check_positive_integer(batch_size, strict=True)

        # Check that train_size, test_size and val_size sum to 1
        total = train_size + val_size + test_size
        if not torch.isclose(torch.tensor(total), torch.tensor(1.0)):
            raise ValueError(
                "`train_size`, `val_size`, and `test_size` must sum to 1."
            )

        # Check consistency
        if batching_mode not in self._AVAIL_BATCHING_MODES:
            raise ValueError(
                f"Invalid batching mode '{batching_mode}'. "
                f"Expected one of: {sorted(self._AVAIL_BATCHING_MODES)}."
            )

        # Set inference mode to false when usiing physics-informed mixin
        if isinstance(solver, _PhysicsInformedMixin):
            kwargs["inference_mode"] = False

        # Set log_every_n_steps to 0 if batch_size is None, otherwise default
        kwargs["log_every_n_steps"] = (
            0 if batch_size is None else kwargs.get("log_every_n_steps", 50)
        )

        # Set default value for enable_progress_bar to True if not provided
        kwargs.setdefault("enable_progress_bar", True)

        # Initialize the parent class with the provided keyword arguments
        super().__init__(**kwargs)

        # Disable compilation for Windows and Python 3.14+
        if sys.platform == "win32" or sys.version_info >= (3, 14) and compile:

            # Raise a warning if compilation is requested but not supported
            warnings.warn(
                "Model compilation is not supported on Windows or with Python "
                "3.14+. Compilation has been disabled.",
                UserWarning,
            )

            # Set compile to False if not supported
            compile = False

        # Raise warning if batch size and batching mode are incompatible
        if batch_size is None and batching_mode != "common_batch_size":
            warnings.warn(
                f"Batching mode '{batching_mode}' is ignored when the batch "
                "size is None. Setting batching_mode to 'common_batch_size'.",
                UserWarning,
            )

            # Set batching mode to common_batch_size if incompatible
            batching_mode = "common_batch_size"

        # Raise warning if batch size and batching mode are incompatible
        if (
            batch_size is not None
            and batching_mode == "proportional"
            and batch_size <= len(solver.problem.conditions)
        ):
            warnings.warn(
                "Batching mode 'proportional' requires the batch size to be "
                "larger than the number of conditions. Setting batching_mode "
                "to 'common_batch_size'.",
                UserWarning,
            )

            # Set batching mode to common_batch_size if incompatible
            batching_mode = "common_batch_size"

        # Initialize the class attributes
        self.solver = solver
        self.compile = compile
        self.batch_size = batch_size

        # Move the unknown parameters to the correct device
        self._move_to_device()

        # Check that all domains are discretised, otherwise raise an error
        if not self.solver.problem.are_all_domains_discretised:

            # Get the list of sampled domains from the problem
            sampled_domains = self.solver.problem.discretised_domains

            # Create a status message for each domain
            status = "\n".join(
                f"    - Domain '{name}': "
                f"{'sampled' if name in sampled_domains else 'not sampled'}"
                for name in self.solver.problem.domains
            )

            # Raise an error with the status of each domain
            raise RuntimeError(
                "Cannot create the Trainer because some domains have not been "
                f"sampled. Domain status:\n{status}"
            )

        # Create the data module
        self.data_module = DataModule(
            problem=self.solver.problem,
            train_size=train_size,
            test_size=test_size,
            val_size=val_size,
            batch_size=self.batch_size,
            batching_mode=batching_mode,
            automatic_batching=automatic_batching,
            num_workers=num_workers,
            pin_memory=pin_memory,
            shuffle=shuffle,
        )

        # Set logging kwargs
        self.logging_kwargs = {
            "sync_dist": bool(
                len(self._accelerator_connector._parallel_devices) > 1
            ),
            "on_step": bool(kwargs["log_every_n_steps"] > 0),
            "prog_bar": bool(kwargs["enable_progress_bar"]),
            "on_epoch": True,
        }

    def _move_to_device(self):
        """
        Move problem unknown parameters to the trainer device.

        If the associated problem defines ``unknown_parameters``, each parameter
        is moved to the first device configured by the Lightning accelerator
        connector.
        """
        # Get the device from the accelerator connector
        device = self._accelerator_connector._parallel_devices[0]

        # Get the problem instance from the solver
        problem = self.solver.problem

        # Move the unknown parameters to the correct device if they exist
        if hasattr(problem, "unknown_parameters"):
            for key in problem.unknown_parameters:
                problem.unknown_parameters[key] = torch.nn.Parameter(
                    problem.unknown_parameters[key].data.to(device)
                )

    def train(self, **kwargs):
        """
        Fit the solver using the trainer data module.

        :param dict kwargs: Additional keyword arguments forwarded to the
            Lightning trainer ``fit`` method.
        :return: Result returned by Lightning's ``fit`` method.
        :rtype: Any
        """
        return super().fit(self.solver, datamodule=self.data_module, **kwargs)

    def test(self, **kwargs):
        """
        Test the solver using the trainer data module.

        :param dict kwargs: Additional keyword arguments forwarded to the
            Lightning trainer ``test`` method.
        :return: Result returned by Lightning's ``test`` method.
        :rtype: Any
        """
        return super().test(self.solver, datamodule=self.data_module, **kwargs)

    @property
    def solver(self):
        """
        Return the solver attached to the trainer.

        :return: The solver used by the trainer.
        :rtype: BaseSolver
        """
        return self._solver

    @solver.setter
    def solver(self, solver):
        """
        Set the solver attached to the trainer.

        :param BaseSolver solver: The solver instance to attach.
        """
        self._solver = solver

    @property
    def compile(self):
        """
        Return whether model compilation is enabled.

        :return: ``True`` if compilation is enabled, otherwise ``False``.
        :rtype: bool
        """
        return self._compile

    @compile.setter
    def compile(self, value):
        """
        Set the value of compile.

        :param bool value: Whether compilation is required or not.
        """
        self._compile = value
