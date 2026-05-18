"""Module for the Physics-Informed Neural Network solver."""

import warnings
import torch
from pina._src.solver.single_model_simple_solver import (
    SingleModelSimpleSolver,
)


class PINN(SingleModelSimpleSolver):
    r"""
    Physics-Informed Neural Network (PINN) solver class.
    This class implements Physics-Informed Neural Network solver, using a user
    specified ``model`` to solve a specific ``problem``.
    It can be used to solve both forward and inverse problems.

    The Physics Informed Neural Network solver aims to find the solution
    :math:`\mathbf{u}:\Omega\rightarrow\mathbb{R}^m` of a differential problem:

    .. math::

        \begin{cases}
        \mathcal{A}[\mathbf{u}](\mathbf{x})=0\quad,\mathbf{x}\in\Omega\\
        \mathcal{B}[\mathbf{u}](\mathbf{x})=0\quad,
        \mathbf{x}\in\partial\Omega
        \end{cases}

    minimizing the loss function:

    .. math::
        \mathcal{L}_{\rm{problem}} = \frac{1}{N}\sum_{i=1}^N
        \mathcal{L}(\mathcal{A}[\mathbf{u}](\mathbf{x}_i)) +
        \frac{1}{N}\sum_{i=1}^N
        \mathcal{L}(\mathcal{B}[\mathbf{u}](\mathbf{x}_i)),

    where :math:`\mathcal{L}` is a specific loss function, typically the MSE:

    .. math::
        \mathcal{L}(v) = \| v \|^2_2.

    .. seealso::

        **Original reference**: Karniadakis, G. E., Kevrekidis, I. G., Lu, L.,
        Perdikaris, P., Wang, S., & Yang, L. (2021).
        *Physics-informed machine learning.*
        Nature Reviews Physics, 3, 422-440.
        DOI: `10.1038 <https://doi.org/10.1038/s42254-021-00314-5>`_.
    """

    def __init__(
        self,
        problem,
        model,
        optimizer=None,
        scheduler=None,
        weighting=None,
        loss=None,
        use_lt=True,
    ):
        """
        Initialization of the :class:`PINN` class.

        :param BaseProblem problem: The problem to be solved.
        :param torch.nn.Module model: The neural network model to be used.
        :param OptimizerInterface optimizer: The optimizer to be used.
            If ``None``, the :class:`torch.optim.Adam` optimizer is used.
            Default is ``None``.
        :param SchedulerInterface scheduler: Learning rate scheduler.
            If ``None``, the :class:`torch.optim.lr_scheduler.ConstantLR`
            scheduler is used. Default is ``None``.
        :param WeightingInterface weighting: The weighting schema to be used.
            If ``None``, no weighting schema is used. Default is ``None``.
        :param torch.nn.Module loss: The loss function to be minimized.
            If ``None``, the :class:`torch.nn.MSELoss` loss is used.
            Default is `None`.
        """
        SingleModelSimpleSolver.__init__(
            self,
            model=model,
            problem=problem,
            optimizer=optimizer,
            scheduler=scheduler,
            weighting=weighting,
            loss=loss,
            use_lt=use_lt,
        )

    def setup(self, stage):
        """
        Preserve the old PINN compile guard for problematic torch versions.

        :param str stage: The current stage of the training process.
        :return: The result of the parent setup method.
        :rtype: Any
        """
        if torch.__version__ >= "2.8":
            self.trainer.compile = False
            warnings.warn(
                "Compilation is disabled for torch >= 2.8. "
                "Forcing compilation may cause runtime errors or instability.",
                UserWarning,
            )
        return super().setup(stage)

    @torch.enable_grad()
    def validation_step(self, batch, **kwargs):
        """
        Run validation with gradients enabled for physics residual operators.

        :param batch: Validation batch.
        :type batch: list[tuple[str, dict]]
        :return: Validation loss.
        :rtype: torch.Tensor
        """
        return super().validation_step(batch, **kwargs)

    @torch.enable_grad()
    def test_step(self, batch, **kwargs):
        """
        Run test with gradients enabled for physics residual operators.

        :param batch: Test batch.
        :type batch: list[tuple[str, dict]]
        :return: Test loss.
        :rtype: torch.Tensor
        """
        return super().test_step(batch, **kwargs)
