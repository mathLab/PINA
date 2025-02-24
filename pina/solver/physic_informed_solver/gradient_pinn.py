"""Module for Gradient PINN."""

import torch

from .pinn import PINN
from pina.operator import grad
from pina.problem import SpatialProblem


class GradientPINN(PINN):
    r"""
    Gradient Physics Informed Neural Network (GradientPINN) solver class.
    This class implements Gradient Physics Informed Neural
    Network solver, using a user specified ``model`` to solve a specific
    ``problem``. It can be used for solving both forward and inverse problems.

    The Gradient Physics Informed Network aims to find
    the solution :math:`\mathbf{u}:\Omega\rightarrow\mathbb{R}^m`
    of the differential problem:

    .. math::

        \begin{cases}
        \mathcal{A}[\mathbf{u}](\mathbf{x})=0\quad,\mathbf{x}\in\Omega\\
        \mathcal{B}[\mathbf{u}](\mathbf{x})=0\quad,
        \mathbf{x}\in\partial\Omega
        \end{cases}

    minimizing the loss function

    .. math::
        \mathcal{L}_{\rm{problem}} =& \frac{1}{N}\sum_{i=1}^N
        \mathcal{L}(\mathcal{A}[\mathbf{u}](\mathbf{x}_i)) +
        \frac{1}{N}\sum_{i=1}^N
        \mathcal{L}(\mathcal{B}[\mathbf{u}](\mathbf{x}_i)) + \\
        &\frac{1}{N}\sum_{i=1}^N
        \nabla_{\mathbf{x}}\mathcal{L}(\mathcal{A}[\mathbf{u}](\mathbf{x}_i)) +
        \frac{1}{N}\sum_{i=1}^N
        \nabla_{\mathbf{x}}\mathcal{L}(\mathcal{B}[\mathbf{u}](\mathbf{x}_i))


    where :math:`\mathcal{L}` is a specific loss function,
    default Mean Square Error:

    .. math::
        \mathcal{L}(v) = \| v \|^2_2.

    .. seealso::

        **Original reference**: Yu, Jeremy, et al. "Gradient-enhanced
        physics-informed neural networks for forward and inverse
        PDE problems." Computer Methods in Applied Mechanics
        and Engineering 393 (2022): 114823.
        DOI: `10.1016 <https://doi.org/10.1016/j.cma.2022.114823>`_.

    .. note::
        This class can only work for problems inheriting
        from at least :class:`~pina.problem.spatial_problem.SpatialProblem`
        class.
    """

    def __init__(
        self,
        problem,
        model,
        optimizer=None,
        scheduler=None,
        weighting=None,
        loss=None,
    ):
        """
        :param torch.nn.Module model: The neural network model to use.
        :param AbstractProblem problem: The formulation of the problem. It must
            inherit from at least
            :class:`~pina.problem.spatial_problem.SpatialProblem` to compute
            the gradient of the loss.
        :param torch.optim.Optimizer optimizer: The neural network optimizer to
            use; default `None`.
        :param torch.optim.LRScheduler scheduler: Learning rate scheduler;
            default `None`.
        :param WeightingInterface weighting: The weighting schema to use;
            default `None`.
        :param torch.nn.Module loss: The loss function to be minimized;
            default `None`.
        """
        super().__init__(
            model=model,
            problem=problem,
            optimizer=optimizer,
            scheduler=scheduler,
            weighting=weighting,
            loss=loss,
        )

        if not isinstance(self.problem, SpatialProblem):
            raise ValueError(
                "Gradient PINN computes the gradient of the "
                "PINN loss with respect to the spatial "
                "coordinates, thus the PINA problem must be "
                "a SpatialProblem."
            )

    def loss_phys(self, samples, equation):
        """
        Computes the physics loss for the GPINN solver based on given
        samples and equation.

        :param LabelTensor samples: The samples to evaluate the physics loss.
        :param EquationInterface equation: The governing equation
            representing the physics.
        :return: The physics loss calculated based on given
            samples and equation.
        :rtype: LabelTensor
        """
        # classical PINN loss
        residual = self.compute_residual(samples=samples, equation=equation)
        loss_value = self.loss(
            torch.zeros_like(residual, requires_grad=True), residual
        )

        # gradient PINN loss
        loss_value = loss_value.reshape(-1, 1)
        loss_value.labels = ["__loss"]
        loss_grad = grad(loss_value, samples, d=self.problem.spatial_variables)
        g_loss_phys = self.loss(
            torch.zeros_like(loss_grad, requires_grad=True), loss_grad
        )
        return loss_value + g_loss_phys
