""" Module for RBAPINN. """

from copy import deepcopy
import torch
from torch.optim.lr_scheduler import ConstantLR
from .pinn import PINN
from ...utils import check_consistency


class RBAPINN(PINN):
    r"""
    Residual-based Attention PINN (RBAPINN) solver class.
    This class implements Residual-based Attention Physics Informed Neural
    Network solvers, using a user specified ``model`` to solve a specific
    ``problem``. It can be used for solving both forward and inverse problems.

    The Residual-based Attention Physics Informed Neural Network aims to find
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

        \mathcal{L}_{\rm{problem}} = \frac{1}{N} \sum_{i=1}^{N_\Omega} 
        \lambda_{\Omega}^{i} \mathcal{L} \left( \mathcal{A}
        [\mathbf{u}](\mathbf{x}) \right) + \frac{1}{N} 
        \sum_{i=1}^{N_{\partial\Omega}}
        \lambda_{\partial\Omega}^{i} \mathcal{L} 
        \left( \mathcal{B}[\mathbf{u}](\mathbf{x})
        \right),
    
    denoting the weights as
    :math:`\lambda_{\Omega}^1, \dots, \lambda_{\Omega}^{N_\Omega}` and
    :math:`\lambda_{\partial \Omega}^1, \dots, 
    \lambda_{\Omega}^{N_\partial \Omega}`
    for :math:`\Omega` and :math:`\partial \Omega`, respectively.

    Residual-based Attention Physics Informed Neural Network computes 
    the weights by updating them at every epoch as follows

    .. math::

        \lambda_i^{k+1} \leftarrow \gamma\lambda_i^{k} + 
        \eta\frac{\lvert r_i\rvert}{\max_j \lvert r_j\rvert},

    where :math:`r_i` denotes the residual at point :math:`i`, 
    math:`\gamma` denotes the decay rate, and :math:`\eta` is
    the learning rate for the weights' update.

    .. seealso::
        **Original reference**: Sokratis J. Anagnostopoulos, Juan D. Toscano,
        Nikolaos Stergiopulos, and George E. Karniadakis.
        "Residual-based attention and connection to information 
        bottleneck theory in PINNs".
        Computer Methods in Applied Mechanics and Engineering 421 (2024): 116805
        DOI: `10.1016/
        j.cma.2024.116805 <https://doi.org/10.1016/j.cma.2024.116805>`_.
    """

    def __init__(
        self,
        problem,
        model,
        extra_features=None,
        loss=torch.nn.MSELoss(),
        optimizer=torch.optim.Adam,
        optimizer_kwargs={"lr": 0.001},
        scheduler=ConstantLR,
        scheduler_kwargs={"factor": 1, "total_iters": 0},
        eta=0.001,
        gamma=0.999,
    ):
        """
        :param AbstractProblem problem: The formulation of the problem.
        :param torch.nn.Module model: The neural network model to use.
        :param torch.nn.Module extra_features: The additional input
            features to use as augmented input.
        :param torch.nn.Module loss: The loss function used as minimizer,
            default :class:`torch.nn.MSELoss`.
        :param torch.optim.Optimizer optimizer: The neural network optimizer to
            use; default is :class:`torch.optim.Adam`.
        :param dict optimizer_kwargs: Optimizer constructor keyword args.
        :param torch.optim.LRScheduler scheduler: Learning
            rate scheduler.
        :param dict scheduler_kwargs: LR scheduler constructor keyword args.
        :param float | int eta: The learning rate for the
            weights of the residual.
        :param float  gamma: The decay parameter in the update of the weights
            of the residual.
        """
        super().__init__(
            problem=problem,
            model=model,
            extra_features=extra_features,
            loss=loss,
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
            scheduler=scheduler,
            scheduler_kwargs=scheduler_kwargs,
        )

        # check consistency
        check_consistency(eta, (float, int))
        check_consistency(gamma, float)
        self.eta = eta
        self.gamma = gamma

        # initialize weights
        self.weights = {}
        for condition_name in problem.conditions:
            self.weights[condition_name] = 0

        # define vectorial loss
        self._vectorial_loss = deepcopy(loss)
        self._vectorial_loss.reduction = "none"

    def _vect_to_scalar(self, loss_value):
        """
        Elaboration of the pointwise loss.

        :param LabelTensor loss_value: the matrix of pointwise loss.

        :return: the scalar loss.
        :rtype LabelTensor
        """
        if self.loss.reduction == "mean":
            ret = torch.mean(loss_value)
        elif self.loss.reduction == "sum":
            ret = torch.sum(loss_value)
        else:
            raise RuntimeError(
                f"Invalid reduction, got {self.loss.reduction} "
                "but expected mean or sum."
            )
        return ret

    def loss_phys(self, samples, equation):
        """
        Computes the physics loss for the residual-based attention PINN
        solver based on given samples and equation.

        :param LabelTensor samples: The samples to evaluate the physics loss.
        :param EquationInterface equation: The governing equation
            representing the physics.
        :return: The physics loss calculated based on given
            samples and equation.
        :rtype: LabelTensor
        """
        residual = self.compute_residual(samples=samples, equation=equation)
        cond = self.current_condition_name

        r_norm = self.eta * torch.abs(residual) / torch.max(torch.abs(residual))
        self.weights[cond] = (self.gamma * self.weights[cond] + r_norm).detach()

        loss_value = self._vectorial_loss(
            torch.zeros_like(residual, requires_grad=True), residual
        )

        self.store_log(loss_value=float(self._vect_to_scalar(loss_value)))

        return self._vect_to_scalar(self.weights[cond] ** 2 * loss_value)
