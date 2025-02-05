""" Module for Causal PINN. """

import torch

from .pinn import PINN
from pina.utils import check_consistency
from pina.problem import TimeDependentProblem


class CausalPINN(PINN):
    r"""
    Causal Physics Informed Neural Network (CausalPINN) solver class.
    This class implements Causal Physics Informed Neural
    Network solvers, using a user specified ``model`` to solve a specific
    ``problem``. It can be used for solving both forward and inverse problems.

    The Causal Physics Informed Network aims to find
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
        \mathcal{L}_{\rm{problem}} = \frac{1}{N_t}\sum_{i=1}^{N_t}
        \omega_{i}\mathcal{L}_r(t_i),

    where:

    .. math::
        \mathcal{L}_r(t) = \frac{1}{N}\sum_{i=1}^N
        \mathcal{L}(\mathcal{A}[\mathbf{u}](\mathbf{x}_i, t)) +
        \frac{1}{N}\sum_{i=1}^N
        \mathcal{L}(\mathcal{B}[\mathbf{u}](\mathbf{x}_i, t))

    and,

    .. math::
        \omega_i = \exp\left(\epsilon \sum_{k=1}^{i-1}\mathcal{L}_r(t_k)\right).

    :math:`\epsilon` is an hyperparameter, default set to :math:`100`, while
    :math:`\mathcal{L}` is a specific loss function,
    default Mean Square Error:

    .. math::
        \mathcal{L}(v) = \| v \|^2_2.


    .. seealso::

        **Original reference**: Wang, Sifan, Shyam Sankaran, and Paris
        Perdikaris. "Respecting causality for training physics-informed
        neural networks." Computer Methods in Applied Mechanics
        and Engineering 421 (2024): 116813.
        DOI `10.1016 <https://doi.org/10.1016/j.cma.2024.116813>`_.

    .. note::
        This class can only work for problems inheriting
        from at least
        :class:`~pina.problem.timedep_problem.TimeDependentProblem` class.
    """

    def __init__(self,
                 problem,
                 model,
                 optimizer=None,
                 scheduler=None,
                 weighting=None,
                 loss=None,
                 eps=100):
        """
        :param AbstractProblem problem: The formulation of the problem.
        :param torch.nn.Module model: The neural network model to use.
        :param torch.nn.Module loss: The loss function used as minimizer,
            default :class:`torch.nn.MSELoss`.
        :param torch.optim.Optimizer optimizer: The neural network optimizer to
            use; default is :class:`torch.optim.Adam`.
        :param torch.optim.LRScheduler scheduler: Learning rate scheduler.
        :param int | float eps: The exponential decay parameter. Note that this
            value is kept fixed during training, but it can be changed by means
            of a callback, e.g. for annealing.
        """
        super().__init__(
            problem=problem,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            weighting=weighting,
            loss=loss
        )

        # checking consistency
        check_consistency(eps, (int, float))
        self._eps = eps
        if not isinstance(self.problem, TimeDependentProblem):
            raise ValueError(
                "Casual PINN works only for problems"
                "inheriting from TimeDependentProblem."
            )

    def loss_phys(self, samples, equation):
        """
        Computes the physics loss for the Causal PINN solver based on given
        samples and equation.

        :param LabelTensor samples: The samples to evaluate the physics loss.
        :param EquationInterface equation: The governing equation
            representing the physics.
        :return: The physics loss calculated based on given
            samples and equation.
        :rtype: LabelTensor
        """
        # split sequentially ordered time tensors into chunks
        chunks, labels = self._split_tensor_into_chunks(samples)
        # compute residuals - this correspond to ordered loss functions
        # values for each time step. Apply `flatten` to ensure obtaining
        # a tensor of shape #chunks after concatenating the residuals
        time_loss = []
        for chunk in chunks:
            chunk.labels = labels
            # classical PINN loss
            residual = self.compute_residual(samples=chunk, equation=equation)
            loss_val = self.loss(
                torch.zeros_like(residual, requires_grad=True), residual
            )
            time_loss.append(loss_val)

        # concatenate residuals
        time_loss = torch.stack(time_loss)
        # compute weights without storing the gradient
        with torch.no_grad():
            weights = self._compute_weights(time_loss)
        return (weights * time_loss).mean()

    @property
    def eps(self):
        """
        The exponential decay parameter.
        """
        return self._eps

    @eps.setter
    def eps(self, value):
        """
        Setter method for the eps parameter.

        :param float value: The exponential decay parameter.
        """
        check_consistency(value, float)
        self._eps = value

    def _sort_label_tensor(self, tensor):
        """
        Sorts the label tensor based on time variables.

        :param LabelTensor tensor: The label tensor to be sorted.
        :return: The sorted label tensor based on time variables.
        :rtype: LabelTensor
        """
        # labels input tensors
        labels = tensor.labels
        # extract time tensor
        time_tensor = tensor.extract(self.problem.temporal_domain.variables)
        # sort the time tensors (this is very bad for GPU)
        _, idx = torch.sort(time_tensor.tensor.flatten())
        tensor = tensor[idx]
        tensor.labels = labels
        return tensor

    def _split_tensor_into_chunks(self, tensor):
        """
        Splits the label tensor into chunks based on time.

        :param LabelTensor tensor: The label tensor to be split.
        :return: Tuple containing the chunks and the original labels.
        :rtype: Tuple[List[LabelTensor], List]
        """
        # extract labels
        labels = tensor.labels
        # sort input tensor based on time
        tensor = self._sort_label_tensor(tensor)
        # extract time tensor
        time_tensor = tensor.extract(self.problem.temporal_domain.variables)
        # count unique tensors in time
        _, idx_split = time_tensor.unique(return_counts=True)
        # split the tensor based on time
        chunks = torch.split(tensor, tuple(idx_split))
        return chunks, labels

    def _compute_weights(self, loss):
        """
        Computes the weights for the physics loss based on the cumulative loss.

        :param LabelTensor loss: The physics loss values.
        :return: The computed weights for the physics loss.
        :rtype: LabelTensor
        """
        # compute comulative loss and multiply by epsilon
        cumulative_loss = self._eps * torch.cumsum(loss, dim=0)
        # return the exponential of the negative weighted cumulative sum
        return torch.exp(-cumulative_loss)
