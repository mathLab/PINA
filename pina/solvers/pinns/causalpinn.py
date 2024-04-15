""" Module for PINN """

import torch


from torch.optim.lr_scheduler import ConstantLR

from pina.solvers import PINN
from pina.problem import TimeDependentProblem
from pina.utils import check_consistency


class CausalPINN(PINN):
    """
    Causal PINN solver class. This class implements Causal Physics Informed
    Neural Network solvers, using a user specified ``model`` to solve a specific
    ``problem``. It can be used for solving both forward and inverse problems.

    .. seealso::

        **Original reference**: Wang, Sifan, Shyam Sankaran, and Paris
        Perdikaris. "Respecting causality for training physics-informed
        neural networks." Computer Methods in Applied Mechanics
        and Engineering 421 (2024): 116813.
        <https://doi.org/10.1016/j.cma.2024.116813>`_.

    .. note::
        This class can only work for problems inheriting
        from at least
        :class:`~pina.problem.timedep_problem.TimeDependentProblem` class.
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
        eps=100,
    ):
        """
        :param AbstractProblem problem: The formulation of the problem.
        :param torch.nn.Module model: The neural network model to use.
        :param torch.nn.Module loss: The loss function used as minimizer,
            default :class:`torch.nn.MSELoss`.
        :param torch.nn.Module extra_features: The additional input
            features to use as augmented input.
        :param torch.optim.Optimizer optimizer: The neural network optimizer to
            use; default is :class:`torch.optim.Adam`.
        :param dict optimizer_kwargs: Optimizer constructor keyword args.
        :param torch.optim.LRScheduler scheduler: Learning
            rate scheduler.
        :param dict scheduler_kwargs: LR scheduler constructor keyword args.
        :param int | float eps: The exponential decay parameter. Note that this
            value is kept fixed during the training, but can be changed by means
            of a callback, e.g. for annealing. 
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

        # checking consistency
        check_consistency(eps, (int,float))
        self._eps = eps
        if not isinstance(self.problem, TimeDependentProblem):
            raise ValueError('Casual PINN works only for problems'
                             'inheritig from TimeDependentProblem.')

    @property
    def eps(self):
        """
        The exponential decay parameter
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

    def _loss_phys(self, samples, equation, condition_name):
        # split sequentially ordered time tensors into chunks
        chunks, labels = self._split_tensor_into_chunks(samples)
        # compute residuals - this correspond to ordered loss functions
        # values for each time step. We apply `flatten` such that after
        # concataning the residuals we obtain a tensor of shape #chunks
        time_loss = []
        for chunk in chunks:
            chunk.labels = labels
            loss_val = self.loss_phys(chunk, equation).as_subclass(torch.Tensor)
            time_loss.append(loss_val)
        # store results
        self.store_log(name=condition_name+'_loss',
                       loss_val=float(sum(time_loss)/len(time_loss)))
        # concatenate residuals
        time_loss = torch.stack(time_loss)
        # compute weights (without the gradient storing)
        with torch.no_grad():
            weights = self._compute_weights(time_loss)
        return (weights * time_loss).mean().as_subclass(torch.Tensor)
    
    def _sort_label_tensor(self, tensor):
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
        # labels input tensors
        labels = tensor.labels
        # labels input tensors
        tensor = self._sort_label_tensor(tensor)
        # extract time tensor
        time_tensor = tensor.extract(self.problem.temporal_domain.variables)
        # count unique tensors in time
        _, idx_split = time_tensor.unique(return_counts=True)
        # splitting
        chunks = torch.split(tensor, tuple(idx_split))
        return chunks, labels # return chunks
    
    def _compute_weights(self, loss):
        # compute comulative loss and multiply by epsilos
        cumulative_loss = self._eps * torch.cumsum(loss, dim=0)
        # return the exponential of the weghited negative cumulative sum
        return torch.exp(-cumulative_loss)
