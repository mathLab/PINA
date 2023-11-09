""" Module for PINN """
import torch
import inspect
try:
    from torch.optim.lr_scheduler import LRScheduler  # torch >= 2.0
except ImportError:
    from torch.optim.lr_scheduler import _LRScheduler as LRScheduler  # torch < 2.0

from torch.optim.lr_scheduler import ConstantLR

from .solver import SolverInterface
from ..label_tensor import LabelTensor
from ..utils import check_consistency
from ..loss import LossInterface
from ..problem import InverseProblem
from torch.nn.modules.loss import _Loss
from inspect import signature

torch.pi = torch.acos(torch.zeros(1)).item() * 2  # which is 3.1415927410125732


class PINN(SolverInterface):
    """
    PINN solver class. This class implements Physics Informed Neural
    Network solvers, using a user specified ``model`` to solve a specific
    ``problem``. It can be used for solving both forward and inverse problems.

    .. seealso::

        **Original reference**: Karniadakis, G. E., Kevrekidis, I. G., Lu, L.,
        Perdikaris, P., Wang, S., & Yang, L. (2021).
        Physics-informed machine learning. Nature Reviews Physics, 3(6), 422-440.
        <https://doi.org/10.1038/s42254-021-00314-5>`_.
    """

    def __init__(
        self,
        problem,
        model,
        extra_features=None,
        loss=torch.nn.MSELoss(),
        optimizer=torch.optim.Adam,
        optimizer_kwargs={'lr': 0.001},
        scheduler=ConstantLR,
        scheduler_kwargs={
            "factor": 1,
            "total_iters": 0
        },
    ):
        '''
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
        '''
        super().__init__(models=[model],
                         problem=problem,
                         optimizers=[optimizer],
                         optimizers_kwargs=[optimizer_kwargs],
                         extra_features=extra_features)

        # check consistency
        check_consistency(scheduler, LRScheduler, subclass=True)
        check_consistency(scheduler_kwargs, dict)
        check_consistency(loss, (LossInterface, _Loss), subclass=False)

        # assign variables
        self._scheduler = scheduler(self.optimizers[0], **scheduler_kwargs)
        self._loss = loss
        self._neural_net = self.models[0]

    def forward(self, x):
        """
        Forward pass implementation for the PINN
        solver.

        :param torch.Tensor x: Input tensor.
        :return: PINN solution.
        :rtype: torch.Tensor
        """
        return self.neural_net(x)

    def configure_optimizers(self):
        """
        Optimizer configuration for the PINN
        solver.

        :return: The optimizers and the schedulers
        :rtype: tuple(list, list)
        """
        # if the problem is an InverseProblem, add the unknown parameters
        # to the parameters that the optimizer needs to optimize
        if isinstance(self.problem, InverseProblem):
            self.optimizers[0].add_param_group(
                {'params': [self.problem.unknown_parameters[var] for var in self.problem.unknown_variables]}
                )
        return self.optimizers, [self.scheduler]

    def _loss_data(self, input, output):
        return self.loss(self.forward(input), output)


    def _loss_phys(self, samples, equation):
        try:
            residual = equation.residual(samples, self.forward(samples))
        except TypeError:
            residual = equation.residual(samples, self.forward(samples),
                params_=self.problem.unknown_parameters)
        return self.loss(torch.zeros_like(residual, requires_grad=True), residual)


    def training_step(self, batch, batch_idx):
        """
        PINN solver training step.

        :param batch: The batch element in the dataloader.
        :type batch: tuple
        :param batch_idx: The batch index.
        :type batch_idx: int
        :return: The sum of the loss functions.
        :rtype: LabelTensor
        """

        dataloader = self.trainer.train_dataloader
        condition_losses = []

        condition_idx = batch['condition']

        for condition_id in range(condition_idx.min(), condition_idx.max()+1):

            condition_name = dataloader.condition_names[condition_id]
            condition = self.problem.conditions[condition_name]
            pts = batch['pts']

            if len(batch) == 2:
                samples = pts[condition_idx == condition_id]
                loss = self._loss_phys(samples, condition.equation)
            elif len(batch) == 3:
                samples = pts[condition_idx == condition_id]
                ground_truth = batch['output'][condition_idx == condition_id]
                loss = self._loss_data(samples, ground_truth)
            else:
                raise ValueError("Batch size not supported")

            # TODO for users this us hard to remebeber when creating a new solver, to fix in a smarter way
            loss = loss.as_subclass(torch.Tensor)

#            # add condition losses and accumulate logging for each epoch
            condition_losses.append(loss * condition.data_weight)
            self.log(condition_name + '_loss', float(loss),
                     prog_bar=True, logger=True, on_epoch=True, on_step=False)

        # TODO Fix the bug, tot_loss is a label tensor without labels
        # we need to pass it as a torch tensor to make everything work
        total_loss = sum(condition_losses)
        self.log('mean_loss', float(total_loss / len(condition_losses)),
                 prog_bar=True, logger=True, on_epoch=True, on_step=False)
        # clamp unknown parameters of the InverseProblem to their domain ranges
        if isinstance(self.problem, InverseProblem):
            for v in self.problem.unknown_variables:
                self.problem.unknown_parameters[v].data.clamp_(
                        self.problem.unknown_parameter_domain.range_[v][0],
                        self.problem.unknown_parameter_domain.range_[v][1])


        return total_loss

    @property
    def scheduler(self):
        """
        Scheduler for the PINN training.
        """
        return self._scheduler

    @property
    def neural_net(self):
        """
        Neural network for the PINN training.
        """
        return self._neural_net

    @property
    def loss(self):
        """
        Loss for the PINN training.
        """
        return self._loss
