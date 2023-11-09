""" Module for SupervisedSolver """
import torch
try:
    from torch.optim.lr_scheduler import LRScheduler  # torch >= 2.0
except ImportError:
    from torch.optim.lr_scheduler import _LRScheduler as LRScheduler  # torch < 2.0

from torch.optim.lr_scheduler import ConstantLR

from .solver import SolverInterface
from ..label_tensor import LabelTensor
from ..utils import check_consistency
from ..loss import LossInterface
from torch.nn.modules.loss import _Loss


class SupervisedSolver(SolverInterface):
    """
    SupervisedSolver solver class. This class implements a SupervisedSolver,
    using a user specified ``model`` to solve a specific ``problem``. 
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
        :param AbstractProblem problem: The formualation of the problem.
        :param torch.nn.Module model: The neural network model to use.
        :param torch.nn.Module loss: The loss function used as minimizer,
            default :class:`torch.nn.MSELoss`.
        :param torch.nn.Module extra_features: The additional input
            features to use as augmented input.
        :param torch.optim.Optimizer optimizer: The neural network optimizer to
            use; default is :class:`torch.optim.Adam`.
        :param dict optimizer_kwargs: Optimizer constructor keyword args.
        :param float lr: The learning rate; default is 0.001.
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
        """Forward pass implementation for the solver.

        :param torch.Tensor x: Input tensor. 
        :return: Solver solution.
        :rtype: torch.Tensor
        """
        return self.neural_net(x)

    def configure_optimizers(self):
        """Optimizer configuration for the solver.

        :return: The optimizers and the schedulers
        :rtype: tuple(list, list)
        """
        return self.optimizers, [self.scheduler]

    def training_step(self, batch, batch_idx):
        """Solver training step.

        :param batch: The batch element in the dataloader.
        :type batch: tuple
        :param batch_idx: The batch index.
        :type batch_idx: int
        :return: The sum of the loss functions.
        :rtype: LabelTensor
        """

        dataloader = self.trainer.train_dataloader
        condition_idx = batch['condition']

        for condition_id in range(condition_idx.min(), condition_idx.max()+1):

            condition_name = dataloader.condition_names[condition_id]
            condition = self.problem.conditions[condition_name]
            pts = batch['pts']
            out = batch['output']

            if condition_name not in self.problem.conditions:
                raise RuntimeError('Something wrong happened.')

            # for data driven mode
            if not hasattr(condition, 'output_points'):
                raise NotImplementedError('Supervised solver works only in data-driven mode.')
            
            output_pts = out[condition_idx == condition_id]
            input_pts = pts[condition_idx == condition_id]

            loss = self.loss(self.forward(input_pts), output_pts) * condition.data_weight
            loss = loss.as_subclass(torch.Tensor)

        self.log('mean_loss', float(loss), prog_bar=True, logger=True)
        return loss

    @property
    def scheduler(self):
        """
        Scheduler for training.
        """
        return self._scheduler

    @property
    def neural_net(self):
        """
        Neural network for training.
        """
        return self._neural_net

    @property
    def loss(self):
        """
        Loss for training.
        """
        return self._loss
