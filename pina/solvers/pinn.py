""" Module for PINN """
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


torch.pi = torch.acos(torch.zeros(1)).item() * 2  # which is 3.1415927410125732


class PINN(SolverInterface):
    """
    PINN solver class. This class implements Physics Informed Neural 
    Network solvers, using a user specified ``model`` to solve a specific
    ``problem``. 

    .. seealso::

        **Original reference**: Karniadakis, G. E., Kevrekidis, I. G., Lu, L., 
        Perdikaris, P., Wang, S., & Yang, L. (2021). 
        Physics-informed machine learning. Nature Reviews Physics, 3(6), 422-440.
        <https://doi.org/10.1038/s42254-021-00314-5>`_.
    """

    def __init__(self,
                 problem,
                 model,
                 extra_features=None,
                 loss=torch.nn.MSELoss(),
                 optimizer=torch.optim.Adam,
                 optimizer_kwargs={'lr': 0.001},
                 scheduler=ConstantLR,
                 scheduler_kwargs={"factor": 1, "total_iters": 0},
                 ):
        '''
        :param AbstractProblem problem: The formualation of the problem.
        :param torch.nn.Module model: The neural network model to use.
        :param torch.nn.Module loss: The loss function used as minimizer,
            default torch.nn.MSELoss().
        :param torch.nn.Module extra_features: The additional input
            features to use as augmented input.
        :param torch.optim.Optimizer optimizer: The neural network optimizer to
            use; default is `torch.optim.Adam`.
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
        """Forward pass implementation for the PINN
           solver.

        :param torch.tensor x: Input data. 
        :return: PINN solution.
        :rtype: torch.tensor
        """
        # extract labels
        x = x.extract(self.problem.input_variables)
        # perform forward pass
        output = self.neural_net(x).as_subclass(LabelTensor)
        # set the labels
        output.labels = self.problem.output_variables
        return output

    def configure_optimizers(self):
        """Optimizer configuration for the PINN
           solver.

        :return: The optimizers and the schedulers
        :rtype: tuple(list, list)
        """
        return self.optimizers, [self.scheduler]

    def training_step(self, batch, batch_idx):
        """PINN solver training step.

        :param batch: The batch element in the dataloader.
        :type batch: tuple
        :param batch_idx: The batch index.
        :type batch_idx: int
        :return: The sum of the loss functions.
        :rtype: LabelTensor
        """

        condition_losses = []

        for condition_name, samples in batch.items():

            if condition_name not in self.problem.conditions:
                raise RuntimeError('Something wrong happened.')

            condition = self.problem.conditions[condition_name]

            # PINN loss: equation evaluated on location or input_points
            if hasattr(condition, 'equation'):
                target = condition.equation.residual(
                    samples, self.forward(samples))
                loss = self.loss(torch.zeros_like(target), target)
            # PINN loss: evaluate model(input_points) vs output_points
            elif hasattr(condition, 'output_points'):
                input_pts, output_pts = samples
                loss = self.loss(self.forward(input_pts), output_pts)

            condition_losses.append(loss * condition.data_weight)

        # TODO Fix the bug, tot_loss is a label tensor without labels
        # we need to pass it as a torch tensor to make everything work
        total_loss = sum(condition_losses)
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
