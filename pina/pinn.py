""" Module for PINN """
import torch
import torch.optim.lr_scheduler as lrs


from .solver import SolverInterface
from .label_tensor import LabelTensor
from .utils import check_consistency
from .writer import Writer


torch.pi = torch.acos(torch.zeros(1)).item() * 2  # which is 3.1415927410125732


class PINN(SolverInterface):

    def __init__(self,
                 problem,
                 model,
                 extra_features=None,
                 loss = torch.nn.MSELoss,  # TODO to be changed in LossInstance
                 optimizer=torch.optim.Adam,
                 optimizer_kwargs={'lr' : 0.001},
                 scheduler=lrs.ConstantLR,
                 scheduler_kwargs={"factor": 1, "total_iters": 0},
                 ):
        '''
        :param AbstractProblem problem: the formualation of the problem.
        :param torch.nn.Module model: the neural network model to use.
        :param torch.nn.Module loss: the loss function used as minimizer,
            default torch.nn.MSELoss.
        :param torch.nn.Module extra_features: the additional input
            features to use as augmented input.
        :param torch.optim.Optimizer optimizer: the neural network optimizer to
            use; default is `torch.optim.Adam`.
        :param dict optimizer_kwargs: Optimizer constructor keyword args.
        :param float lr: the learning rate; default is 0.001.
        :param torch.optim.LRScheduler scheduler: Learning
            rate scheduler.
        :param dict scheduler_kwargs: LR scheduler constructor keyword args.
        '''
        super().__init__(model=model, problem=problem, extra_features=extra_features)
        
        # check consistency 
        check_consistency(optimizer, torch.optim.Optimizer, 'optimizer', subclass=True)
        check_consistency(optimizer_kwargs, dict, 'optimizer_kwargs')
        check_consistency(scheduler, lrs.LRScheduler, 'scheduler', subclass=True)
        check_consistency(scheduler_kwargs, dict, 'scheduler_kwargs')
        # TODO check consistency loss

        # assign variables
        self._optimizer = optimizer(self.model.parameters(), **optimizer_kwargs)
        self._scheduler = scheduler(self._optimizer, **scheduler_kwargs)
        self._loss = loss()
        self._writer = Writer()


    def forward(self, x):
        """ Forward pass implementation for the PINN
            solver.

        :param torch.tensor x: Input data. 
        :return: PINN solution.
        :rtype: torch.tensor
        """
        x = x.extract(self.problem.input_variables)

        for feature in self._extra_features:
            x = x.append(feature(x))

        output = self.model(x).as_subclass(LabelTensor)
        output.labels = self.problem.output_variables

        return output

    def configure_optimizers(self):
        """Optimizer configuration for the PINN
           solver.

        :return: The optimizers and the schedulers
        :rtype: tuple(list, list)
        """
        return [self._optimizer], [self._scheduler]
    
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

            if samples is None or samples.nelement() == 0:
                continue

            condition = self.problem.conditions[condition_name]

            if hasattr(condition, 'equation'): # TODO FIX for any loss
                target = condition.equation.residual(samples, self.forward(samples))
                loss = self._loss(torch.zeros_like(target), target)
            elif hasattr(condition, 'output_points'):
                loss = self._loss(samples, condition.output_points)

            condition_losses.append(loss * condition.data_weight)

        # TODO Fix the bug, tot_loss is a label tensor without labels
        # we need to pass it as a torch tensor to make everything work
        total_loss = sum(condition_losses)
        return total_loss