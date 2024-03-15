""" Module for PINN """

import torch
import sys

from abc import ABCMeta, abstractmethod
from ..solver import SolverInterface
from ...utils import check_consistency
from ...loss import LossInterface
from ...problem import InverseProblem
from torch.nn.modules.loss import _Loss


class PINNInterface(SolverInterface, metaclass=ABCMeta):
    """
    Base PINN solver class. This class implements the Solver Interface
    for Physics Informed Neural Network solvers. It is used internally in PINA
    to buld new PINNs solvers by inheriting from it.
    """

    def __init__(
        self,
        models,
        problem,
        optimizers,
        optimizers_kwargs,
        extra_features=None,
        loss=torch.nn.MSELoss(),
    ):
        """
        :param models: A torch neural network model instance.
        :type models: torch.nn.Module
        :param problem: A problem definition instance.
        :type problem: AbstractProblem
        :param list(torch.optim.Optimizer) optimizer: A list of neural network optimizers to
            use.
        :param list(dict) optimizer_kwargs: A list of optimizer constructor keyword args.
        :param list(torch.nn.Module) extra_features: The additional input
            features to use as augmented input. If ``None`` no extra features
            are passed. If it is a list of :class:`torch.nn.Module`, the extra feature
            list is passed to all models. If it is a list of extra features' lists,
            each single list of extra feature is passed to a model.
        :param torch.nn.Module loss: The loss function used as minimizer,
            default :class:`torch.nn.MSELoss`.
        """
        super().__init__(
            models=models,
            problem=problem,
            optimizers=optimizers,
            optimizers_kwargs=optimizers_kwargs,
            extra_features=extra_features,
        )

        # check consistency
        check_consistency(loss, (LossInterface, _Loss), subclass=False)

        # assign variables
        self._loss = loss

        # inverse problem handling
        if isinstance(self.problem, InverseProblem):
            self._params = self.problem.unknown_parameters
        else:
            self._params = None

        # variable used internally to store residual losses at each epoch
        # this variable save the residual at each iteration (not weighted)
        self.__res_losses = []


    def on_train_start(self):
        """
        On training epoch start this function is call to do global checks for
        the PINN training.
        """
        
        # 1. Check the verison for dataloader
        dataloader = self.trainer.train_dataloader
        if sys.version_info < (3, 8):
            dataloader = dataloader.loaders
        self._dataloader = dataloader

        # 2. Check if we are dealing with inverse problem
        if isinstance(self.problem, InverseProblem):
            self._clamp_params = self._clamp_inverse_problem_params
        else:
            self._clamp_params = lambda : None

        return super().on_train_start()


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

        condition_losses = []
        condition_idx = batch["condition"]

        for condition_id in range(condition_idx.min(), condition_idx.max() + 1):

            condition_name = self._dataloader.condition_names[condition_id]
            condition = self.problem.conditions[condition_name]
            pts = batch["pts"]

            if len(batch) == 2:
                samples = pts[condition_idx == condition_id]
                loss = self._loss_phys(samples, condition.equation,
                                       condition_name)
            elif len(batch) == 3:
                samples = pts[condition_idx == condition_id]
                ground_truth = batch["output"][condition_idx == condition_id]
                loss = self._loss_data(samples, ground_truth, condition_name)
            else:
                raise ValueError("Batch size not supported")

            # add condition losses for each epoch
            condition_losses.append(loss * condition.data_weight)

        # clamp unknown parameters in InverseProblem (if needed)
        self._clamp_params()

        # storing logs
        self._store_log('mean_loss',
                        sum(self.__res_losses)/len(self.__res_losses))
        self.__res_losses = []
        total_loss = sum(condition_losses)
        return total_loss


    def loss_data(self, input, output):
        """
        The data loss for the PINN solver. It computes the loss between
        the network output against the true solution.

        :param LabelTensor input: The input to the neural networks.
        :param LabelTensor output: The true solution to compare the network
            solution
        :return: The residual loss averaged on the input coordinates
        :rtype: torch.Tensor
        """
        return self.loss(self.forward(input), output)


    @abstractmethod
    def loss_phys(self, samples, equation):
        """
        Computes the physics loss for the PINN solver based on given
        samples and equation.

        :param LabelTensor samples: The samples to evaluate the physics loss.
        :param EquationInterface equation: The governing equation
            representing the physics.
        :return: The physics loss calculated based on given
            samples and equation.
        :rtype: torch.Tensor
        """
        pass


    def _clamp_inverse_problem_params(self):
        """
        Clamps the parameters of the inverse problem
        solver to the specified ranges.
        """
        for v in self._params:
            self._params[v].data.clamp_(
                self.problem.unknown_parameter_domain.range_[v][0],
                self.problem.unknown_parameter_domain.range_[v][1],
            )


    def _loss_data(self, input, output, condition_name):
        """
        Computes the data loss for the PINN solver based on input,
        output, and condition name. This function is a wrapper of the function
        :meth:`loss_data` used internally in PINA to handle the logging step.

        :param LabelTensor input: The input to the neural networks.
        :param LabelTensor output: The true solution to compare the network
            solution
        :param str condition_name: The condition name for tracking purposes.
        :return: The computed data loss.
        :rtype: torch.Tensor
        """
        loss_val = self.loss_data(input, output)
        self._store_log(name=condition_name+'_loss', loss_val=float(loss_val))
        return loss_val.as_subclass(torch.Tensor)


    def _loss_phys(self, samples, equation, condition_name):
        """
        Computes the physics loss for the PINN solver based on input,
        output, and condition name. This function is a wrapper of the function
        :meth:`loss_phys` used internally in PINA to handle the logging step.

        :param LabelTensor samples: The samples to evaluate the physics loss.
        :param EquationInterface equation: The governing equation
            representing the physics.
        :param str condition_name: The condition name for tracking purposes.
        :return: The computed data loss.
        :rtype: torch.Tensor
        """
        loss_val = self.loss_phys(samples, equation)
        self._store_log(name=condition_name+'_loss', loss_val=float(loss_val))
        return loss_val.as_subclass(torch.Tensor)


    def _store_log(self, name, loss_val):
        """
        Stores the loss value in the logger.

        :param str name: The name of the loss.
        :param torch.Tensor loss_val: The value of the loss.
        """
        self.log(
                name,
                loss_val,
                prog_bar=True,
                logger=True,
                on_epoch=True,
                on_step=False,
            )
        self.__res_losses.append(loss_val)
        


    @property
    def loss(self):
        """
        Loss for the PINN training.
        """
        return self._loss
