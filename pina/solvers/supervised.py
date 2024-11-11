""" Module for SupervisedSolver """
import torch
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch.nn.modules.loss import _Loss
from ..optim import TorchOptimizer, TorchScheduler
from .solver import SolverInterface
from ..label_tensor import LabelTensor
from ..utils import check_consistency
from ..loss.loss_interface import LossInterface
from ..condition import InputOutputPointsCondition


class SupervisedSolver(SolverInterface):
    r"""
    SupervisedSolver solver class. This class implements a SupervisedSolver,
    using a user specified ``model`` to solve a specific ``problem``.

    The  Supervised Solver class aims to find
    a map between the input :math:`\mathbf{s}:\Omega\rightarrow\mathbb{R}^m`
    and the output :math:`\mathbf{u}:\Omega\rightarrow\mathbb{R}^m`. The input
    can be discretised in space (as in :obj:`~pina.solvers.rom.ROMe2eSolver`),
    or not (e.g. when training Neural Operators).

    Given a model :math:`\mathcal{M}`, the following loss function is
    minimized during training:

    .. math::
        \mathcal{L}_{\rm{problem}} = \frac{1}{N}\sum_{i=1}^N
        \mathcal{L}(\mathbf{u}_i - \mathcal{M}(\mathbf{v}_i))

    where :math:`\mathcal{L}` is a specific loss function,
    default Mean Square Error:

    .. math::
        \mathcal{L}(v) = \| v \|^2_2.

    In this context :math:`\mathbf{u}_i` and :math:`\mathbf{v}_i` means that
    we are seeking to approximate multiple (discretised) functions given
    multiple (discretised) input functions.
    """
    accepted_condition_types = [InputOutputPointsCondition.condition_type[0]]
    __name__ = 'SupervisedSolver'

    def __init__(self,
                 problem,
                 model,
                 loss=None,
                 optimizer=None,
                 scheduler=None,
                 extra_features=None):
        """
        :param AbstractProblem problem: The formualation of the problem.
        :param torch.nn.Module model: The neural network model to use.
        :param torch.nn.Module loss: The loss function used as minimizer,
            default :class:`torch.nn.MSELoss`.
        :param torch.nn.Module extra_features: The additional input
            features to use as augmented input.
        :param torch.optim.Optimizer optimizer: The neural network optimizer to
            use; default is :class:`torch.optim.Adam`.
        :param torch.optim.LRScheduler scheduler: Learning
            rate scheduler.
        """
        if loss is None:
            loss = torch.nn.MSELoss()

        if optimizer is None:
            optimizer = TorchOptimizer(torch.optim.Adam, lr=0.001)

        if scheduler is None:
            scheduler = TorchScheduler(torch.optim.lr_scheduler.ConstantLR)

        super().__init__(models=model,
                         problem=problem,
                         optimizers=optimizer,
                         schedulers=scheduler,
                         extra_features=extra_features)

        # check consistency
        check_consistency(loss, (LossInterface, _Loss),
                          subclass=False)
        self._loss = loss
        self._model = self._pina_models[0]
        self._optimizer = self._pina_optimizers[0]
        self._scheduler = self._pina_schedulers[0]
        self.validation_condition_losses = {
            k: {'loss': [],
                'count': []} for k in self.problem.conditions.keys()}

    def forward(self, x):
        """Forward pass implementation for the solver.

        :param torch.Tensor x: Input tensor.
        :return: Solver solution.
        :rtype: torch.Tensor
        """

        output = self._model(x)

        output.labels = self.problem.output_variables
        return output

    def configure_optimizers(self):
        """Optimizer configuration for the solver.

        :return: The optimizers and the schedulers
        :rtype: tuple(list, list)
        """
        self._optimizer.hook(self._model.parameters())
        self._scheduler.hook(self._optimizer)
        return ([self._optimizer.optimizer_instance],
                [self._scheduler.scheduler_instance])

    def training_step(self, batch):
        """Solver training step.

        :param batch: The batch element in the dataloader.
        :type batch: tuple
        :param batch_idx: The batch index.
        :type batch_idx: int
        :return: The sum of the loss functions.
        :rtype: LabelTensor
        """
        condition_loss = []
        batches = batch.get_supervised_data()
        for points in batches:
            input_pts, output_pts, _ = points
            loss_ = self.loss_data(input_pts=input_pts, output_pts=output_pts)
            condition_loss.append(loss_.as_subclass(torch.Tensor))
        loss = sum(condition_loss)
        self.log("mean_loss", float(loss), prog_bar=True, logger=True,
                 on_epoch=True,
                 on_step=False, batch_size=self.trainer.data_module.batch_size)
        return loss

    def validation_step(self, batch):
        """
        Solver validation step.
        """

        batch = batch.supervised
        condition_idx = batch.condition_indices
        for i in range(condition_idx.min(), condition_idx.max() + 1):
            condition_name = self.trainer.data_module.condition_names[i]
            condition = self.problem.conditions[condition_name]
            pts = batch.input_points
            out = batch.output_points
            if condition_name not in self.problem.conditions:
                raise RuntimeError("Something wrong happened.")

            # for data driven mode
            if not hasattr(condition, "output_points"):
                raise NotImplementedError(
                    f"{type(self).__name__} works only in data-driven mode.")

            output_pts = out[condition_idx == i]
            input_pts = pts[condition_idx == i]

            loss_ = self.loss_data(input_pts=input_pts, output_pts=output_pts)
            self.validation_condition_losses[condition_name]['loss'].append(
                loss_)
            self.validation_condition_losses[condition_name]['count'].append(
                len(input_pts))

    def on_validation_epoch_end(self):
        """
        Solver validation epoch end.
        """
        total_loss = []
        total_count = []
        for k, v in self.validation_condition_losses.items():
            local_counter = torch.tensor(v['count']).to(self.device)
            n_elements = torch.sum(local_counter)
            loss = torch.sum(
                torch.stack(v['loss']) * local_counter) / n_elements
            loss = loss.as_subclass(torch.Tensor)
            total_loss.append(loss)
            total_count.append(n_elements)
            self.log(
                k + "_loss",
                loss,
                prog_bar=True,
                logger=True,
                on_epoch=True,
                on_step=False,
                batch_size=self.trainer.data_module.batch_size,
            )
        total_count = (torch.tensor(total_count, dtype=torch.float32).
                       to(self.device))
        mean_loss = (torch.sum(torch.stack(total_loss) * total_count) /
                     total_count)
        self.log(
            "val_loss",
            mean_loss,
            prog_bar=True,
            logger=True,
            on_epoch=True,
            on_step=False,
            batch_size=self.trainer.data_module.batch_size,
        )
        for key in self.validation_condition_losses.keys():
            self.validation_condition_losses[key]['loss'] = []
            self.validation_condition_losses[key]['count'] = []

    def test_step(self, batch, batch_idx) -> STEP_OUTPUT:
        """
        Solver test step.
        """

        raise NotImplementedError("Test step not implemented yet.")

    def loss_data(self, input_pts, output_pts):
        """
        The data loss for the Supervised solver. It computes the loss between
        the network output against the true solution. This function
        should not be override if not intentionally.

        :param LabelTensor input_pts: The input to the neural networks.
        :param LabelTensor output_pts: The true solution to compare the
            network solution.
        :return: The residual loss averaged on the input coordinates
        :rtype: torch.Tensor
        """
        return self._loss(self.forward(input_pts), output_pts)

    @property
    def scheduler(self):
        """
        Scheduler for training.
        """
        return self._scheduler

    @property
    def optimizer(self):
        """
        Optimizer for training.
        """
        return self._optimizer

    @property
    def model(self):
        """
        Neural network for training.
        """
        return self._model

    @property
    def loss(self):
        """
        Loss for training.
        """
        return self._loss
