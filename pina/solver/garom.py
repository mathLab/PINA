"""Module for GAROM"""

import torch
from torch.nn.modules.loss import _Loss
from .solver import MultiSolverInterface
from ..condition import InputTargetCondition
from ..utils import check_consistency
from ..loss import LossInterface, PowerLoss


class GAROM(MultiSolverInterface):
    """
    GAROM solver class. This class implements Generative Adversarial
    Reduced Order Model solver, using user specified ``models`` to solve
    a specific order reduction``problem``.

    .. seealso::

        **Original reference**: Coscia, D., Demo, N., & Rozza, G. (2023).
        *Generative Adversarial Reduced Order Modelling*.
        DOI: `arXiv preprint arXiv:2305.15881.
        <https://doi.org/10.48550/arXiv.2305.15881>`_.
    """

    accepted_conditions_types = InputTargetCondition

    def __init__(
        self,
        problem,
        generator,
        discriminator,
        loss=None,
        optimizer_generator=None,
        optimizer_discriminator=None,
        scheduler_generator=None,
        scheduler_discriminator=None,
        gamma=0.3,
        lambda_k=0.001,
        regularizer=False,
    ):
        """
        :param AbstractProblem problem: The formualation of the problem.
        :param torch.nn.Module generator: The neural network model to use
            for the generator.
        :param torch.nn.Module discriminator: The neural network model to use
            for the discriminator.
        :param torch.nn.Module loss: The loss function used as minimizer,
            default ``None``. If ``loss`` is ``None`` the defualt
            ``PowerLoss(p=1)`` is used, as in the original paper.
        :param Optimizer optimizer_generator: The neural
            network optimizer to use for the generator network
            , default is `torch.optim.Adam`.
        :param Optimizer optimizer_discriminator: The neural
            network optimizer to use for the discriminator network
            , default is `torch.optim.Adam`.
        :param Scheduler scheduler_generator: Learning
            rate scheduler for the generator.
        :param Scheduler scheduler_discriminator: Learning
            rate scheduler for the discriminator.
        :param dict scheduler_discriminator_kwargs: LR scheduler constructor
            keyword args.
        :param gamma: Ratio of expected loss for generator and discriminator,
            defaults to 0.3.
        :type gamma: float
        :param lambda_k: Learning rate for control theory optimization,
            defaults to 0.001.
        :type lambda_k: float
        :param regularizer: Regularization term in the GAROM loss,
            defaults to False.
        :type regularizer: bool

        .. warning::
            The algorithm works only for data-driven model. Hence in the
            ``problem`` definition the codition must only contain ``input``
            (e.g. coefficient parameters, time parameters), and ``target``.
        """

        # set loss
        if loss is None:
            loss = PowerLoss(p=1)

        super().__init__(
            models=[generator, discriminator],
            problem=problem,
            optimizers=[optimizer_generator, optimizer_discriminator],
            schedulers=[
                scheduler_generator,
                scheduler_discriminator,
            ],
            use_lt=False,
        )

        # check consistency
        check_consistency(
            loss, (LossInterface, _Loss, torch.nn.Module), subclass=False
        )
        self._loss = loss

        # set automatic optimization for GANs
        self.automatic_optimization = False

        # check consistency
        check_consistency(gamma, float)
        check_consistency(lambda_k, float)
        check_consistency(regularizer, bool)

        # began hyperparameters
        self.k = 0
        self.gamma = gamma
        self.lambda_k = lambda_k
        self.regularizer = float(regularizer)

    def forward(self, x, mc_steps=20, variance=False):
        """
        Forward step for GAROM solver

        :param x: The input tensor.
        :type x: torch.Tensor
        :param mc_steps: Number of montecarlo samples to approximate the
            expected value, defaults to 20.
        :type mc_steps: int
        :param variance: Returining also the sample variance of the solution,
            defaults to False.
        :type variance: bool
        :return: The expected value of the generator distribution. If
            ``variance=True`` also the
            sample variance is returned.
        :rtype: torch.Tensor | tuple(torch.Tensor, torch.Tensor)
        """

        # sampling
        field_sample = [self.sample(x) for _ in range(mc_steps)]
        field_sample = torch.stack(field_sample)

        # extract mean
        mean = field_sample.mean(dim=0)

        if variance:
            var = field_sample.var(dim=0)
            return mean, var

        return mean

    def sample(self, x):
        """TODO"""
        # sampling
        return self.generator(x)

    def _train_generator(self, parameters, snapshots):
        """
        Private method to train the generator network.
        """
        optimizer = self.optimizer_generator
        optimizer.zero_grad()

        generated_snapshots = self.sample(parameters)

        # generator loss
        r_loss = self._loss(snapshots, generated_snapshots)
        d_fake = self.discriminator([generated_snapshots, parameters])
        g_loss = (
            self._loss(d_fake, generated_snapshots) + self.regularizer * r_loss
        )

        # backward step
        g_loss.backward()
        optimizer.step()

        return r_loss, g_loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        """
        This method is called at the end of each training batch, and ovverides
        the PytorchLightining implementation for logging the checkpoints.

        :param torch.Tensor outputs: The output from the model for the
            current batch.
        :param tuple batch: The current batch of data.
        :param int batch_idx: The index of the current batch.
        :return: Whatever is returned by the parent
            method ``on_train_batch_end``.
        :rtype: Any
        """
        # increase by one the counter of optimization to save loggers
        (
            self.trainer.fit_loop.epoch_loop.manual_optimization.optim_step_progress.total.completed
        ) += 1

        return super().on_train_batch_end(outputs, batch, batch_idx)

    def _train_discriminator(self, parameters, snapshots):
        """
        Private method to train the discriminator network.
        """
        optimizer = self.optimizer_discriminator
        optimizer.zero_grad()

        # Generate a batch of images
        generated_snapshots = self.sample(parameters)

        # Discriminator pass
        d_real = self.discriminator([snapshots, parameters])
        d_fake = self.discriminator([generated_snapshots, parameters])

        # evaluate loss
        d_loss_real = self._loss(d_real, snapshots)
        d_loss_fake = self._loss(d_fake, generated_snapshots.detach())
        d_loss = d_loss_real - self.k * d_loss_fake

        # backward step
        d_loss.backward()
        optimizer.step()

        return d_loss_real, d_loss_fake, d_loss

    def _update_weights(self, d_loss_real, d_loss_fake):
        """
        Private method to Update the weights of the generator and discriminator
        networks.
        """

        diff = torch.mean(self.gamma * d_loss_real - d_loss_fake)

        # Update weight term for fake samples
        self.k += self.lambda_k * diff.item()
        self.k = min(max(self.k, 0), 1)  # Constraint to interval [0, 1]
        return diff

    def optimization_cycle(self, batch):
        """GAROM solver training step.

        :param batch: The batch element in the dataloader.
        :type batch: tuple
        :return: The sum of the loss functions.
        :rtype: LabelTensor
        """
        condition_loss = {}
        for condition_name, points in batch:
            parameters, snapshots = (
                points["input"],
                points["target"],
            )
            d_loss_real, d_loss_fake, d_loss = self._train_discriminator(
                parameters, snapshots
            )
            r_loss, g_loss = self._train_generator(parameters, snapshots)
            diff = self._update_weights(d_loss_real, d_loss_fake)
            condition_loss[condition_name] = r_loss

        # some extra logging
        self.store_log("d_loss", float(d_loss), self.get_batch_size(batch))
        self.store_log("g_loss", float(g_loss), self.get_batch_size(batch))
        self.store_log(
            "stability_metric",
            float(d_loss_real + torch.abs(diff)),
            self.get_batch_size(batch),
        )
        return condition_loss

    def validation_step(self, batch):
        condition_loss = {}
        for condition_name, points in batch:
            parameters, snapshots = (
                points["input"],
                points["target"],
            )
            snapshots_gen = self.generator(parameters)
            condition_loss[condition_name] = self._loss(
                snapshots, snapshots_gen
            )
        loss = self.weighting.aggregate(condition_loss)
        self.store_log("val_loss", loss, self.get_batch_size(batch))
        return loss

    def test_step(self, batch):
        condition_loss = {}
        for condition_name, points in batch:
            parameters, snapshots = (
                points["input"],
                points["target"],
            )
            snapshots_gen = self.generator(parameters)
            condition_loss[condition_name] = self._loss(
                snapshots, snapshots_gen
            )
        loss = self.weighting.aggregate(condition_loss)
        self.store_log("test_loss", loss, self.get_batch_size(batch))
        return loss

    @property
    def generator(self):
        """TODO"""
        return self.models[0]

    @property
    def discriminator(self):
        """TODO"""
        return self.models[1]

    @property
    def optimizer_generator(self):
        """TODO"""
        return self.optimizers[0].instance

    @property
    def optimizer_discriminator(self):
        """TODO"""
        return self.optimizers[1].instance

    @property
    def scheduler_generator(self):
        """TODO"""
        return self.schedulers[0].instance

    @property
    def scheduler_discriminator(self):
        """TODO"""
        return self.schedulers[1].instance
