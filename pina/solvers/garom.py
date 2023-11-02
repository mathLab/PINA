""" Module for GAROM """

import torch
try:
    from torch.optim.lr_scheduler import LRScheduler  # torch >= 2.0
except ImportError:
    from torch.optim.lr_scheduler import _LRScheduler as LRScheduler # torch < 2.0

from torch.optim.lr_scheduler import ConstantLR
from .solver import SolverInterface
from ..utils import check_consistency
from ..loss import LossInterface, PowerLoss
from torch.nn.modules.loss import _Loss


class GAROM(SolverInterface):
    """
    GAROM solver class. This class implements Generative Adversarial
    Reduced Order Model solver, using user specified ``models`` to solve
    a specific order reduction``problem``. 

    .. seealso::

        **Original reference**: Coscia, D., Demo, N., & Rozza, G. (2023).
        Generative Adversarial Reduced Order Modelling. 
        arXiv preprint arXiv:2305.15881.
        <https://doi.org/10.48550/arXiv.2305.15881>`_.
    """
    def __init__(self,
                 problem,
                 generator,
                 discriminator,
                 extra_features=None,
                 loss = None,
                 optimizer_generator=torch.optim.Adam,
                 optimizer_generator_kwargs={'lr' : 0.001},
                 optimizer_discriminator=torch.optim.Adam,
                 optimizer_discriminator_kwargs={'lr' : 0.001},
                 scheduler_generator=ConstantLR,
                 scheduler_generator_kwargs={"factor": 1, "total_iters": 0},
                 scheduler_discriminator=ConstantLR,
                 scheduler_discriminator_kwargs={"factor": 1, "total_iters": 0},
                 gamma = 0.3,
                 lambda_k = 0.001,
                 regularizer = False,
                 ):
        """
        :param AbstractProblem problem: The formualation of the problem.
        :param torch.nn.Module generator: The neural network model to use
            for the generator.
        :param torch.nn.Module discriminator: The neural network model to use
            for the discriminator.
        :param torch.nn.Module extra_features: The additional input
            features to use as augmented input. It should either be a
            list of torch.nn.Module, or a dictionary. If a list it is 
            passed the extra features are passed to both network. If a 
            dictionary is passed, the keys must be ``generator`` and
            ``discriminator`` and the values a list of torch.nn.Module
            extra features for each.
        :param torch.nn.Module loss: The loss function used as minimizer,
            default ``None``. If ``loss`` is ``None`` the defualt
            ``PowerLoss(p=1)`` is used, as in the original paper.
        :param torch.optim.Optimizer optimizer_generator: The neural
            network optimizer to use for the generator network
            , default is `torch.optim.Adam`.
        :param dict optimizer_generator_kwargs: Optimizer constructor keyword
            args. for the generator.
        :param torch.optim.Optimizer optimizer_discriminator: The neural
            network optimizer to use for the discriminator network
            , default is `torch.optim.Adam`.
        :param dict optimizer_discriminator_kwargs: Optimizer constructor keyword
            args. for the discriminator.
        :param torch.optim.LRScheduler scheduler_generator: Learning
            rate scheduler for the generator.
        :param dict scheduler_generator_kwargs: LR scheduler constructor keyword args.
        :param torch.optim.LRScheduler scheduler_discriminator: Learning
            rate scheduler for the discriminator.
        :param dict scheduler_discriminator_kwargs: LR scheduler constructor keyword args.
        :param gamma: Ratio of expected loss for generator and discriminator, defaults to 0.3.
        :type gamma: float, optional
        :param lambda_k: Learning rate for control theory optimization, defaults to 0.001.
        :type lambda_k: float, optional
        :param regularizer: Regularization term in the GAROM loss, defaults to False.
        :type regularizer: bool, optional

        .. warning::
            The algorithm works only for data-driven model. Hence in the ``problem`` definition
            the codition must only contain ``input_points`` (e.g. coefficient parameters, time
            parameters), and ``output_points``.
        """

        if isinstance(extra_features, dict):
            extra_features = [extra_features['generator'], extra_features['discriminator']]

        super().__init__(models=[generator, discriminator],
                         problem=problem,
                         extra_features=extra_features,
                         optimizers=[optimizer_generator, optimizer_discriminator],
                         optimizers_kwargs=[optimizer_generator_kwargs, optimizer_discriminator_kwargs])
        
        # set automatic optimization for GANs
        self.automatic_optimization = False

        # set loss
        if loss is None:
            loss = PowerLoss(p=1)
        
        # check consistency 
        check_consistency(scheduler_generator, LRScheduler, subclass=True)
        check_consistency(scheduler_generator_kwargs, dict)
        check_consistency(scheduler_discriminator, LRScheduler, subclass=True)
        check_consistency(scheduler_discriminator_kwargs, dict)
        check_consistency(loss, (LossInterface, _Loss))
        check_consistency(gamma, float)
        check_consistency(lambda_k, float)
        check_consistency(regularizer, bool)

        # assign schedulers
        self._schedulers = [
            scheduler_generator(
                self.optimizers[0], **scheduler_generator_kwargs),
            scheduler_discriminator(
                self.optimizers[1],
                **scheduler_discriminator_kwargs)
        ]

        # loss and writer 
        self._loss = loss

        # began hyperparameters
        self.k = 0
        self.gamma = gamma
        self.lambda_k = lambda_k
        self.regularizer = float(regularizer)

    def forward(self, x, mc_steps=20, variance=False):

        # sampling
        field_sample = [self.sample(x) for _ in range(mc_steps)]
        field_sample = torch.stack(field_sample)

        # extract mean
        mean = field_sample.mean(dim=0)

        if variance:
            var = field_sample.var(dim=0)
            return mean, var

        return mean
    
    def configure_optimizers(self):
        """Optimizer configuration for the GAROM
           solver.

        :return: The optimizers and the schedulers
        :rtype: tuple(list, list)
        """
        return self.optimizers, self._schedulers

    def sample(self, x):
        # sampling
        return self.generator(x)
    
    def training_step(self, batch, batch_idx):
        """PINN solver training step.

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
                raise NotImplementedError('GAROM works only in data-driven mode.')

            # get data
            snapshots = out[condition_idx == condition_id]
            parameters = pts[condition_idx == condition_id]


            # ---------------------
            #  Train Discriminator
            # ---------------------
            def _train_discriminator(parameters, snapshots):

                optimizer = self.optimizers[1]
                optimizer.zero_grad()

                # Generate a batch of images
                generated_snapshots = self.generator(parameters)
        
                # Discriminator pass
                d_real = self.discriminator([snapshots, parameters])
                d_fake = self.discriminator([generated_snapshots, parameters]) 

                # evaluate loss
                d_loss_real = self._loss(d_real, snapshots)
                d_loss_fake = self._loss(d_fake, generated_snapshots.detach())
                d_loss = d_loss_real - self.k * d_loss_fake

                # backward step
                d_loss.backward(retain_graph=True)
                optimizer.step()

                return d_loss_real, d_loss_fake, d_loss
            
            d_loss_real, d_loss_fake, d_loss = _train_discriminator(
                parameters, snapshots)

            # -----------------
            #  Train Generator
            # -----------------
            def _train_generator(parameters, snapshots):

                optimizer = self.optimizers[0]

                generated_snapshots = self.generator(parameters)

                # generator loss
                r_loss = self._loss(snapshots, generated_snapshots)
                d_fake = self.discriminator([generated_snapshots, parameters])
                g_loss = self._loss(d_fake, generated_snapshots) + self.regularizer * r_loss

                # backward step
                g_loss.backward()
                optimizer.step()

                return r_loss, g_loss

            r_loss, g_loss = _train_generator(parameters, snapshots)
            # ----------------
            # Update weights
            # ----------------
            def _update_weights(d_loss_real, d_loss_fake):

                diff = torch.mean(self.gamma * d_loss_real - d_loss_fake)

                # Update weight term for fake samples
                self.k += self.lambda_k * diff.item()
                self.k = min(max(self.k, 0), 1)  # Constraint to interval [0, 1]
                return diff
            
            diff = _update_weights(d_loss_real, d_loss_fake)

            # logging
            self.log('mean_loss', float(r_loss), prog_bar=True, logger=True)
            self.log('d_loss', float(d_loss), prog_bar=True, logger=True)
            self.log('g_loss', float(g_loss), prog_bar=True, logger=True)
            self.log('stability_metric', float(d_loss_real + torch.abs(diff)), prog_bar=True, logger=True)

        return
    
    @property
    def generator(self):
        return self.models[0]
    
    @property
    def discriminator(self):
        return self.models[1]
    
    @property
    def optimizer_generator(self):
        return self.optimizers[0]
    
    @property
    def optimizer_discriminator(self):
        return self.optimizers[1]
    
    @property
    def scheduler_generator(self):
        return self._schedulers[0]
    
    @property
    def scheduler_discriminator(self):
        return self._schedulers[1]
