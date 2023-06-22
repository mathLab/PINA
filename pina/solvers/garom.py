""" Module for PINN """
import torch
try:
    from torch.optim.lr_scheduler import LRScheduler  # torch >= 2.0
except ImportError:
    from torch.optim.lr_scheduler import _LRScheduler as LRScheduler # torch < 2.0

from torch.optim.lr_scheduler import ConstantLR
from .solver import SolverInterface
from ..utils import check_consistency
from ..loss import LossInterface, LpLoss
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
                 loss = LpLoss(p=1),
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
                 regularizer = False
                 ):
        """
        :param AbstractProblem problem: The formualation of the problem.
        :param torch.nn.Module generator: The neural network model to use
            for the generator.
        :param torch.nn.Module discriminator: The neural network model to use
            for the discriminator.
        :param torch.nn.Module loss: The loss function used as minimizer,
            default ``LpLoss(p=1)`` as in the original paper.
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

        super().__init__(models=[generator, discriminator],
                         problem=problem,
                         optimizers=[optimizer_generator, optimizer_discriminator],
                         optimizers_kwargs=[optimizer_generator_kwargs, optimizer_discriminator_kwargs])
        
        # set automatic optimization for GANs
        self.automatic_optimization = False
        
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
        self._schedulers = [scheduler_generator(self.optimizers[0], 
                                                  **scheduler_generator_kwargs),
                               scheduler_generator(self.optimizers[1],
                                                    **scheduler_discriminator_kwargs)]
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

        for condition_name, samples in batch.items():

            if condition_name not in self.problem.conditions:
                raise RuntimeError('Something wrong happened.')

            condition = self.problem.conditions[condition_name]

            # for data driven mode
            if hasattr(condition, 'output_points'):

                # get data
                parameters, input_pts = samples

                # get optimizers
                opt_gen, opt_disc = self.optimizers

                # ---------------------
                #  Train Discriminator
                # ---------------------
                opt_disc.zero_grad()

                # Generate a batch of images
                gen_imgs = self.generator(parameters)
            
                # Discriminator pass
                d_real = self.discriminator([input_pts, parameters])
                d_fake = self.discriminator([gen_imgs.detach(), parameters]) 

                # evaluate loss
                d_loss_real = self._loss(d_real, input_pts)
                d_loss_fake = self._loss(d_fake, gen_imgs.detach())
                d_loss = d_loss_real - self.k * d_loss_fake

                # backward step
                d_loss.backward()
                opt_disc.step()
                
                # -----------------
                #  Train Generator
                # -----------------
                opt_gen.zero_grad()

                # Generate a batch of images
                gen_imgs = self.generator(parameters)

                # generator loss
                r_loss = self._loss(input_pts, gen_imgs)
                d_fake = self.discriminator([gen_imgs, parameters])
                g_loss = self._loss(d_fake, gen_imgs) + self.regularizer * r_loss

                # backward step
                g_loss.backward()
                opt_gen.step()

                # ----------------
                # Update weights
                # ----------------
                diff = torch.mean(self.gamma * d_loss_real - d_loss_fake)

                # Update weight term for fake samples
                self.k += self.lambda_k * diff.item()
                self.k = min(max(self.k, 0), 1)  # Constraint to interval [0, 1]

            else:
                raise NotImplementedError('GAROM works only in data-driven mode.')

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