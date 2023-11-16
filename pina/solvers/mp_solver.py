from .solver import SolverInterface
from abc import ABCMeta, abstractmethod
from ..model.network import Network
import pytorch_lightning
from ..utils import check_consistency
from ..problem import AbstractProblem
import torch
try:
    from torch.optim.lr_scheduler import LRScheduler  # torch >= 2.0
except ImportError:
    from torch.optim.lr_scheduler import _LRScheduler as LRScheduler  # torch < 2.0

from torch.optim.lr_scheduler import ConstantLR
from ..label_tensor import LabelTensor
from ..loss import LossInterface
from ..problem import InverseProblem
from torch.nn.modules.loss import _Loss

torch.pi = torch.acos(torch.zeros(1)).item() * 2

class MessagePassing(SolverInterface):
    
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

        # inverse problem handling
        if isinstance(self.problem, InverseProblem):
            self._params = self.problem.unknown_parameters
        else:
            self._params = None


    def forward(self, x):
        return self.neural_net(x)
    

    @property
    def scheduler(self):
        return self._scheduler

    @property
    def neural_net(self):
        return self._neural_net

    @property
    def loss(self):
        return self._loss
