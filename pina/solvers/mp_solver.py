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
            raise ValueError('Message Passing only for forward problems.')
            #self._params = self.problem.unknown_parameters
        else:
            self._params = None

    def forward(self, x):
        return self.neural_net(x)
    
    def training_step(self, batch, batch_idx):

#############################
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
#############################         
        u, x, variables = pts.shape.....

        # Randomly choose number of unrollings
        unrolling = 2
        import random
        unrolled_graphs = random.choice(unrolling)
        steps = [t for t in range(graph_creator.tw,
                                graph_creator.t_res - graph_creator.tw - (graph_creator.tw * unrolled_graphs) + 1)]
        # Randomly choose starting (time) point at the PDE solution manifold
        random_steps = random.choices(steps, k=u.shape[0])
        data, labels = graph_creator.create_data(u, random_steps)
        if f'{model}' == 'GNN':
            graph = graph_creator.create_graph(data, labels, x, variables, random_steps).to(device)
        else:
            data, labels = data.to(device), labels.to(device)

        # Unrolling of the equation which serves as input at the current step
        # This is the pushforward trick!!!
        with torch.no_grad():
            for _ in range(unrolled_graphs):
                random_steps = [rs + graph_creator.tw for rs in random_steps]
                _, labels = graph_creator.create_data(u_super, random_steps)
                if f'{model}' == 'GNN':
                    pred = model(graph)
                    graph = graph_creator.create_next_graph(graph, pred, labels, random_steps).to(device)
                else:
                    data = model(data)
                    labels = labels.to(device)

        if f'{model}' == 'GNN':
            pred = model(graph)
            loss = criterion(pred, graph.y)
        else:
            pred = model(data)
            loss = criterion(pred, labels)

        return loss
    
    def configure_optimizers(self):
        """Optimizer configuration for the solver.

        :return: The optimizers and the schedulers
        :rtype: tuple(list, list)
        """
        return self.optimizers, [self.scheduler]

    @property
    def scheduler(self):
        return self._scheduler

    @property
    def neural_net(self):
        return self._neural_net

    @property
    def loss(self):
        return self._loss
