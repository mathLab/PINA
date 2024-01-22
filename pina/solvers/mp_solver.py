from pina.solvers import SolverInterface
from pina.utils import check_consistency
from pina.loss import LossInterface
from pina.problem import InverseProblem
import sys
import random
import torch
try:
    from torch.optim.lr_scheduler import LRScheduler
except ImportError:
    from torch.optim.lr_scheduler import _LRScheduler as LRScheduler
from torch.optim.lr_scheduler import ConstantLR
from torch.nn.modules.loss import _Loss
from pina.model.graph_handler import GraphHandler


class MessagePassing(SolverInterface):
    """
    Message Passing Neural PDE solver class.
    """
    def __init__(
        self,
        problem,
        model,
        dt,
        time_window,
        unrolling_list = [2],
        time_res = 250,
        adversarial = True,
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
        """
        :param AbstractProblem problem: The formulation of the problem.
        :param torch.nn.Module model: The neural network model to use.
        :param float dt: length of a temporal step.
        :param int time_window: temporal window length.
        :param list unrolling_list: The list of possible unrollings;
            default: :list:[2].
        :param int time_res: The time resolution; default is :int:`250`.
        :param bool adversarial: Whether to use or not adversarial training; 
            default is :bool:`True`.
        :param torch.nn.Module extra_features: The additional input
            features to use as augmented input.
        :param torch.nn.Module loss: The loss function used as minimizer,
            default :class:`torch.nn.MSELoss`.
        :param torch.optim.Optimizer optimizer: The neural network optimizer to
            use; default is :class:`torch.optim.Adam`.
        :param dict optimizer_kwargs: Optimizer constructor keyword args.
        :param torch.optim.LRScheduler scheduler: Learning rate scheduler.
        :param dict scheduler_kwargs: LR scheduler constructor keyword args.
        """
        super().__init__(models=[model],
                         problem=problem,
                         optimizers=[optimizer],
                         optimizers_kwargs=[optimizer_kwargs],
                         extra_features=extra_features)

        # check consistency
        check_consistency(scheduler, LRScheduler, subclass=True)
        check_consistency(scheduler_kwargs, dict)
        check_consistency(loss, (LossInterface, _Loss), subclass=False)
        
        # inverse problem handling
        if isinstance(self.problem, InverseProblem):
            raise ValueError('Message Passing works only for forward problems.')
        else:
            self._params = None

        # assign variables
        self._scheduler = scheduler(self.optimizers[0], **scheduler_kwargs)
        self._loss = loss
        self._neural_net = self.models[0]
        self.unrolling_list = unrolling_list if adversarial else [1]
        self.num_iter = time_res if adversarial else 1
        self.time_res = time_res
        self.time_window = time_window
        self.dt = dt
        self.handler = GraphHandler(self.dt, num_neighs=5)
        
        
    def configure_optimizers(self):
        """
        Optimizer configuration for the solver.
        
        :return: The optimizers and the schedulers.
        :rtype: tuple(list, list).
        """
        return self.optimizers, [self.scheduler]


    def forward(self, x):
        """
        Message passing solver forward step.
        
        :param x: Input graph.
        :return: Message Passing solution.
        :rtype: torch.Tensor
        """
        return self.neural_net.torchmodel(x)
    
    
    def training_step(self, batch, batch_idx):
        """
        Message passing training step.

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
            if sys.version_info >= (3,8):
                condition_name = dataloader.condition_names[condition_id]
            else:
                condition_name = dataloader.loaders.condition_names[condition_id]
            condition = self.problem.conditions[condition_name]
            pts = batch['pts']
            batch_size = pts.shape[0]
            
            if condition_name not in self.problem.conditions:
                raise RuntimeError("Something wrong happened.")
    
            input_pts = pts[condition_idx == condition_id]
            
            for _ in range(self.num_iter):
                unrolling = random.choice(self.unrolling_list)
                steps = [t for t in range(self.time_window, self.time_res - self.time_window - (self.time_window*unrolling) +1)]
                random_steps = random.choices(steps, k = batch_size)
                data, variables, coordinates, btc = self.create_data(input_pts, random_steps)
                self.handler.graph = self.handler.create_ball_graph(coordinates=coordinates, data = data, variables=variables, batch=btc.long())
                with torch.no_grad():
                    for _ in range(unrolling):
                        random_steps = [rs + self.time_window for rs in random_steps]
                        self.handler.graph.u = self.forward(self.handler.graph)
                        self.handler.graph.variables = self.update_variables(input_pts, random_steps)

                labels = self.create_labels(input_pts, random_steps)
                pred = self.forward(self.handler.graph)
                
                loss = self.loss(pred, labels) * condition.data_weight
                loss = loss.as_subclass(torch.Tensor)

        self.log('mean_loss', float(loss), prog_bar=True, logger=True)
        return loss
    
    
    def create_data(self, pts, steps):
        """
        Creation of the data to be inserted in the graph.
        
        :param torch.Tensor pts: batched points.
        :param list steps: random temporal steps.
        :return: input, variables, coordinates, batch and target data in a reduced time window.
        :rtype: tuple(torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor).
        """
        
        data, variables, coordinates, batch = torch.Tensor(), torch.Tensor(), torch.Tensor(), torch.Tensor()
        for i,st in enumerate(steps):
            d = pts[i,:,st-self.time_window:st].extract(['u'])
            coord = pts[i,:,0].extract(['x'])
            v = pts[i,:,0].extract(['t', 'alpha', 'beta', 'gamma'])
            data = torch.cat((data, d), 0)
            variables = torch.cat((variables, v), 0)
            coordinates = torch.cat((coordinates, coord), 0)
            batch = torch.cat((batch, torch.ones(pts.shape[1])*i), 0)
        
        return data.squeeze(-1), variables, torch.Tensor(coordinates).squeeze(-1), batch
    
    
    def update_variables(self, pts, steps):
        variables = torch.Tensor()
        for i in range(len(steps)):
            v = pts[i,:,0].extract(['t', 'alpha', 'beta', 'gamma'])
            variables = torch.cat((variables, v), 0)
        return variables
    
    
    def create_labels(self, pts, steps):
        labels = torch.Tensor()
        for i,st in enumerate(steps):
            l = pts[i,:,st:self.time_window+st].extract(['u'])
            labels = torch.cat((labels, l), 0)
        return labels.squeeze(-1)

        
    @property
    def scheduler(self):
        return self._scheduler

    @property
    def neural_net(self):
        return self._neural_net

    @property
    def loss(self):
        return self._loss