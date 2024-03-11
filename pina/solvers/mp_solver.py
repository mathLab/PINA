import sys
import torch
import random
from pina.loss import LossInterface
from pina.problem import InverseProblem
from pina.solvers import SolverInterface
from pina.utils import check_consistency
from torch.nn.modules.loss import _Loss
from torch.optim.lr_scheduler import ConstantLR
from graph_handler import GraphHandler
try:
    from torch.optim.lr_scheduler import LRScheduler
except ImportError:
    from torch.optim.lr_scheduler import _LRScheduler as LRScheduler

class MessagePassing(SolverInterface):
    """
    Message passing solver class. It implements the message passing neural PDE
    solver, using user-specified ``models`` to solve a specific ``problem``.
    It can be used to solve conservation form PDEs.
    """

    def __init__(self,
                 problem,
                 model,
                 neighbors=2,
                 unrolling=2,
                 extra_features = None,
                 loss=torch.nn.MSELoss(reduction='mean'),
                 optimizer=torch.optim.Adam, 
                 optimizer_kwargs={'lr': 0.001},
                 scheduler = ConstantLR,
                 scheduler_kwargs={'factor': 1, 'total_iters': 0}):
        """
        Initialization.
        
        :param AbstractProblem problem: formualation of the problem.
        :param torch.nn.Module model: neural network model.
        :param int neighbors: number of neighbors in the graph. Default: 2.
        :param int unrolling: number of times the model is run without back
            propagation since the previous occurrence. Default: 2.
        :param torch.nn.Module extra_features: additional input features to be
            used as augmented input.
        :param torch.nn.Module loss: loss function to be minimized.
            Default: class:`torch.nn.MSELoss`.
        :param torch.optim.Optimizer optimizer: neural network optimizer.
            Default: class:`torch.optim.Adam`.
        :param dict optimizer_kwargs: optimizer constructor keyword args.
        :param torch.optim.LRScheduler scheduler: Learning rate scheduler.
        :param dict scheduler_kwargs: LR scheduler constructor keyword args.
        """
        
        super().__init__(models=[model],
                         problem=problem,
                         optimizers=[optimizer],
                         optimizers_kwargs=[optimizer_kwargs],
                         extra_features=extra_features)
        
        check_consistency(scheduler, LRScheduler, subclass=True)
        check_consistency(scheduler_kwargs, dict)
        check_consistency(loss, (LossInterface, _Loss), subclass=False)

        if isinstance(self.problem, InverseProblem):
            raise ValueError('MessagePassing works only for forward problems.')
        else:
            self._params = None

        self._loss = loss
        self._neural_net = self.models[0]
        self._scheduler = scheduler(self.optimizers[0], **scheduler_kwargs)
        self.automatic_optimization = False
        
        self.unrolling = unrolling
        self.time_window = model.time_window
        self.dt = torch.Tensor(self.problem.input_pts['data'].extract(['t'])[0,0,1] - self.problem.input_pts['data'].extract(['t'])[0,0,0])
        self.handler = GraphHandler(neighbors=neighbors, dt=self.dt)
        self.t_res = len(self.problem.input_pts['data'].extract('t').unique())
       

    def configure_optimizers(self):
        """
        Optimizer configuration for the solver.

        :return: optimizers and schedulers
        :rtype: tuple(list, list)
        """
        
        return self.optimizers, [self._scheduler]
    

    def forward(self, graph):
        """
        Forward pass implementation for the solver.

        :param Data graph: graph to be used for message passing.
        :return: solver solution.
        :rtype: torch.Tensor
        """
        
        return self.neural_net.torchmodel(graph)
    

    def create_labels(self, pts, steps):
        """
        Definition of the labels, i.e. targets of the data points sampled at 
        random time steps.
        
        :param torch.Tensor pts: data points.
        :param list steps: list of random time steps.
        :return: target.
        :rtype: torch.Tensor
        """
        
        target = [pts[i,:,st:st+self.time_window].extract(['u']) for i,st in enumerate(steps)]
        return torch.cat(target).squeeze(-1)
    

    def training_step(self, batch, batch_idx):
        """
        Solver training step.

        :param tuple batch: batch element in the dataloader.
        :param int batch_idx: batch index.
        """
    
        dataloader = self.trainer.train_dataloader
        condition_idx = batch['condition']
        optimizer = self.optimizers[0]
        optimizer.zero_grad()
        
        for condition_id in range(condition_idx.min(), condition_idx.max()+1):
            if sys.version_info >= (3,8):
                condition_name = dataloader.condition_names[condition_id]
            else:
                condition_name = dataloader.loaders.condition_names[condition_id]
            condition = self.problem.conditions[condition_name]
            pts = batch['pts']
            batch_size = pts.shape[0]
            if condition_name not in self.problem.conditions:
                raise RuntimeError('Something wrong happened.')
            
            input_pts = pts[condition_idx == condition_id]

            
            # List of candidate time steps
            steps = [t for t in range(self.time_window, self.t_res - self.time_window - (self.time_window*self.unrolling) +1)]
            
            # List of randomly sampled time steps
            random_steps = random.choices(steps, k=batch_size)
            
            # Creation of the graph
            labels = self.create_labels(input_pts, random_steps)
            graph = self.handler.create_graph(input_pts, labels, random_steps)
            
            # Unrolling
            with torch.no_grad():
                for _ in range(self.unrolling):
                    random_steps = [rs + self.time_window for rs in random_steps]
                    labels = self.create_labels(input_pts, random_steps)
                    pred = self.forward(graph)
                    graph = self.handler.update_graph(graph, pred, labels, random_steps, batch_size)
            
            # Computation of the loss
            pred = self.forward(graph)
            loss = self.loss(pred, graph.y) * condition.data_weight
            loss = loss.as_subclass(torch.Tensor)
            loss.backward()
            optimizer.step()

            self.log('mean_loss', float(loss), prog_bar=False, logger=True, on_step=False, on_epoch=True)
            del loss, graph, input_pts, pred, pts

        return
    

    def on_train_start(self):
        """
        Customization of the training loop. Every epoch set by the user
        corresponds to ``self.t_res`` internal epochs. This is performed
        to simulate a mean over the random time steps sampled during the 
        training loop.
        """
        
        self.trainer.fit_loop.max_epochs *= self.t_res
        self.tot_loss = 0
        self.count = 0


    def on_train_epoch_end(self):
        """
        Customization of the training loop. It prints the loss after every
        ``self.t_res`` internal epochs, corresponding to one external epoch.
        """
        loss = self.trainer.logged_metrics['mean_loss']
        self.tot_loss += loss
        self.count += 1
        if (self.trainer.current_epoch + 1) % 250 == 0:
            print(f'External epoch {(self.trainer.current_epoch + 1)//250} - Loss {self.tot_loss/self.count}')
            self.count = 0
            self.tot_loss = 0
    

    @property
    def scheduler(self):
        """
        Training scheduler.
        """
        return self._scheduler
    

    @property
    def neural_net(self):
        """
        Training neural network.
        """
        return self._neural_net
    

    @property
    def loss(self):
        """
        Training loss.
        """
        return self._loss