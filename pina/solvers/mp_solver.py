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
from graph_handler import GraphHandler
#import networkx as nx
#import torch_geometric
#from matplotlib import pyplot as plt

class MessagePassing(SolverInterface):

    def __init__(self, problem, model, time_window, time_res, dt, neighbors=2, unrolling=2, adversarial=True,
                 extra_features = None, loss=torch.nn.MSELoss(reduction='sum'), optimizer=torch.optim.Adam,
                 optimizer_kwargs={'lr': 0.001}, scheduler = ConstantLR, scheduler_kwargs={'factor': 1, 'total_iters': 0}):
        super().__init__(models=[model], problem=problem, optimizers=[optimizer],
                         optimizers_kwargs=[optimizer_kwargs], extra_features=extra_features)
        
        check_consistency(scheduler, LRScheduler, subclass=True)
        check_consistency(scheduler_kwargs, dict)
        check_consistency(loss, (LossInterface, _Loss), subclass=False)

        if isinstance(self.problem, InverseProblem):
            raise ValueError('MessagePassing works only for forward problems.')
        else:
            self._params = None

        self._scheduler = scheduler(self.optimizers[0], **scheduler_kwargs)
        self._loss = loss
        self._neural_net = self.models[0]
        self.unrolling = unrolling if adversarial else 0
        self.num_iter = time_res if adversarial else 1
        self.time_res = time_res
        self.time_window = time_window
        self.handler = GraphHandler(neighbors=neighbors, dt=dt)


    def configure_optimizers(self):
        return self.optimizers, [self._scheduler]
    

    def forward(self, data, pos, time, variables, batch, edge_index, dt):
        return self.neural_net.torchmodel(data, pos, time, variables, batch, edge_index, dt)
    

    def create_labels(self, pts, steps):
        target = [pts[i,:,st:st+self.time_window].extract(['u']) for i,st in enumerate(steps)]
        return torch.cat(target).squeeze(-1)
    

    def training_step(self, batch, batch_idx):
        dataloader = self.trainer.train_dataloader
        condition_idx = batch['condition']
        for _ in range(self.num_iter):
            #self.optimizers[0].zero_grad() #Serve?
            for condition_id in range(condition_idx.min(), condition_idx.max()+1):
                if sys.version_info >= (3,8):
                    condition_name = dataloader.condition_names[condition_id]
                else:
                    condition_name = dataloader.loaders.condition_names[condition_id]
                condition = self.problem.conditions[condition_name]
                pts = batch['pts']
                out = batch['output']   #inutile
                batch_size = pts.shape[0]
                if condition_name not in self.problem.conditions:
                    raise RuntimeError('Something wrong happened.')
                input_pts = pts[condition_idx == condition_id]
                output_pts = out[condition_idx == condition_id] #inutile

                steps = [t for t in range(self.time_window, self.time_res - self.time_window - (self.time_window*self.unrolling) +1)]
                random_steps = random.choices(steps, k=batch_size)
                labels = self.create_labels(input_pts, random_steps)
                graph = self.handler.create_graph(input_pts, labels, random_steps)
                
                #nx_graph = torch_geometric.utils.to_networkx(graph)
                #nodes = nx_graph.nodes()
                #positions = {node: (index, 0) for index, node in enumerate(nodes)}
                #nx.draw_networkx_nodes(nx_graph, pos=positions, node_size=1, node_color='skyblue')
                #nx.draw_networkx_edges(nx_graph, pos=positions, edge_color='gray')
                #nx.draw_networkx_labels(nx_graph, pos=positions, font_color='black', font_size=8)
                #plt.title("Line Graph with Edges Visualization")
                #plt.axis('off')
                #plt.savefig("graph_visualization.png")
                
                with torch.no_grad():
                    for _ in range(self.unrolling):
                        random_steps = [rs + self.time_window for rs in random_steps]
                        labels = self.create_labels(input_pts, random_steps)
                        pred = self.forward(graph.x, graph.pos, graph.time, graph.variables, graph.batch, graph.edge_index, graph.dt)
                        graph = self.handler.update_graph(graph, pred, labels, random_steps, batch_size)
                
                pred = self.forward(graph.x, graph.pos, graph.time, graph.variables, graph.batch, graph.edge_index, graph.dt)
                loss = self.loss(pred, graph.y) * condition.data_weight
                loss = loss.as_subclass(torch.Tensor)

        self.log('mean_loss', float(loss), prog_bar=True, logger=True)
        return loss
    

    @property
    def scheduler(self):
        return self._scheduler
    

    @property
    def neural_net(self):
        return self._neural_net
    

    @property
    def loss(self):
        return self._loss