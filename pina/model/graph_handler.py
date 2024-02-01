from torch_geometric.data import Data
from torch_geometric.nn import radius_graph
from pina import LabelTensor
import torch

class GraphHandler():

    def __init__(self, dt, neighbors = 2):
        super().__init__()
        self.neighbors = neighbors
        self.dt = dt


    def create_graph(self, pts, labels, steps):
        time_window = labels.shape[1]
        x = torch.cat([pts[i,:,st-time_window:st].extract(['u']) for i,st in enumerate(steps)]).squeeze(-1)
        coordinates = torch.Tensor(torch.cat([pts[i,:,st].extract(['x']) for i,st in enumerate(steps)])).squeeze(-1)
        variables = torch.cat([pts[i,:,st].extract(['alpha', 'beta', 'gamma']) for i,st in enumerate(steps)])
        variables = LabelTensor(variables, labels=['alpha', 'beta', 'gamma'])
        time = torch.cat([pts[i,:,st].extract(['t']) for i,st in enumerate(steps)])
        time = LabelTensor(time, labels=['t'])
        num_x = torch.unique(coordinates).shape[0]
        batch = torch.cat([torch.ones(num_x)*i for i in range(pts.shape[0])]).to(device=pts.device)
        dx = coordinates[1] - coordinates[0]
        radius = self.neighbors * dx + 0.0001
        edge_index = radius_graph(coordinates, r=radius, loop=False, batch=batch)
        graph = Data(x=x, edge_index=edge_index, edge_attr=None)
        graph.y = labels
        graph.pos = coordinates.unsqueeze(-1)
        graph.variables = variables
        graph.dt = self.dt
        graph.batch = batch.long()
        graph.time = time
        return graph
    

    def update_graph(self, graph, pred, labels, steps, batch_size):
        time_window = labels.shape[1]
        graph.x = torch.cat((graph.x, pred), dim=1)[:,time_window:]
        graph.y = labels
        num_x = labels.shape[0] // batch_size
        time = [torch.ones(num_x)*steps[i]*self.dt for i in range(len(steps))]
        graph.time = torch.cat(time).unsqueeze(-1).to(device=pred.device)
        return graph
