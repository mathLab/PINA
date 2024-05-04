import torch
from pina import LabelTensor
from torch_geometric.data import Data
from torch_geometric.nn import radius_graph

class GraphHandler():
    """
    Class that manages the creation and update of a graph.
    """
    
    def __init__(self, dt, neighbors=2):
        """
        Initialization.
        
        :param torch.float32 dt: time step value.
        :param int neighbors: number of neighbors in a radius graph. Default: 2.
        """
        
        super().__init__()
        self.neighbors = neighbors
        self.dt = dt


    def create_graph(self, pts, labels, steps):
        """
        Creator of a radius graph.
        
        :param torch.Tensor pts: tensor of data points.
        :param torch.Tensor labels: tensor of labels corresponding to data points.
        :param list steps: list of time steps.
        :return: graph.
        :rtype: Data
        """
        
        time_window = labels.shape[1]
        
        # Features, cartesian coordinates, time, and additional variables
        x = torch.cat([pts[i,:,st-time_window:st].extract(['u']) for i,st in enumerate(steps)]).squeeze(-1)
        coordinates = torch.Tensor(torch.cat([pts[i,:,st].extract(['x']) for i,st in enumerate(steps)])).squeeze(-1)
        variables = torch.cat([pts[i,:,st].extract(['alpha', 'beta', 'gamma']) for i,st in enumerate(steps)])
        variables = LabelTensor(variables, labels=['alpha', 'beta', 'gamma'])
        time = torch.cat([pts[i,:,st].extract(['t']) for i,st in enumerate(steps)])
        time = LabelTensor(time, labels=['t'])
        
        # Batch index
        num_x = torch.unique(coordinates).shape[0]
        batch = torch.cat([torch.ones(num_x)*i for i in range(pts.shape[0])]).to(device=pts.device)
        
        # Edge index
        dx = coordinates[1] - coordinates[0]
        radius = self.neighbors * dx + 0.0001
        edge_index = radius_graph(coordinates, r=radius, loop=False, batch=batch)
        
        # Graph: features, labels, positions, time, variables, time step, batch index
        graph = Data(x=x, edge_index=edge_index, edge_attr=None)
        graph.y = labels
        graph.pos = coordinates.unsqueeze(-1)
        graph.time = time
        graph.variables = variables
        graph.dt = self.dt
        graph.batch = batch.long()
        
        return graph
    

    def update_graph(self, graph, pred, labels, steps, batch_size):
        """
        Update of the graph: only affects features, time, and labels.
        
        :param Data graph: graph to be updated.
        :param torch.Tensor pred: new features to be put in the graph.
        :param torch.Tensor labels: new labels to be put in the graph.
        :param list steps: list of time steps.
        :param int batch_size: size of the batch.
        :return: graph.
        :rtype: Data
        """
        
        time_window = labels.shape[1]
        num_x = labels.shape[0] // batch_size
        time = [torch.ones(num_x)*steps[i]*self.dt for i in range(len(steps))]
        
        # Update
        graph.x = torch.cat((graph.x, pred), dim=1)[:,time_window:]
        graph.y = labels
        graph.time = torch.cat(time).unsqueeze(-1).to(device=pred.device)
        
        return graph