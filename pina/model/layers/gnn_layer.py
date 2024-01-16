import torch
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing, InstanceNorm
from torch_cluster import radius_graph

class GNN_Layer(MessagePassing):
    """
    Message passing layer
    """
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 hidden_features: int,
                 time_window: int,
                 n_variables: int,
                 n_spatial: int = 1):
        """
        Initialize message passing layers
        Args:
            in_features (int): number of node input features
            out_features (int): number of node output features
            hidden_features (int): number of hidden features
            time_window (int): number of input/output timesteps (temporal bundling)
            n_variables (int): number of equation specific parameters used in the solver
            n_spatial (int): number of spatial variables (ex: x --> 1, [x,y] --> 2)
        """
        super(GNN_Layer, self).__init__(node_dim=-2, aggr='mean')
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = hidden_features
        self.time_window = time_window
        self.n_variables = n_variables
        self.n_spatial = n_spatial

        # Message network -- equation 8
        self.message_net_1 = torch.nn.Sequential(torch.nn.Linear(2*self.in_features + self.time_window + self.n_spatial + self.n_variables, self.hidden_features), torch.nn.SiLU())
        self.message_net_2 = torch.nn.Sequential(torch.nn.Linear(self.hidden_features, self.hidden_features), torch.nn.SiLU())

        # Update network -- equation 9
        self.update_net_1 = torch.nn.Sequential(torch.nn.Linear(self.in_features + self.hidden_features + self.n_variables, self.hidden_features), torch.nn.SiLU())
        self.update_net_2 = torch.nn.Sequential(torch.nn.Linear(self.hidden_features, self.out_features), torch.nn.SiLU())

        self.norm = InstanceNorm(self.hidden_features)


    def forward(self, graph):
        """
        Propagate messages along edges
        """
        f = self.propagate(edge_index=graph.edge_index, 
                           x=graph.x,
                           u=graph.u, 
                           pos=graph.pos, 
                           variables=graph.variables)
        return f


    def message(self, x_i, x_j, u_i, u_j, pos_i, pos_j, variables_i):
        """
        Message update following formula 8 of the paper
        """
        message = self.message_net_1(torch.cat((x_i, x_j, u_i - u_j, pos_i - pos_j, variables_i), dim=-1))
        message = self.message_net_2(message)
        return message


    def update(self, message, x, variables):
        """
        Node update following formula 9 of the paper
        """
        update = self.update_net_1(torch.cat((x, message, variables), dim=-1))
        update = self.update_net_2(update)
        if self.in_features == self.out_features:
            return x + update
        else:
            return update


class GraphHandler():
    """
    Creates and manages a graph with following attrubutes:
    - graph.u: values of u(x,t) at point x and time t in considered window
    - graph.pos: spatial coordinates
    - graph.variables: variables of the equation (time and parameters)
    - graph.x: node features
    """
    def __init__(self, coordinates, variables, dt, num_neighs=10):
        super().__init__()
        self.n = num_neighs
        self.graph = self.create_ball_graph(coordinates, variables, dt)

    def create_ball_graph(self, coordinates, variables, dt):
        # Get the smallest distance between the coordinates
        if len(coordinates.shape) == 1:
            dx = coordinates[1]-coordinates[0]
        else:
            dx = torch.pdist(coordinates).min()

        # Set the radius so as to include the nearest neighbours 
        radius = self.n * dx + 0.000001
        edge_index = radius_graph(coordinates, r=radius, loop=False)

        # Features x are computed by the encoder preceeding the gnn_layer
        graph = Data(x = None, edge_index = edge_index, edge_attr = None)
        graph.pos = coordinates
        graph.u = None
        graph.variables = variables
        graph.dt = dt
        return graph
    
    def data_to_graph(self, data):
        self.graph.u = data