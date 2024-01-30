import torch
from torch_geometric.nn import MessagePassing, InstanceNorm

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


    def forward(self, x, u, pos, variables, edge_index, batch):
        """
        Propagate messages along edges
        """
        f = self.propagate(edge_index=edge_index, x=x, u=u, pos=pos, variables=variables)
        f = self.norm(f, batch)
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