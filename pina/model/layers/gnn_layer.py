import torch
from torch.nn import Sequential, Linear, SiLU
from torch_geometric.nn import MessagePassing, InstanceNorm

class GNN_Layer(MessagePassing):
    """
    Implementation of a message-passing layer of a graph neural network.
    """

    def __init__(self, 
                 in_features,
                 out_features,
                 hidden_features,
                 time_window,
                 n_variables,
                 n_spatial=1):
        """ 
        Initialization.
        
        :param int in_features: node-wise dimension of the features tensor 
            in input.
        :param int out_features: node-wise dimension of the features tensor 
            in output.
        :param int hidden_features: hidden node-wise dimension of the 
            features tensor.
        :param int time_window: width of the considered time interval.
        :param int n_variables: number of variables (including time).
        :param int n_spatial: spatial dimension of the considered problem. 
            Default: 1.
        """
        
        super(GNN_Layer, self).__init__(node_dim=-2, aggr='mean')
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = hidden_features
        self.time_window = time_window
        self.n_variables = n_variables
        self.n_spatial = n_spatial

        self.message_net_1 = Sequential(Linear(2 * self.in_features + self.time_window + self.n_spatial + self.n_variables, 
                                               self.hidden_features), SiLU())
        
        self.message_net_2 = Sequential(Linear(self.hidden_features, 
                                               self.hidden_features), SiLU())

        self.update_net_1 = Sequential(Linear(self.in_features + self.hidden_features + self.n_variables, 
                                              self.hidden_features), SiLU())
        
        self.update_net_2 = Sequential(Linear(self.hidden_features, 
                                              self.out_features), SiLU())

        self.norm = InstanceNorm(self.hidden_features)


    def forward(self, edge_index, x, u, pos, variables, batch):
        """
        Trigger of the message-passing routine. It invokes the message
        and the update methods.

        :param torch.Tensor edge_index: index of the edges.
        :param torch.Tensor x: nodes features.
        :param torch.Tensor u: values of the PDE field in the nodes.
        :param torch.Tensor pos: cartesian coordinates of the nodes.
        :param torch.Tensor variables: additional attributes of the nodes.
        :return: updated features tensor.
        :rtype: torch.Tensor
        """
        f = self.propagate(edge_index=edge_index, x=x, 
                           u=u, pos=pos, variables=variables)
        f= self.norm(f, batch)
        
        return f
    

    def message(self, x_i, x_j, u_i, u_j, pos_i, pos_j, variables_i):
        """
        Definition of the message between node i (recipient) and j (sender).

        :param torch.Tensor x_i: tensor of features at node i.
        :param torch.Tensor x_j: tensor of features at node j.
        :param torch.Tensor u_i: values of the PDE field at node i.
        :param torch.Tensor u_j: values of the PDE field at node j.
        :param torch.Tensor pos_i: cartesian coordinates of node i.
        :param torch.Tensor pos_j: cartesian coordinates of node j.
        :param torch.Tensor variables_i: additional attributes at node i.
        :return: updated message.
        :rtype: torch.Tensor
        """
        message = self.message_net_1(torch.cat((x_i, x_j, u_i-u_j, pos_i-pos_j, variables_i), dim=-1))
        message = self.message_net_2(message)
        
        return message
    

    def update(self, message, x, variables):
        """
        Update of the node features.

        :param torch.Tensor message: tensor of messages.
        :param torch.Tensor x: tensor of features.
        :param torch.Tensor variables: tensor of additional attributes.
        :return: updated features.
        :rtype: torch.Tensor
        """
        update = self.update_net_1(torch.cat((x, message, variables), dim=-1))
        update = self.update_net_2(update)
        
        if self.in_features == self.out_features:
            return x + update
        else:
            return update