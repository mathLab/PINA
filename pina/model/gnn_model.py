import torch
from torch.nn import Sequential, Linear, SiLU, ModuleList, Conv1d
from gnn_layer import GNN_Layer

class GNN(torch.nn.Module):
    """
    Implementation of a message-passing graph neural network.
    """
    
    def __init__(self, 
                 time_window, 
                 t_max, 
                 n_variables, 
                 n_spatial=1, 
                 embedding_dim=164, 
                 processing_layers=6):
        """
        Initialization.
        
        :param int time_window: width of the considered time interval.
        :param int t_max: upper extreme of the considered time interval.
        :param int n_variables: number of variables (including time).
        :param int spatial dimension of the considered problem.
            Default: 1.
        :param int embedding_dim: dimension of the embedding space.
            Default: 164.
        :param int processing_layers: number of message-passing layers.
            Default: 6.
        """
        
        super().__init__()
        self.time_window = time_window
        self.t_max = t_max
        self.n_variables = n_variables
        self.n_spatial = n_spatial
        self.embedding_dim = embedding_dim
        self.processing_layers = processing_layers
        
        # Encoder
        self.encoder = Sequential(Linear(self.time_window + self.n_variables + self.n_spatial, self.embedding_dim), 
                                  SiLU(), 
                                  Linear(self.embedding_dim, self.embedding_dim), 
                                  SiLU())
        
        # Processor
        self.gnn_layers = ModuleList(modules=(GNN_Layer(in_features=self.embedding_dim,
                                                        hidden_features=self.embedding_dim,
                                                        out_features=self.embedding_dim,
                                                        time_window=self.time_window,
                                                        n_variables=self.n_variables,
                                                        n_spatial=self.n_spatial) for _ in range(self.processing_layers)))
        
        # Decoder
        self.decoder = Sequential(Conv1d(in_channels=1, out_channels=8, kernel_size=16, stride=3),
                                  SiLU(),
                                  Conv1d(in_channels=8, out_channels=1, kernel_size=14, stride=9))
        

    def forward(self, graph):
        """
        Trigger of the encoder-processor-decoder routine.

        :param Data graph: graph to be used for message passing.
        :return: updated features tensor.
        :rtype: torch.Tensor
        """
        
        # Extraction
        data = graph.x
        pos = graph.pos
        time = graph.time
        variables = graph.variables
        batch = graph.batch
        edge_index = graph.edge_index
        dt = graph.dt
        
        # Normalize pos and time     
        pos = pos/pos.max()
        time = time/self.t_max

        # Input of the encoder
        var = torch.cat((time, variables), dim=-1)
        node_input = torch.cat((data, pos, var), dim=-1)
        
        # Encoder
        h = self.encoder(node_input)
        
        # Processor
        for i in range(self.processing_layers):
            h = self.gnn_layers[i](edge_index, h, data, pos, var, batch)
        
        # Decoder
        dt = torch.cumsum((torch.ones(1, self.time_window)*dt).to(device=h.device), dim=1)
        diff = self.decoder(h[:, None]).squeeze(1)
        out = data[:,-1].repeat(1, self.time_window) + dt*diff
        
        return out