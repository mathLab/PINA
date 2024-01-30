import torch
from pina.model.layers.gnn_layer import GNN_Layer

class GNN(torch.nn.Module):
    """
    Message passing model class.
    """
    def __init__(self, 
                 time_window: int,
                 n_variables: int,
                 t_max: float,
                 embedding_dimension: int = 128, 
                 processing_layers: int = 6):
        """
        Initialize Message Passing model class.
        Args:
            time_window: temporal bundling parameter
            n_variables: number of paramaters of the PDE
            output_dimension: dimension of the output
            embedding_dimension: dimension of node features
            processing_layers: number of message passing layers
        """
        
        super().__init__()
        self.output_dimension = time_window
        self.embedding_dimension = embedding_dimension
        self.processing_layers = processing_layers
        self.time_window = time_window
        self.n_variables = n_variables
        self.t_max = t_max
        
        # Encoder
        # TODO: the user should be able to define as many layers as wanted
        self.encoder = torch.nn.Sequential(torch.nn.Linear(self.time_window + self.n_variables +1, self.embedding_dimension), torch.nn.SiLU(),
                                           torch.nn.Linear(self.embedding_dimension, self.embedding_dimension), torch.nn.SiLU())
        
        # GNN layers
        self.gnn_layers = torch.nn.ModuleList(modules=(GNN_Layer(in_features=self.embedding_dimension, 
                                                                 hidden_features=self.embedding_dimension,
                                                                 out_features=self.embedding_dimension,
                                                                 time_window=self.time_window,
                                                                 n_variables=self.n_variables) for _ in range(self.processing_layers)))
        
        # Decoder
        # TODO: use a linear layer after convolutions to allow easier management of strides and kernel sizes.
        # However, it is not clean nor always correct: for self.embedding_dimension < 55, it is meaningless.
        # self.decoder = torch.nn.Sequential(torch.nn.Conv1d(in_channels=1, out_channels=8, kernel_size=15, stride=4),
                                           #torch.nn.SiLU(), 
                                           #torch.nn.Conv1d(in_channels=8, out_channels=1, kernel_size=10, stride=1),
                                           #torch.nn.SiLU(),
                                           #torch.nn.Linear(in_features= (int((self.embedding_dimension-15)/4)-8), out_features=self.output_dimension))
        
        # At the moment we use a fixed time_window = 25                                   
        self.decoder = torch.nn.Sequential(torch.nn.Conv1d(in_channels=1, out_channels=8, kernel_size=16, stride=3),
                                           torch.nn.SiLU(), 
                                           torch.nn.Conv1d(in_channels=8, out_channels=1, kernel_size=14, stride=1))

    def forward(self, graph):
        
        #Normalization
        vars = graph.variables.extract(['alpha', 'beta', 'gamma'])
        time = graph.variables.extract(['t'])/self.t_max
        graph.pos = graph.pos/graph.pos.max()

        # Encoder
        input = torch.cat((graph.x, graph.pos, time, vars), dim = -1)
        h = self.encoder(input)
        
        # Processor
        for i in range(self.processing_layers):
            h = self.gnn_layers[i](h, graph.x, graph.pos, graph.variables, graph.edge_index, graph.batch)

        # Decoder -- controllare che funzioni dt
        dt = (torch.ones(1, self.time_window)*graph.dt).to(h.device)
        dt = torch.cumsum(dt, dim=1)
        diff = self.decoder(h[:, None]).squeeze(1)
        out = graph.x[:, -1].repeat(1, self.time_window) + dt*diff
        
        return out