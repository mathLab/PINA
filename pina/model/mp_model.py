import torch
import torch.nn as nn
from layers.gnn_layer import GNN_Layer

class Model(torch.nn.Module):
    """
    Message passing model class.
    """
    def __init__(self,
                 handler, 
                 time_window: int,
                 n_variables: int,
                 input_dimension: int, 
                 output_dimension: int, 
                 embedding_dimension: int = 128, 
                 processing_layers: int = 6):
        """
        Initialize Message Passing model class.
        Args:
            handler: GraphHandler object to manage the graph
            time_window: temporal bundling parameter
            n_variables: number of paramaters of the PDE
            input_dimension: dimension of the input
            output_dimension: dimension of the output
            embedding_dimension: dimension of node features
            processing_layers: number of message passing layers
        """
        
        super().__init__()
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        self.embedding_dimension = embedding_dimension
        self.processing_layers = processing_layers
        self.handler = handler
        self.time_window = time_window
        self.n_variables = n_variables
        
        # Encoder
        self.encoder = nn.Sequential(nn.Linear(self.input_dimension, self.embedding_dimension), nn.SiLU(),
                                     nn.Linear(self.embedding_dimension, self.embedding_dimension), nn.SiLU())
        
        # GNN layers
        self.gnn_layers = torch.nn.ModuleList(modules=(GNN_Layer(in_features=self.embedding_dimension, 
                                                                 hidden_features=self.embedding_dimension,
                                                                 out_features=self.embedding_dimension,
                                                                 time_window=self.time_window,
                                                                 n_variables=self.n_variables) for _ in range(self.processing_layers)))
        
        # Decoder
        # TODO: to be transformed in a 1d-CNN
        self.decoder = torch.nn.Linear(self.embedding_dimension, self.output_dimension)
        # parameters to be set in a correct way
        # self.decoder = nn.Sequential(nn.Conv1d(in_channels=1, out_channels=8, kernel_size=15, stride=4),
                                        # nn.Swish(),
                                        # nn.Conv1d(in_channels=8, out_channels=1, kernel_size=10, stride=1))

    def forward(self, x):
        # Insert graph.u data for message passing
        self.handler.data_to_graph(x.extract(['k']))
        graph = self.handler.graph

        # Encoder
        input = torch.cat((graph.u, graph.pos, graph.variables), dim = -1)
        graph.x = self.encoder(input)

        # Processor
        for i in range(self.processing_layers):
            h = self.gnn_layers[i](graph)
            graph.x = h

        # Decoder
        out = self.decoder(graph.x)
        return out