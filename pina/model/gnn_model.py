import torch
from gnn_layer import GNN_Layer
from pina import LabelTensor

class GNN(torch.nn.Module):

    def __init__(self, time_window, t_max, n_variables, embedding_dimension=128, processing_layers=6):
        super().__init__()
        self.embedding_dimension = embedding_dimension
        self.processing_layers = processing_layers
        self.time_window = time_window
        self.n_variables = n_variables
        self.t_max = t_max

        self.encoder = torch.nn.Sequential(torch.nn.Linear(self.time_window + self.n_variables + 1, self.embedding_dimension),
                                           torch.nn.SiLU(),
                                           torch.nn.Linear(self.embedding_dimension, self.embedding_dimension),
                                           torch.nn.SiLU())
        
        self.gnn_layers = torch.nn.ModuleList(modules=(GNN_Layer(in_features=self.embedding_dimension,
                                                                 hidden_features=self.embedding_dimension,
                                                                 out_features=self.embedding_dimension,
                                                                 time_window=self.time_window,
                                                                 n_variables=self.n_variables) for _ in range(self.processing_layers)))
        
        self.decoder = torch.nn.Sequential(torch.nn.Conv1d(in_channels=1, out_channels=8, kernel_size=16, stride=3),
                                           torch.nn.SiLU(),
                                           torch.nn.Conv1d(in_channels=8, out_channels=1, kernel_size=14, stride=1))
        

    def forward(self, data, pos, time, variables, batch, edge_index, dt):        
        pos = pos/pos.max()
        time = time/self.t_max
        var = torch.cat((time, variables), dim=-1)
        node_input = torch.cat((data, pos, var), dim=-1)
        h = self.encoder(node_input)
        for i in range(self.processing_layers):
            h = self.gnn_layers[i](edge_index, h, data, pos, var, batch)
        dt = torch.cumsum((torch.ones(1, self.time_window)*dt).to(device=h.device), dim=1)
        diff = self.decoder(h[:, None]).squeeze(1)
        out = data[:,-1].repeat(1, self.time_window) + dt*diff
        return out