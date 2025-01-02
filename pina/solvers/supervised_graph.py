import torch
from . import SupervisedSolver

class GraphSupervisedSolver(SupervisedSolver):
    def __init__(self,
                 problem,
                 model,
                 loss=None,
                 optimizer=None,
                 scheduler=None,
                ):
        super().__init__(problem, model, loss, optimizer, scheduler, None, False)

    def forward(self, batch):
        x = batch.x
        edge_index = batch.edge_index
        edge_attr = batch.edge_attr
        return self._model(x, edge_index, edge_attr)
    
    def loss_data(self, input_pts, output_pts):
        if isinstance(output_pts, torch.Tensor):
            output_pts = output_pts.reshape(-1, * output_pts.shape[2:])
        return super().loss_data(input_pts, output_pts)