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