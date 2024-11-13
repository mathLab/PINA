from .supervised import SupervisedSolver
from ..graph import Graph


class GraphSupervisedSolver(SupervisedSolver):

    def __init__(
        self,
        problem,
        model,
        loss=None,
        optimizer=None,
        scheduler=None,
        use_lt=True,):
        super().__init__(problem, model, loss, optimizer, scheduler, use_lt=use_lt)

    def forward(self, batch):
        return self._model(batch.x, batch.edge_index, batch.batch)
