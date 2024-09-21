from .supervised import SupervisedSolver
from ..graph import Graph


class GraphSupervisedSolver(SupervisedSolver):

    def __init__(
        self,
        problem,
        model,
        nodes_coordinates,
        nodes_data,
        loss=None,
        optimizer=None,
        scheduler=None):
        super().__init__(problem, model, loss, optimizer, scheduler)
        if isinstance(nodes_coordinates, str):
            self._nodes_coordinates = [nodes_coordinates]
        else:
            self._nodes_coordinates = nodes_coordinates
        if isinstance(nodes_data, str):
            self._nodes_data = nodes_data
        else:
            self._nodes_data = nodes_data

    def forward(self, input):
        input_coords = input.extract(self._nodes_coordinates)
        input_data = input.extract(self._nodes_data)

        if not isinstance(input, Graph):
            input = Graph.build('radius', nodes_coordinates=input_coords, nodes_data=input_data, radius=0.2)
        g = self.model(input.data, edge_index=input.data.edge_index)
        g.labels = {1: {'name': 'output', 'dof': ['u']}}
        return g
