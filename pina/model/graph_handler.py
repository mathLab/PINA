from torch_geometric.data import Data
from torch_cluster import radius_graph


class GraphHandler():
    """
    Creates and manages a graph with following attrubutes:
    - graph.u: values of u(x,t) at point x and time t in considered window
    - graph.pos: spatial coordinates
    - graph.variables: variables of the equation (time and parameters)
    - graph.x: node features
    """
    def __init__(self, dt, num_neighs=10):
        super().__init__()
        self.num_neighs = num_neighs
        self.dt = dt
        self.graph = None


    def create_ball_graph(self, coordinates, data, variables, batch):
        dx = coordinates[1] - coordinates[0]
        radius = self.num_neighs * dx + 0.000001
        edge_index = radius_graph(coordinates, r=radius, loop=False, batch=batch)
        
        graph = Data(x = data, edge_index = edge_index, edge_attr = None)
        graph.pos = coordinates.unsqueeze(-1)
        graph.variables = variables
        graph.dt = self.dt
        graph.batch = batch
        return graph