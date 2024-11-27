""" Module for Loss class """

import logging
from torch_geometric.nn import MessagePassing, InstanceNorm, radius_graph
from torch_geometric.data import Data
import torch

class Graph:
    """
    PINA Graph managing the PyG Data class.
    """
    def __init__(self, data):
        self.data = data
            
    @staticmethod
    def _build_triangulation(**kwargs):
        logging.debug("Creating graph with triangulation mode.")

        # check for mandatory arguments
        if "nodes_coordinates" not in kwargs:
            raise ValueError("Nodes coordinates must be provided in the kwargs.")
        if "nodes_data" not in kwargs:
            raise ValueError("Nodes data must be provided in the kwargs.")
        if "triangles" not in kwargs:
            raise ValueError("Triangles must be provided in the kwargs.")

        nodes_coordinates = kwargs["nodes_coordinates"]
        nodes_data = kwargs["nodes_data"]
        triangles = kwargs["triangles"]



        def less_first(a, b):
            return [a, b] if a < b else [b, a]

        list_of_edges = []

        for triangle in triangles:
            for e1, e2 in [[0, 1], [1, 2], [2, 0]]:
                list_of_edges.append(less_first(triangle[e1],triangle[e2]))

        array_of_edges = torch.unique(torch.Tensor(list_of_edges), dim=0) # remove duplicates
        array_of_edges = array_of_edges.t().contiguous()
        print(array_of_edges)

        # list_of_lengths = []

        # for p1,p2 in array_of_edges:
        #     x1, y1 = tri.points[p1]
        #     x2, y2 = tri.points[p2]
        #     list_of_lengths.append((x1-x2)**2 + (y1-y2)**2)

        # array_of_lengths = np.sqrt(np.array(list_of_lengths))

        # return array_of_edges, array_of_lengths

        return Data(
            x=nodes_data,
            pos=nodes_coordinates.T,
            
            edge_index=array_of_edges,
        )

    @staticmethod
    def _build_radius(**kwargs):
        logging.debug("Creating graph with radius mode.")

        # check for mandatory arguments
        if "nodes_coordinates" not in kwargs:
            raise ValueError("Nodes coordinates must be provided in the kwargs.")
        if "nodes_data" not in kwargs:
            raise ValueError("Nodes data must be provided in the kwargs.")
        if "radius" not in kwargs:
            raise ValueError("Radius must be provided in the kwargs.")

        nodes_coordinates = kwargs["nodes_coordinates"]
        nodes_data = kwargs["nodes_data"]
        radius = kwargs["radius"]

        edges_data = kwargs.get("edge_data", None) 
        loop = kwargs.get("loop", False)
        batch = kwargs.get("batch", None)

        logging.debug(f"radius: {radius}, loop: {loop}, "
                        f"batch: {batch}")

        edge_index = radius_graph(
            x=nodes_coordinates.tensor,
            r=radius,
            loop=loop,
            batch=batch,
        )

        logging.debug(f"edge_index computed")
        return Data(
            x=nodes_data.tensor,
            pos=nodes_coordinates.tensor,
            edge_index=edge_index,
            edge_attr=edges_data,
        )

    @staticmethod
    def build(mode, **kwargs):
        """
        Constructor for the `Graph` class.
        """
        if mode == "radius":
            graph = Graph._build_radius(**kwargs)
        elif mode == "triangulation":
            graph = Graph._build_triangulation(**kwargs)
        else:
            raise ValueError(f"Mode {mode} not recognized")
        
        return Graph(graph)


    def __repr__(self):
        return f"Graph(data={self.data})"