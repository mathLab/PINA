import torch
import numpy as np
import math

#from .location import Location
#from ..label_tensor import LabelTensor
#from ..utils import torch_lhs, chebyshev_roots


class TriangularDomain:#(Location):
    """PINA implementation of a Triangle."""

    def __init__(self, span_dict):
        """
        :param span_dict: A dictionary with dict-key a string representing
            the input variables for the pinn, and dict-value a list
            representing vertices of triangle.
        :type span_dict: dict

        :Example:
            >>> spatial_domain = TriangularDomain({'vertex1': [0, 0], 'vertex2': [1, 1], 'vertex3': [0, 2]})
        """

        self.fixed_, self.range_ = {}, {}

        for k, v in span_dict.items():
            if isinstance(v, (int, float)):
                self.fixed_[k] = v
            elif isinstance(v, (list, tuple)) and len(v) >= 2: # length might be able to be > 2
                self.range_[k] = v
            else:
                raise TypeError

    @property
    def variables(self):
        """
        Spatial variables.

        :return: Spatial variables defined in '__init__()'
        :rtype: list[str]
        """

        return list(self.fixed_.keys()) + list(self.range_.keys())
    
    @property
    def vertices(self):
        """
        Vertices of triangle.

        :return: Vectors defined in '__init__()'
        :rtype: list[list]
        """

        return list(self.fixed_.values()) + list(self.range_.values())
    
    @property
    def vectors(self):
        """
        Vertices of triangular domain.

        :return: Vertices
        :rtype: list[tuple]
        """

        vertices = self.vertices
            
        return [[vertices[i+1][j]-vertices[i][j] for j in range(len(vertices[0]))] for i in range(len(vertices) - 1)] + [[vertices[-1][j]-vertices[0][j] for j in range(len(vertices[0]))]]
    
    def is_inside(self, point, check_border=False):
        """
        Check if a point is inside the triangle.

        :param point: Point to be checked
        :type point: LabelTensor
        :param check_border: Check if the point is also on the frontier
            of the triangle, default False.
        :type check_border: bool
        :return: Returning True if the point is inside, False otherwise.
        :rtype: bool
        """

        def _gramian(vector1, vector2):
            """
            Computes Gramian from vectors for sake of finding area.

            :param vector1: a vector
            :type vector1: a list
            :param vector2: a vector
            :type vector2: a list
            :return: Returns the Gramian of two vectors
            :rtype: np matrix
            """

            x = [vector1, vector2]

            return np.matmul(x, np.transpose(x))
        
        def _afg(gramian):
            """
            Area from Gramian.
            
            :param gramian: a Gramian matrix
            :type gramian: np matrix
            :return: Returns the area derived from Gramian
            :rtype: float
            """

            return math.sqrt(abs(np.exp(np.linalg.slogdet(gramian)[1]))) / 2
        
        vertices, vectors = self.vertices, self.vectors
        vectors_to_point = [[point[i] - vertex[i] for i in range(len(vectors[0]))] for vertex in vertices]
        barycentric_coords = [_afg(_gramian([vector, vector_to_point])) for vector, vector_to_point in zip(vectors, vectors_to_point)]
        
        if not check_border and 0 in barycentric_coords:
            return False

        return sum(barycentric_coords) - _afg(_gramian(self.vectors)) <= 1e-6