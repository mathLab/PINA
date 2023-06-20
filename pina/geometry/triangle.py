import torch
import numpy as np

from pina.geometry import CartesianDomain
from .location import Location


class TriangleDomain(Location):
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

        if len(span_dict) <= 2:
            raise ValueError("Too few vertices for a triangular domain")

        if len(span_dict) == 3:  # 2D
            self.dimension = 2

            if not all(len(vertex) == 2 for vertex in span_dict.values()):
                raise ValueError("Unsupported dimensions")

        if len(span_dict) == 4:  # 3D
            self.dimension = 3

            if not all(len(vertex) == 3 for vertex in span_dict.values()):
                raise ValueError("Unsupported dimensions")

        if len(span_dict) >= 5:  # Won't handle greater than 3 dimensions
            raise ValueError("Too many vertices for a triangular domain")

        self.fixed_, self.range_ = {}, {}

        for k, v in span_dict.items():
            if isinstance(v, (int, float)):
                self.fixed_[k] = v
            elif (
                isinstance(v, (list, tuple)) and len(v) >= 2
            ):  # length might be able to be > 2
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
        :rtype: tuple[list]
        """

        return tuple(self.fixed_.values()) + tuple(self.range_.values())

    @property
    def vectors(self):
        """
        Vertices of triangular domain.

        :return: Vertices
        :rtype: list[tuple]
        """

        return [
            [self.vertices[i][j] - self.vertices[0][j] for j in range(self.dimension)]
            for i in range(1, self.dimension + 1)
        ]

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

        def _area(vertices):
            """
            Given vertices of triangle, calculates area

            :param vertices: Vertices of triangle
            :type vertices: tuple
            :return Returns area of triangle
            :rtype: float
            """
            vertex1, vertex2, vertex3 = vertices

            return (
                abs(
                    (
                        vertex1[0] * (vertex2[1] - vertex3[1])
                        + vertex2[0] * (vertex3[1] - vertex1[1])
                        + vertex3[0] * (vertex1[1] - vertex2[1])
                    )
                )
                / 2.0
            )

        def _volume(vertices):
            """
            Given vertices of tetrahedron, calculates volume

            :param vertices: Vertices of tetrahedron
            :type vertices: tuple
            :return Returns area of tetrahedron
            :rtype: float
            """
            vertex1, vertex2, vertex3, vertex4 = vertices

            return (
                abs(
                    np.dot(
                        (np.array(vertex1) - np.array(vertex4)),
                        np.cross(
                            (np.array(vertex2) - np.array(vertex4)),
                            (np.array(vertex3) - np.array(vertex4)),
                        ),
                    )
                )
                / 6.0
            )

        pt = [point.extract([variable]) for variable in point.labels]

        if not check_border:
            if pt in self.vertices:
                return False

            vec = np.array(pt) - np.array(self.vertices[0])
            vector_to_point = list(vec / np.linalg.norm(vec))

            for vector in self.vectors:
                if list(np.array(vector) / np.linalg.norm(vector)) == vector_to_point:
                    return False

        if self.dimension == 2:
            vertex1, vertex2, vertex3 = self.vertices

            # subareas
            area1 = _area([pt, vertex2, vertex3])
            area2 = _area([vertex1, pt, vertex3])
            area3 = _area([vertex1, vertex2, pt])

            return _area(self.vertices) == area1 + area2 + area3

        # otherwise, dimension is 3
        else:
            vertex1, vertex2, vertex3, vertex4 = self.vertices

            # subvolumes
            volume1 = _volume([pt, vertex2, vertex3, vertex4])
            volume2 = _volume([vertex1, pt, vertex3, vertex4])
            volume3 = _volume([vertex1, vertex2, pt, vertex4])
            volume4 = _volume([vertex1, vertex2, vertex3, pt])

            return _volume(self.vertices) == volume1 + volume2 + volume3 + volume4

    def sample(self, n, mode="random", variables="all"):
        if mode != "random":
            raise ValueError("Mode can only be random")

        # for 2D

        # Construct CartesianDomain that contains triangle
        vertices_by_x = sorted(self.vertices, lambda vertex: vertex[0])
        vertices_by_y = sorted(self.vertices, lambda vertex: vertex[1])

        if self.dimension == 3:
            vertices_by_z = sorted(self.vertices, lambda vertex: vertex[2])

            circumscribing_domain = CartesianDomain(
                {
                    "x": [vertices_by_x[0], vertices_by_x[-1]],
                    "y": [vertices_by_y[0], vertices_by_y[-1]],
                    "z": [vertices_by_z[0], vertices_by_z[-1]],
                }
            )

        else:
            circumscribing_domain = CartesianDomain(
                {
                    "x": [vertices_by_x[0], vertices_by_x[-1]],
                    "y": [vertices_by_y[0], vertices_by_y[-1]],
                }
            )

        # Sample points on the domain
        sampled_points = []
        for _ in range(n):
            sampled_point = circumscribing_domain.sample(
                n=1, mode="random", variables=variables
            )

            # Keep sampling until you get a point that is inside
            while not self.is_inside(sampled_point):
                sampled_point = circumscribing_domain.sample(
                    n=1, mode="random", variables=variables
                )

            sampled_points.append(sampled_point)

        return sampled_points
