import torch

from pina.geometry import CartesianDomain
from .location import Location
from pina import LabelTensor
from ..utils import check_consistency

class SimplexDomain(Location):
    """PINA implementation of a Simplex."""

    def __init__(self, simplex_dict, labels, sample_surface=False):
        """
        :param simplex_dict: A dictionary with dict-key a string representing
            the input variables for the pinn, and dict-value a list
            representing vertices of the simplex.
        :type simplex_dict: dict
        :param labels: A list of labels for vertex components
        :type labels: list[str]
        :param sample_surface: A variable for choosing sample strategies. If
            `sample_surface=True` only samples on the Simplex surface
            frontier are taken. If `sample_surface=False`, no such criteria
            is followed.
        :type sample_surface: bool
        
        :Example:

            >>> spatial_domain = SimplexDomain({'vertex1': [0, 0], 'vertex2': [1, 1], 'vertex3': [0, 2]}, ['x', 'y'])
        """

        # check consistency of labels
        if not isinstance(labels, list):
            raise ValueError(f"{type(labels).__name__} must be {list}.")
        for lab in labels:
            check_consistency(lab, str)
        self._coordinate_labels = labels
        
        # check consistency of sample_surface
        check_consistency(sample_surface, bool)
        self._sample_surface = sample_surface

        # check consistency of simplex_dict
        check_consistency(simplex_dict, dict)
        for vertex in simplex_dict.values():
            if not isinstance(vertex, list):
                raise ValueError(f"{type(vertex).__name__} must be {list}.")
        

        # vertices, vectors, dimension 
        self._vertices = {label: LabelTensor(torch.tensor([vertex]), self.variables) for label, vertex in simplex_dict.items()}
        self._vectors = self._basis_vectors(self._vertices_list)
        self._dimension = len(simplex_dict)-1


        # build cartesian_bound
        self._cartesian_bound = self._build_cartesian(list(simplex_dict.values()), labels)


    @property
    def variables(self):
        """
        Coordinate labels of simplex.

        :return: Coordinate labels
        :rtype: list[str]
        """

        return self._coordinate_labels


    @property
    def vertices(self):
        """
        Vertices of simplex.

        :return: Vectors
        :rtype: list[list]
        """

        return self._vertices
    

    @property
    def vectors(self):
        """
        Vectors.

        :return: Vectors
        :rtype: dict(LabelTensor)
        """
        return self._vectors
    

    @property
    def _vertices_list(self):
        """
        List of vectors.

        :return: List of vectors
        :rtype: list[LabelTensor]
        """
        return list(self.vertices.values())
    

    @property
    def cartesian_bound(self):
        """
        Cartesian border for Simplex domain.

        :return: Cartesian border for Simplex domain
        :rtype: CartesianDomain
        """

        return self._cartesian_bound
    

    @property
    def dimension(self):
        """
        Dimension of Simplex domain.
        
        :return: dimension
        :rtype: int
        """

        return self._dimension
    

    @property
    def sample_surface(self):
        """
        Whether the surface should be sampled or not.

        :return: Whether the surface should be sampled or not
        :rtype: bool
        """

        return self._sample_surface
    

    def _basis_vectors(self, vertices):
        """
        Basis vectors for simplex.

        :return: Basis vectors
        :rtype: dict(LabelTensor)
        """
        
        vectors = {}
        origin = vertices[0]
        num_vertices = len(self.vertices)

        for i in range(1, num_vertices):
            vectors[f'vector{i}'] = LabelTensor(torch.subtract(vertices[i], origin), self.variables)

        return vectors


    def _build_cartesian(self, vertices, labels):
        """
        Build Cartesian border for Simplex domain to be used in sampling.

        :param vertices: list of Simplex domain's vertices
        :type vertices: list[list]
        :return: Cartesian border for triangular domain
        :rtype: CartesianDomain
        """

        span_dict = {}

        for i, coord in enumerate(labels):
            sorted_vertices = sorted(vertices, key=lambda vertex: vertex[i])

            # respective coord bounded by the lowest and highest values
            span_dict[coord] = [sorted_vertices[0][i], sorted_vertices[-1][i]]

        return CartesianDomain(span_dict)
    

    def _volume(self, vectors):
        """
        Volume of Simplex spanned by vectors. Uses the determinant of the Grammian matrix
        to calculate the volume. See link below for formula.
        Formula: https://en.wikipedia.org/wiki/Simplex#Geometric_properties

        :param vectors: list of vectors
        :type vectors: list(LabelTensor)
        :return: Returns volume of Simplex spanned by vectors
        :rtype: float
        """

        gram_matrix = torch.matmul(torch.transpose(vectors, 0, -1), vectors).type(torch.FloatTensor)
        sqrt_det = torch.sqrt(torch.det(gram_matrix))
        
        return float(1/torch.jit._builtins.math.factorial(len(vectors[0])) * sqrt_det)
    

    def _on_border(self, point):
        """
        Whether a point is on Simplex domain border or not.

        :param point: a point
        :type point: LabelTensor
        :return: Whether a point is on border or not
        :rtype: bool
        """

        def _normalize_and_label(vector, labels):
            """
            Normalize vector and label.

            :param vector: a vector
            :type vector: LabelTensor
            :param labels: a list of labels
            :type labels: list[str]
            :return: Normalized vector with labels
            :rtype: LabelTensor
            """

            vector = vector.type(torch.FloatTensor)
            normalized_vector = torch.divide(vector, torch.linalg.norm(vector))
            normalized_vector.labels = labels

            return normalized_vector
        
        vectors_list = list(self._vectors.values()) + [LabelTensor(torch.subtract(self._vertices_list[self.dimension], self._vertices_list[self.dimension-1]), self.variables)]

        normalized_vectors_to_point, normalized_simplex_vectors = [], []
        for vertex, vector in zip(self._vertices_list, vectors_list):
            normalized_vectors_to_point.append(_normalize_and_label(torch.subtract(point, vertex), point.labels))
            normalized_simplex_vectors.append(_normalize_and_label(vector, point.labels))

        for vector1 in normalized_vectors_to_point:
            for vector2 in normalized_simplex_vectors:
                v1 = [vector1.extract(label) for label in vector1.labels]
                v2 = [vector2.extract(label) for label in vector2.labels]

                if v1 == v2:
                    return True
        
        return False


    def is_inside(self, point, check_border=False):
        """
        Check if a point is inside the simplex.

        :param point: Point to be checked
        :type point: LabelTensor
        :param check_border: Check if the point is also on the frontier
            of the simplex, default False.
        :type check_border: bool
        :return: Returning True if the point is inside, False otherwise.
        :rtype: bool
        """

        if not all([label in self.variables for label in point.labels]):
            raise ValueError('Point labels different from constructor'
                             f' dictionary labels. Got {point.labels},'
                             f' expected {self.variables}.')
        
        if not check_border and self._on_border(point):
            return False
            
        if check_border and not self._on_border(point):
            return False
        
        vector_to_point = torch.subtract(point, self._vertices_list[0])
        matrix = list(self.vectors.values())
        sum_of_sub_volumes = 0
        tot_volume = self._volume(torch.cat(matrix, dim=0))

        for i in range(self.dimension):
            vectors = matrix[:i] + [vector_to_point] + matrix[i+1:]
            subvolume = self._volume(torch.cat(vectors, dim=0))

            if not check_border and subvolume == 0:
                return False
            
            sum_of_sub_volumes += subvolume
            if sum_of_sub_volumes > tot_volume:
                return False
        
        return tot_volume >= sum_of_sub_volumes


    def sample(self, n, mode="random", variables="all"):
        """
        Sample n points from Simplex domain.

        :param n: Number of points to sample in the shape.
        :type n: int
        :param mode: Mode for sampling, defaults to 'random'.
            Available modes include: 'random'.
        :type mode: str, optional
        :param variables: pinn variable to be sampled, defaults to 'all'.
        :type variables: str or list[str], optional
        :return: Returns LabelTensor of n sampled points
        :rtype: LabelTensor(tensor)
        """

        if mode != "random":
            raise ValueError("Mode can only be random")

        # Sample points on the domain
        sampled_points = []
        for _ in range(n):
            sampled_point = self.cartesian_bound.sample(
                n=1, mode="random", variables=variables
            )

            #check = self._on_border if self.sample_surface else self.is_inside

            # Keep sampling until you get a point that is inside
            while not self.is_inside(sampled_point, self.sample_surface):
                sampled_point = self.cartesian_bound.sample(
                    n=1, mode=mode, variables=variables
                )
                
            sampled_points.append(sampled_point)
        
        return LabelTensor(torch.cat(sampled_points, dim=0), labels=self.variables)