"""Module for Location class."""
import torch
from .location import Location
from ..utils import check_consistency
from ..label_tensor import LabelTensor


class Difference(Location):
    """
    """

    def __init__(self, geometries):

        super().__init__()

        # difference checks
        check_consistency(geometries, Location)
        self._check_difference_dimesions(geometries)

        # assign geometries
        self._geometries = geometries

    @property
    def geometries(self):
        """ 
        The geometries."""
        return self._geometries

    @property
    def variables(self):
        """
        Spatial variables.

        :return: All the spatial variables defined in '__init__()' in order.
        :rtype: list[str]
        """
        all_variables = []
        seen_variables = set()
        for geometry in self.geometries:
            for variable in geometry.variables:
                if variable not in seen_variables:
                    all_variables.append(variable)
                    seen_variables.add(variable)
        return all_variables

    def is_inside(self, point, check_border=False):
        """Check if a point is inside the difference domain.

        :param point: Point to be checked.
        :type point: torch.Tensor   
        :param bool check_border: If True, the border is considered inside.
        :return: True if the point is inside the union domain, False otherwise.
        :rtype: bool
        """
        flag = 0
        for geometry in self.geometries:
            if geometry.is_inside(point, check_border):
                flag += 1
        return flag == 1

    def sample(self, n, mode='random', variables='all'):
        sampled_points = []
        remainder = n % len(self.geometries)
        num_points = n // len(self.geometries)

        for i, geometry in enumerate(self.geometries):
            if i < remainder:
                num_points += 1
            points = geometry.sample(num_points, mode, variables)

            for point in points:
                point.labels = [f'{i}' for i in self.variables]
                if self.is_inside(point):
                    sampled_points.append(point)

        return LabelTensor(torch.cat(sampled_points), labels=[f'{i}' for i in self.variables])

    def og_sample(self, n, mode='random', variables='all'):
        """
        """
        # assert mode is 'random', 'Only random mode is implemented'

        samples = []
        while len(samples) < n:
            sample = self.first.sample(1, 'random')
            if not self.second.is_inside(sample):
                samples.append(sample.tolist()[0])

        import torch
        return LabelTensor(torch.tensor(samples), labels=['x', 'y'])

    def _check_difference_dimesions(self, geometries):
        """Check if the dimensions of the geometries are consistent.

        :param geometries: Geometries to be checked.
        :type geometries: list[Location]
        """
        for geometry in geometries:
            if geometry.variables != geometries[0].variables:
                raise NotImplementedError(
                    f'The geometries need to be the same dimensions. {geometry.variables} is not equal to {geometries[0].variables}')
