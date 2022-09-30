from .chebyshev import chebyshev_roots
import torch

from .location import Location
from .label_tensor import LabelTensor


class Span(Location):
    def __init__(self, span_dict):

        self.fixed_ = {}
        self.range_ = {}

        for k, v in span_dict.items():
            if isinstance(v, (int, float)):
                self.fixed_[k] = v
            elif isinstance(v, (list, tuple)) and len(v) == 2:
                self.range_[k] = v
            else:
                raise TypeError

    @property
    def variables(self):
        return list(self.fixed_.keys()) + list(self.range_.keys())

    def update(self, new_span):
        self.fixed_.update(new_span.fixed_)
        self.range_.update(new_span.range_)

    def _sample_range(self, n, mode, bounds):
        """
        """
        if mode == 'random':
            pts = torch.rand(size=(n, 1))
        elif mode == 'chebyshev':
            pts = chebyshev_roots(n).mul(.5).add(.5).reshape(-1, 1)
        elif mode == 'grid':
            pts = torch.linspace(0, 1, n).reshape(-1, 1)
        elif mode == 'lh' or mode == 'latin':
            from scipy.stats import qmc
            sampler = qmc.LatinHypercube(d=1)
            pts = sampler.random(n)
            pts = torch.from_numpy(pts)

        pts *= bounds[1] - bounds[0]
        pts += bounds[0]

        return pts

    def sample(self, n, mode='random', variables='all'):

        if variables == 'all':
            variables = list(self.range_.keys()) + list(self.fixed_.keys())

        result = None
        for variable in variables:
            if variable in self.range_.keys():
                bound = torch.tensor(self.range_[variable])
                pts_variable = self._sample_range(n, mode, bound)
                pts_variable = LabelTensor(pts_variable, [variable])

            elif variable in self.fixed_.keys():
                value = self.fixed_[variable]
                pts_variable = LabelTensor(torch.ones(n, 1)*value, [variable])

            if result is None:
                result = pts_variable
            else:
                intersect = 'std' if mode == 'random' else 'cross'
                result = result.append(pts_variable, intersect)

        return result
