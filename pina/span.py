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

        print(span_dict, self.fixed_, self.range_, 'YYYYYYYYYY')

    @property
    def variables(self):
        return list(self.fixed_.keys()) + list(self.range_.keys())

    def update(self, new_span):
        self.fixed_.update(new_span.fixed_)
        self.range_.update(new_span.range_)

    def _sample_range(self, n, mode, bounds):
        """
        """
        dim = bounds.shape[0]
        if mode in ['chebyshev', 'grid'] and dim != 1:
            raise RuntimeError('Something wrong in Span...')

        if mode == 'random':
            pts = torch.rand(size=(n, dim))
        elif mode == 'chebyshev':
            pts = chebyshev_roots(n).mul(.5).add(.5).reshape(-1, 1)
        elif mode == 'grid':
            pts = torch.linspace(0, 1, n).reshape(-1, 1)
        elif mode == 'lh' or mode == 'latin':
            from scipy.stats import qmc
            sampler = qmc.LatinHypercube(d=dim)
            pts = sampler.random(n)
            pts = torch.from_numpy(pts)

        pts *= bounds[:, 1] - bounds[:, 0]
        pts += bounds[:, 0]

        return pts

    def sample(self, n, mode='random', variables='all'):
        """TODO
        """

        def _1d_sampler(n, mode, variables):
            """ Sample independentely the variables and cross the results"""
            tmp = []
            for variable in variables:
                if variable in self.range_.keys():
                    bound = torch.tensor([self.range_[variable]])
                    pts_variable = self._sample_range(n, mode, bound)
                    pts_variable = pts_variable.as_subclass(LabelTensor)
                    pts_variable.labels = [variable]

                    tmp.append(pts_variable)

            result = tmp[0]
            for i in tmp[1:]:
                result = result.append(i, mode='cross')

            for variable in variables:
                if variable in self.fixed_.keys():
                    value = self.fixed_[variable]
                    pts_variable = torch.tensor([[value]]).repeat(
                        result.shape[0], 1)
                    pts_variable = pts_variable.as_subclass(LabelTensor)
                    pts_variable.labels = [variable]

                    result = result.append(pts_variable, mode='std')

            return result

        def _Nd_sampler(n, mode, variables):
            """ Sample ll the variables together """
            bounds = torch.tensor(
                [v for k, v in self.range_.items() if k in variables]
            )
            result = self._sample_range(n, mode, bounds)
            result = result.as_subclass(LabelTensor)
            result.labels = list(self.range_.keys())

            for variable in variables:
                if variable in self.fixed_.keys():
                    value = self.fixed_[variable]
                    pts_variable = torch.tensor([[value]]).repeat(
                        result.shape[0], 1)
                    pts_variable = pts_variable.as_subclass(LabelTensor)
                    pts_variable.labels = [variable]

                    result = result.append(pts_variable, mode='std')
            return result

        if variables == 'all':
            variables = list(self.range_.keys()) + list(self.fixed_.keys())

        if mode in ['grid', 'chebyshev']:
            return _1d_sampler(n, mode, variables)
        elif mode in ['random', 'lhs']:
            return _Nd_sampler(n, mode, variables)
        else:
            raise ValueError(f'mode={mode} is not valid.')
