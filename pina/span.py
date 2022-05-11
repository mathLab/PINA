import numpy as np
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

    def sample(self, n, mode='random', variables='all'):

        if variables=='all':
            spatial_range_ = list(self.range_.keys())
            spatial_fixed_ = list(self.fixed_.keys())
            bounds = np.array(list(self.range_.values()))
            fixed = np.array(list(self.fixed_.values()))
        else:
            bounds = []
            spatial_range_ = []
            spatial_fixed_ = []
            fixed = []
            for variable in variables:
                if variable in self.range_.keys():
                    spatial_range_.append(variable)
                    bounds.append(list(self.range_[variable]))
                elif variable in self.fixed_.keys():
                    spatial_fixed_.append(variable)
                    fixed.append(int(self.fixed_[variable]))
            fixed = torch.Tensor(fixed)
            bounds = np.array(bounds)
        if mode == 'random':
            pts = np.random.uniform(size=(n, bounds.shape[0]))
        elif mode == 'chebyshev':
            pts = np.array([
                chebyshev_roots(n) * .5 + .5
                for _ in range(bounds.shape[0])])
            grids = np.meshgrid(*pts)
            pts = np.hstack([grid.reshape(-1, 1) for grid in grids])
        elif mode == 'grid':
            pts = np.array([
                np.linspace(0, 1, n)
                for _ in range(bounds.shape[0])])
            grids = np.meshgrid(*pts)
            pts = np.hstack([grid.reshape(-1, 1) for grid in grids])
        elif mode == 'lh' or mode == 'latin':
            from scipy.stats import qmc
            sampler = qmc.LatinHypercube(d=bounds.shape[0])
            pts = sampler.random(n)
        # Scale pts
        pts *= bounds[:, 1] - bounds[:, 0]
        pts += bounds[:, 0]

        pts = pts.astype(np.float32)
        pts = torch.from_numpy(pts)

        pts_range_ = LabelTensor(pts, spatial_range_)

        if not len(spatial_fixed_)==0:
            pts_fixed_ = torch.ones(pts.shape[0], len(spatial_fixed_),
                                dtype=pts.dtype) * fixed
            pts_fixed_ = pts_fixed_.float()
            pts_fixed_ = LabelTensor(pts_fixed_, spatial_fixed_)
            pts_range_ = pts_range_.append(pts_fixed_)

        return pts_range_


    def meshgrid(self, n):
        pts = np.array([
            np.linspace(0, 1, n)
            for _ in range(self.bound.shape[0])])

        pts *= self.bound[:, 1] - self.bound[:, 0]
        pts += self.bound[:, 0]

        return np.meshgrid(*pts)
