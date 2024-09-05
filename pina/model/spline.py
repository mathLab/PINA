"""Module for Spline model"""

import torch
import torch.nn as nn
from ..utils import check_consistency
 
class Spline(torch.nn.Module):

    def __init__(self, order=4, knots=None, control_points=None) -> None:
        """
        Spline model.

        :param int order: the order of the spline.

        """
        super().__init__()
        check_consistency(order, int)
        if order < 0:
            raise ValueError("Spline order cannot be negative.")
        if knots is None and control_points is None:
            raise ValueError("Knots and control points cannot be both None.")

        self.order = order
        self.k = order-1

        if knots is not None:
            self.knots = knots
            n = len(knots) - order
            self.control_points = torch.nn.Parameter(
                torch.zeros(n), requires_grad=True)
            
        elif control_points is not None:
            self.control_points = control_points
            n = len(control_points)
            self.knots = {
                'type': 'auto',
                'min': 0,
                'max': 1,
                'n': n}

        else:
            self.knots = knots
            self.control_points = control_points


        if self.knots.ndim != 1:
            raise ValueError("Knot vector must be one-dimensional.")
        # if n < self.k + 1:
        #     raise ValueError("Need at least %d knots for degree %d" %
        #                      (2*k + 2, k))
        # if (torch.diff(self.t) < 0).any():
        #     raise ValueError("Knots must be in a non-decreasing order.")
        # if len(torch.unique(self.t[k:n+1])) < 2:
        #     raise ValueError("Need at least two internal knots.")
        # if not torch.isfinite(self.t).all():
        #     raise ValueError("Knots should not have nans or infs.")
        # if self.c.ndim < 1:
        #     raise ValueError("Coefficients must be at least 1-dimensional.")
        # if self.c.shape[0] < n:
        #     raise ValueError("Knots, coefficients and degree are inconsistent.")

        
        
        
        
        
        
        # torch.nn.init.zeros_(self.c)

        # from pina import LabelTensor
        # import matplotlib.pyplot as plt
        # model = self
        # print(self.knots)
        # xi = torch.linspace(0, 1, 100).reshape(-1, 1)
        # yi = model(LabelTensor(xi, labels=['x']))
        # from scipy.interpolate._bsplines import BSpline
        # bsp = BSpline(t=model.t.detach(), c=model.c.detach(), k=2, extrapolate=False)
        # ui = bsp(xi.detach().flatten())

        # k = self.k

        # left = model.t[:-k]
        # right = model.t[k:]
        # cp_coord = (left + right)/2
        # print(self.c)
        # print(cp_coord)

        # plt.figure()
        # plt.plot(xi.detach(), yi.detach(), label='Spline')
        # plt.plot(xi.detach(), ui, label='Spline scipy')
        # plt.plot(cp_coord, self.c.detach(), 'o')
        # plt.show()

    @staticmethod
    def B(x, k, i, t):
        '''
        x: points to be evaluated
        k: spline degree
        i: counter for the recursive form
        t: vector of knots
        '''
        if k == 0:
            a = torch.where(torch.logical_and(t[i] <= x, x < t[i+1]), 1.0, 0.0)
            if i == len(t) - k - 1:
                 a = torch.where(x == t[-1], 1.0, a)
            a.requires_grad_(True)
            return a


        if t[i+k] == t[i]:
            c1 = torch.tensor([0.0], requires_grad=True)
        else:
            c1 = (x - t[i])/(t[i+k] - t[i]) * Spline.B(x, k-1, i, t)

        if t[i+k+1] == t[i+1]:
            c2 = torch.tensor([0.0], requires_grad=True)
        else:
            c2 = (t[i+k+1] - x)/(t[i+k+1] - t[i+1]) * Spline.B(x, k-1, i+1, t)

        return c1 + c2
    

    @property
    def control_points(self):
        return self._control_points

    @control_points.setter
    def control_points(self, value):
        if isinstance(value, dict):
            if 'n' not in value:
                raise ValueError('Invalid value for control_points')
            n = value['n']
            dim = value.get('dim', 1)
            value = torch.zeros(n, dim)

        if not isinstance(value, torch.Tensor):
            raise ValueError('Invalid value for control_points')

        self._control_points = torch.nn.Parameter(value, requires_grad=True)

    @property
    def knots(self):
        return self._knots
    
    @knots.setter
    def knots(self, value):
        if isinstance(value, dict):

            type_ = value.get('type', 'auto')
            min_ = value.get('min', 0)
            max_ = value.get('max', 1)
            n = value.get('n', 10)

            if type_ == 'uniform':
                value = torch.linspace(min_, max_, n + self.k + 1)
            elif type_ == 'auto':
                value = torch.concatenate(
                    (
                        torch.ones(self.k)*min_,
                        torch.linspace(min_, max_, n - self.k +1),
                        torch.ones(self.k)*max_,
                        # [self.max] * (k-1)
                    )
                )

        if not isinstance(value, torch.Tensor):
            raise ValueError('Invalid value for knots')

        self._knots = value

    def forward(self, x_):
        t = self.knots
        k = self.k
        c = self.control_points

        # return LabelTensor((x_**2).reshape(-1, 1), ['v'])
        # c[0] = c[0] * 0.0
        # c[-1] = c[-1] * 0.0 
        # print(x)

        # assert (n >= k+1) and (len(c) >= n)
        # for i in range(n):
        #     print(self.B(x_, k, i, t), c[i])
        #     print(self.B(x, k, i, t), c[i])
        tmp_result = torch.concatenate([
            (c[i] * Spline.B(x_, k, i, t)).reshape(
                1, x_.shape[0], -1) 
            for i in range(len(c))], axis=0
        )
        # print(tmp_result)
        return tmp_result.sum(axis=0)
