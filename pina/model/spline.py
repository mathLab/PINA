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
        self.k = order - 1

        if knots is not None and control_points is not None:
            self.knots = knots
            self.control_points = control_points

        elif knots is not None:
            print('Warning: control points will be initialized automatically.')
            print('         experimental feature')

            self.knots = knots
            n = len(knots) - order
            self.control_points = torch.nn.Parameter(
                torch.zeros(n), requires_grad=True)
            
        elif control_points is not None:
            print('Warning: knots will be initialized automatically.')
            print('         experimental feature')
            
            self.control_points = control_points

            n = len(self.control_points)-1
            self.knots = {
                'type': 'auto',
                'min': 0,
                'max': 1,
                'n': n+2+self.order}

        else:
            raise RuntimeError


        if self.knots.ndim != 1:
            raise ValueError("Knot vector must be one-dimensional.")

    def basis(self, x, k, i, t):
        '''
        x: points to be evaluated
        k: spline degree
        i: counter for the recursive form
        t: vector of knots
        '''
        if k == 0:
            a = torch.where(torch.logical_and(t[i] <= x, x < t[i+1]), 1.0, 0.0)
            if i == len(t) - self.order - 1:
                 a = torch.where(x == t[-1], 1.0, a)
            a.requires_grad_(True)
            return a


        if t[i+k] == t[i]:
            c1 = torch.tensor([0.0]*len(x), requires_grad=True)
        else:
            c1 = (x - t[i])/(t[i+k] - t[i]) * self.basis(x, k-1, i, t)

        if t[i+k+1] == t[i+1]:
            c2 = torch.tensor([0.0]*len(x), requires_grad=True)
        else:
            c2 = (t[i+k+1] - x)/(t[i+k+1] - t[i+1]) * self.basis(x, k-1, i+1, t)

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
                initial_knots = torch.ones(self.order+1)*min_
                final_knots = torch.ones(self.order+1)*max_

                if n < self.order + 1:
                    value = torch.concatenate((initial_knots, final_knots))
                elif n - 2*self.order + 1 == 1:
                    value = torch.Tensor([(max_ + min_)/2])
                else:
                    value = torch.linspace(min_, max_, n - 2*self.order - 1)

                value = torch.concatenate(
                    (
                        initial_knots, value, final_knots
                    )
                )

        if not isinstance(value, torch.Tensor):
            raise ValueError('Invalid value for knots')

        self._knots = value

    def forward(self, x_):
        t = self.knots
        k = self.k
        c = self.control_points

        basis = map(lambda i: self.basis(x_, k, i, t)[:, None], range(len(c)))
        y = (torch.cat(list(basis), dim=1) * c).sum(axis=1)

        return y