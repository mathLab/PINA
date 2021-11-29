import torch
import numpy as np
from .chebyshev import chebyshev_roots

class Segment():
    def __init__(self, p1, p2):
        self.p1 = p1
        self.p2 = p2

    def discretize(self, n, mode='random'):
        pts = []

        if mode == 'random':
            iterator = np.random.uniform(0, 1, n)
        elif mode == 'grid':
            iterator = np.linspace(0, 1, n)
        elif mode == 'chebyshev':
            iterator = chebyshev_roots(n) * .5 + .5

        for k in iterator:
            x = self.p1[0] + k*(self.p2[0]-self.p1[0])
            y = self.p1[1] + k*(self.p2[1]-self.p1[1])
            pts.append((x, y))

        pts = np.array(pts)
        return pts

