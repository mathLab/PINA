import numpy as np
from .chebyshev import chebyshev_roots

class Cube():
    def __init__(self, bound):
        self.bound = np.asarray(bound)

    def discretize(self, n, mode='random'):

        if mode == 'random':
            pts = np.random.uniform(size=(n, self.bound.shape[0]))
        elif mode == 'chebyshev':
            pts = np.array([chebyshev_roots(n) *.5 + .5 for _ in range(self.bound.shape[0])])
            grids = np.meshgrid(*pts)
            pts = np.hstack([grid.reshape(-1, 1) for grid in grids])
        elif mode == 'grid':
            pts = np.array([np.linspace(0, 1, n)  for _ in range(self.bound.shape[0])])
            grids = np.meshgrid(*pts)
            pts = np.hstack([grid.reshape(-1, 1) for grid in grids])
        elif mode == 'lh' or mode == 'latin':
            from scipy.stats import qmc
            sampler = qmc.LatinHypercube(d=self.bound.shape[0])
            pts = sampler.random(n)

        # Scale pts
        pts *= self.bound[:, 1] - self.bound[:, 0]
        pts += self.bound[:, 0]

        return pts
