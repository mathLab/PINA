from mpmath import chebyt, chop, taylor
import numpy as np
def chebyshev_roots(n):
    """ Return the roots of *n* Chebyshev polynomials (between [-1, 1]) """
    return np.roots(chop(taylor(lambda x: chebyt(n, x), 0, n))[::-1])

