import numpy as np
def chebyshev_roots(n):
    """ Return the roots of *n* Chebyshev polynomials (between [-1, 1]) """
    coefficents = np.zeros(n+1)
    coefficents[-1] = 1
    return np.polynomial.chebyshev.chebroots(coefficents)

