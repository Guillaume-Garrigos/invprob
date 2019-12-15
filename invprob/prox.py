import numpy as np
from . import wavelet

# Imported from I3D

def L0(x, mu):
    """ Proximal operator of the L0 norm
    """
    return x * (abs(x) > mu)

def L1(x, mu):
    """ Proximal operator of the L1 norm
    """
    return np.sign(x) * np.maximum(0, abs(x) - mu)

def L1_wavelet(x, mu):
    """ Compute the proximal operator evaluated at x of the function
        mu*||Wx||_1
        where W is the orthogonal wavelet transform and mu > 0
    """
    w = wavelet.transform(x)
    w = L1(w, mu)  # the prox.L1
    p = wavelet.inverse_transform(w)
    return p

def L2_sq(x, mu):
    """ Proximal operator of the L2 squared norm: 0.5*norm(.,2)**2 """
    return x / (1+mu)

def KL(x, mu, y):
    """ Proximal operator of the Kullback-Liebler divergence: KL(y,.)
            KL(y;x) = \sum_i y_i*ln(y_i/x_i) + x_i - y_i
    """
    return 0.5 * (x - mu + np.sqrt( (x-mu)**2 + 4*mu*y) )

def simplex(x, z=1):
    # Projection sur le simplexe https://gist.github.com/mblondel/6f3b7aaad90606b98f71
    dimension = x.shape
    v = x.flatten()
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - z
    ind = np.arange(u.shape[0]) + 1
    cond = u - cssv / ind > 0
    rho = ind[cond][-1]
    theta = cssv[cond][-1] / float(rho)
    w = np.maximum(v - theta, 0)
    return w.reshape(dimension)

def L1_ball(x, s=1):
    # Projection sur la boule L1 https://gist.github.com/daien/1272551/edd95a6154106f8e28209a1c7964623ef8397246
    dimension = x.shape
    v = x.flatten()
    u = np.abs(v) # compute the vector of absolute values
    if u.sum() <= s: # check if v is already a solution
        return v.reshape(n,1)
    w = proj_simplex(u, s) # project *u* on the simplex
    w *= np.sign(v) # compute the solution to the original problem on v
    return w.reshape(dimension)

def L1_wavelet_ball(x, mu):
    """ Compute the projection of x onto the set
        ||Wx||_1 <= mu
        where W is the orthogonal wavelet transform and mu > 0
    """
    return wavelet.inverse_transform(L1_ball(wavelet.transform(x), mu))

