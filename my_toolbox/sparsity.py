import numpy as np
from numpy import linalg as la

def randn(N,M,s): 
    # Returns a (N,M) sparse array
    # The array has 's' nonzero components, sampled from a gaussian distribution 
    x = np.random.randn(N*M)
    index = np.random.permutation(N*M)
    index = index[0:N*M-s]
    x[np.ix_(index)]*=0
    x = x.reshape((N,M))
    return x

def soft_thresholding(x,t):
    return np.sign(x)*np.maximum(0 , abs(x)-t)

def norm0(x):
    return np.sum(abs(x)>1e-15)

def stem(signal,color="C0",label=None):
    # plots a sparse 1D signal with stem but removes zero components
    # color and label are strings standing for the color and label
    import matplotlib.pyplot as plt
    x = signal.copy() # Prevents modification of the signal
    x[ x==0 ] = np.nan
    (markerline, stemlines, _ ) = plt.stem(x,label=label)
    _ = plt.setp(markerline, color=color)
    _ = plt.setp(stemlines, color=color)
    if label != None:
        plt.legend()
    return 

def rand_plane(N):
    # Returns two orthonormal random vectors in R^N
    x = np.random.randn(N,1)
    y = np.random.randn(N,1)
    y = y / la.norm(y) # normalize y
    x = x - x.T@y * y # project x onto Vect(y)
    x = x / la.norm(x) # normalize x
    return x, y

def proj_plane(x,u,v):
    # Given a plane spanned by two orthonormal vectors u,v
    # Prjoects x onto it
    p = np.zeros((2,1))
    p[0] = x.T@u
    p[1] = x.T@v
    return p[0]*u + p[1]*v, p