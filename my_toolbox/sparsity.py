import numpy as np
from numpy import linalg as la
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def randn(N, M, s):
    # Returns a (N,M) sparse array
    # It has 's' nonzero components, sampled from a gaussian distribution
    x = np.random.randn(N*M)
    index = np.random.permutation(N*M)
    index = index[0:N*M-s]
    x[np.ix_(index)] *= 0
    x = x.reshape((N, M))
    return x


def soft_thresholding(x, t):
    return np.sign(x)*np.maximum(0, abs(x)-t)


def norm0(x):
    return np.sum(abs(x) > 1e-15)


def stem(signal, color="C0", label=None, verbose=False, title=None):
    # plots a sparse 1D signal with stem but removes zero components
    # color and label are strings standing for the color and label
    import matplotlib.pyplot as plt
    x = signal.copy()  # Prevents modification of the signal
    x[x == 0] = np.nan
    markerline, stemlines, baseline = plt.stem(x, label=label)
    _ = plt.setp(markerline, color=color)
    _ = plt.setp(stemlines, color=color)
    if label is not None:
        plt.legend()
    if title is not None:
        plt.title()
    if verbose is False:
        return
    else:
        return markerline, stemlines, baseline


def save_stem_gif(limit, path, param_grid, file_name, title_grid):
    # Given a path of parametrised signals, plot all of them successively
    # and gather them in an animated gif which is saved as name.gif
    path_length = path.shape[1]
    fig = plt.gcf()

    def update(i):
        # animation function. This is called sequentially
        print(f"\r Creating an animated picture ... Step {i+1}/{path_length} ...", end="")
        plt.cla()
        x_reg = path[:, i, None]
        stem(limit, "C0")
        stem(x_reg, "C1")
        plt.title(title_grid[i])
        if i+1 == path_length:
            print("\n")

    anim = FuncAnimation(fig, update, np.arange(path_length), interval=200)
    anim.save(file_name + '.gif', writer='imagemagick')
    anim.event_source.stop()
    plt.close()
    del anim


def rand_plane(N):
    ''' Returns a matrix P which is composed of two orthonormal
    random vectors in R^N. Applying P to x will project x onto
    the plane spanned by these two vectors
    '''
    x = np.random.randn(N, 1)
    y = np.random.randn(N, 1)
    y = y / la.norm(y)  # normalize y
    x = x - x.T@y * y  # project x onto Vect(y)
    x = x / la.norm(x)  # normalize x
    return np.concatenate((x.T, y.T), axis=0)


def proj_plane(x, u, v):
    # Given a plane spanned by two orthonormal vectors u,v
    # Prjoects x onto it
    p = np.zeros((2, 1))
    p[0] = x.T@u
    p[1] = x.T@v
    return p[0]*u + p[1]*v, p
