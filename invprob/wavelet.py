import numpy as np
import matplotlib.pyplot as plt
import pylab

def transform(x, Jmin=2):
    """
    Compute the wavelet transform of x
    """
    return perform_wavelet_transf(x, Jmin, +1)


def inverse_transform(w, Jmin=2):
    """
    Compute the wavelet inverse transform of w
    """
    return perform_wavelet_transf(w, Jmin, -1)


def perform_wavelet_transf(f, Jmin, dir, filter = "9-7",separable = 0, ti = 0):

    """""
    perform_wavelet_transf - peform fast lifting transform
    y = perform_wavelet_transf(x, Jmin, dir, filter = "9-7",separable = 0, ti = 0);
    Implement 1D and 2D symmetric wavelets with symmetric boundary treatements, using
    a lifting implementation.
    filter gives the coefficients of the lifting filter.
    You can use h='linear' or h='7-9' to select automatically biorthogonal
    transform with 2 and 4 vanishing moments.
    You can set ti=1 to compute a translation invariant wavelet transform.
    You can set separable=1 to compute a separable 2D wavelet
    transform.
    Copyright (c) 2008 Gabriel Peyre
    """

    #copy f
    x = np.copy(f)

    #convert Jmin to int
    Jmin = int(Jmin)

    # detect dimensionality
    d = np.ndim(x)
    # P/U/P/U/etc the last coefficient is scaling
    if filter in ["linear","5-3"]:
        h = [1/2, 1/4, np.sqrt(2)]

    elif filter in ["9-7","7-9"]:
        h = [1.586134342, -.05298011854, -.8829110762, .4435068522, 1.149604398]

    else:
        raise ValueError('Unknown filter')

    if d == 2 and separable == 1:
        ti = 0
        if ti == 1:
            wrn.warning("Separable does not works for translation invariant transform")

        # perform a separable wavelet transform
        n = np.shape(x)[0]
        if dir == 1:
            for i in range(n):
                x[:,i] = perform_wavelet_transf(x[:,i], Jmin, dir, filter, separable, ti)
            for i in range(n):
                x[i,:] = np.transpose(perform_wavelet_transf(np.transpose(x[i,:]), Jmin, dir, filter, separable, ti))
        else:
            for i in range(n):
                x[i,:] = np.transpose(perform_wavelet_transf(np.transpose(x[i,:]), Jmin, dir, filter, separable, ti))
            for i in range(n):
                x[:,i] = perform_wavelet_transf(x[:,i], Jmin, dir, filter, separable, ti)


    # number of lifting steps
    if np.ndim(x) == 1:
        n = len(x)
    else:
        n = np.shape(x)[1]
    m = (len(h)-1)//2
    Jmax = int(np.log2(n)-1)
    jlist = range(Jmax,Jmin-1,-1)

    if dir == -1:
        jlist = range(Jmin,Jmax+1,1)

    if ti == 0:
        # subsampled
        for j in jlist:
            if d == 1:
                x[:2**(j+1),:] = lifting_step(x[:2**(j+1)], h, dir)
            else:
                x[:2**(j+1),:2**(j+1)] = lifting_step(x[:2**(j+1),:2**(j+1)], h, dir)
                x[:2**(j+1),:2**(j+1)] = np.transpose(lifting_step(np.transpose(x[:2**(j+1),:2**(j+1)]), h, dir))

    else:
        # TI
        nJ = Jmax - Jmin + 1
        if dir == 1 and d == 1:
            x = np.tile(x,(nJ + 1,1,1))
        elif dir == 1 and d == 2:
            x = np.tile(x,(3*nJ + 1,1,1))
        #elif dir == 1:
        #    x = np.tile(x,(1,1,1))
        for j in jlist:
            dist = 2**(Jmax - j)

            if d == 1:
                if dir == 1:
                    x[:(j-Jmin+2),:,:] = lifting_step_ti(x[0,:,:], h, dir, dist)
                else:
                    x[0,:,:] = lifting_step_ti(x[:(j-Jmin+2),:,:], h, dir, dist)
            else:
                dj = 3*(j-Jmin)

                if dir == 1:
                    x[[0,dj+1],:,:] = lifting_step_ti(x[0,:,:], h, dir, dist)

                    x[[0,dj+2],:,:] = lifting_step_ti(np.transpose(x[0,:,:]), h, dir, dist)
                    x[0,:,:] = np.transpose(x[0,:,:])
                    x[dj+2,:,:] = np.transpose(x[dj+2,:,:])

                    x[[1+dj,3+dj],:,:] = lifting_step_ti(np.transpose(x[dj+1,:,:]), h, dir, dist)
                    x[dj+1,:,:] = np.transpose(x[dj+1,:,:])
                    x[dj+3,:,:] = np.transpose(x[dj+3,:,:])
                else:

                    x[dj+1,:,:] = np.transpose(x[dj+1,:,:])
                    x[dj+3,:,:] = np.transpose(x[dj+3,:,:])

                    x[dj+1,:,:] = np.transpose(lifting_step_ti(x[[1+dj, 3+dj],:,:], h, dir, dist))

                    x[0,:,:] = np.transpose(x[0,:,:])
                    x[dj+2,:,:] = np.transpose(x[dj+2,:,:])
                    x[0,:,:] = np.transpose(lifting_step_ti(x[[0,dj+2],:,:], h, dir, dist))

                    x[0,:,:] = lifting_step_ti(x[[0,dj+1],:,:], h, dir, dist)

        if dir == -1:
            x = x[0,:,:]

    return x

###########################################################################
###########################################################################
###########################################################################

def lifting_step(x0, h, dir):

    #copy x
    x = np.copy(x0)

    # number of lifting steps
    m = (len(h) - 1)//2

    if dir==1:
        # split
        d = x[1::2,]
        x = x[0::2,]
        for i in range(m):
            d = d - h[2*i] * (x + np.vstack((x[1:,],x[-1,])))
            x = x + h[2*i+1] * (d + np.vstack((d[0,],d[:-1,])))
        x = np.vstack((x*h[-1],d/h[-1]))

    else:
        # retrieve detail coefs
        end = len(x)
        d = x[end//2:,]*h[-1]
        x = x[:end//2,]/h[-1]
        for i in range(m,0,-1):
            x = x - h[2*i-1] * (d + np.vstack((d[0,],d[:-1,])))
            d = d + h[2*i-2] * (x + np.vstack((x[1:,],x[-1,])))
        # merge
        x1 = np.vstack((x,x))
        x1[::2,] = x
        x1[1::2,] = d
        x = x1

    return x

###########################################################################
###########################################################################
###########################################################################
def lifting_step_ti(x0, h, dir, dist):

    #copy x
    x = np.copy(x0)

    # number of lifting steps
    m = (len(h) - 1)//2
    n = np.shape(x[0])[0]

    s1 = np.arange(1, n+1) + dist
    s2 = np.arange(1, n+1) - dist

    # boundary conditions
    s1[s1 > n] = 2*n - s1[s1 > n]
    s1[s1 < 1] = 2   - s1[s1 < 1]

    s2[s2 > n] = 2*n - s2[s2 > n]
    s2[s2 < 1] = 2   - s2[s2 < 1]

    #indices in python start from 0
    s1 = s1 - 1
    s2 = s2 - 1

    if dir == 1:
        # split
        d = x
        for i in range(m):
            if np.ndim(x) == 2 :
                x = np.tile(x,(1,1,1))
            d = d - h[2*i]   * (x[:,s1,:] + x[:,s2,:])
            x = x + h[2*i+1] * (d[:,s1,:] + d[:,s2,:])

        #merge
        x = np.concatenate((x*h[-1],d/h[-1]))

    else:
        # retrieve detail coefs

        d = x[1,:,:]*h[-1]
        x = x[0,:,:]/h[-1]

        for i in range(m,0,-1):
            x = x - h[2*i-1] * (d[s1,:] + d[s2,:])
            d = d + h[2*i-2] * (x[s1,:] + x[s2,:])

        # merge
        x = (x + d)/2
    
    return x


def imageplot(f, str='', sbpt=[]):
    """
        Use nearest neighbor interpolation for the display.
    """
    if sbpt != []:
        plt.subplot(sbpt[0], sbpt[1], sbpt[2])
    imgplot = plt.imshow(f, interpolation='nearest')
    imgplot.set_cmap('gray')
    pylab.axis('off')
    if str != '':
        plt.title(str)

def plot_coeff(fW, Jmin=2):
    """
        plot_coeff - plot wavelets coefficients.

        U = plot_coeff(fW, Jmin):

        Copyright (c) 2014 Gabriel Peyre
    """
    def rescaleWav(A):
        v = abs(A).max()
        B = A.copy()
        if v > 0:
            B = .5 + .5 * A / v
        return B

    def rescale(f,a=0,b=1):
        """
            Rescale linearly the dynamic of a vector to fit within a range [a,b]
        """
        v = f.max() - f.min()
        g = (f - f.min()).copy()
        if v > 0:
            g = g / v
        return a + g*(b-a)

    ##
    n = fW.shape[1]
    Jmax = int(np.log2(n)) - 1
    U = fW.copy()
    for j in np.arange(Jmax, Jmin - 1, -1):
        U[:2 ** j:,    2 ** j:2 **
            (j + 1):] = rescaleWav(U[:2 ** j:, 2 ** j:2 ** (j + 1):])
        U[2 ** j:2 ** (j + 1):, :2 **
          j:] = rescaleWav(U[2 ** j:2 ** (j + 1):, :2 ** j:])
        U[2 ** j:2 ** (j + 1):, 2 ** j:2 ** (j + 1):] = (
            rescaleWav(U[2 ** j:2 ** (j + 1):, 2 ** j:2 ** (j + 1):]))
    # coarse scale
    U[:2 ** j:, :2 ** j:] = rescale(U[:2 ** j:, :2 ** j:])
    # plot underlying image
    imageplot(U)
    # display crosses
    for j in np.arange(Jmax, Jmin - 1, -1):
        plt.plot([0, 2 ** (j + 1)], [2 ** j, 2 ** j], 'r')
        plt.plot([2 ** j, 2 ** j], [0, 2 ** (j + 1)], 'r')
    # display box
    plt.plot([0, n], [0, 0], 'r')
    plt.plot([0, n], [n, n], 'r')
    plt.plot([0, 0], [0, n], 'r')
    plt.plot([n, n], [0, n], 'r')
    return U
