import numpy as np
import matplotlib.pyplot as plt
import pylab

def transform(x, Jmin=2):
    """
    Compute the wavelet transform of x
    """
    h = compute_wavelet_filter("Daubechies",10)
    return perform_wavortho_transf(x, Jmin, +1, h)


def inverse_transform(w, Jmin=2):
    """
    Compute the wavelet inverse transform of w
    """
    h = compute_wavelet_filter("Daubechies",10)
    return perform_wavortho_transf(w, Jmin, -1, h)


def compute_wavelet_filter(type,par):
    """
        compute_wavelet_filter - Generate Orthonormal QMF Filter for Wavelet Transform
        
        
           [h,g] = compute_wavelet_filter(Type,Par)
        
         Inputs
           Type   string, 'Haar', 'Beylkin', 'Coiflet', 'Daubechies',
                  'Symmlet', 'Vaidyanathan','Battle'
           Par    integer, it is a parameter related to the support and vanishing
                  moments of the wavelets, explained below for each wavelet.
        
        Outputs
          h   low pass quadrature mirror filter
          g   high pass
        
         Description
           The Haar filter (which could be considered a Daubechies-2) was the
           first wavelet, though not called as such, and is discontinuous.
    
           The Beylkin filter places roots for the frequency response function
           close to the Nyquist frequency on the real axis.
         
           The Coiflet filters are designed to give both the mother and father
           wavelets 2*Par vanishing moments; here Par may be one of 1,2,3,4 or 5.
         
           The Daubechies filters are minimal phase filters that generate wavelets
           which have a minimal support for a given number of vanishing moments.
           They are indexed by their length, Par, which may be one of
           2,4,6,8,10,12,14,16,18 or 20. The number of vanishing moments is par/2.
         
           Symmlets are also wavelets within a minimum size support for a given
           number of vanishing moments, but they are as symmetrical as possible,
           as opposed to the Daubechies filters which are highly asymmetrical.
           They are indexed by Par, which specifies the number of vanishing
           moments and is equal to half the size of the support. It ranges
           from 4 to 10.
         
           The Vaidyanathan filter gives an exact reconstruction, but does not
           satisfy any moment condition.  The filter has been optimized for
           speech coding.
         
           The Battle-Lemarie filter generate spline orthogonal wavelet basis.
           The parameter Par gives the degree of the spline. The number of
           vanishing moments is Par+1.
         
        See Also
           FWT_PO, IWT_PO, FWT2_PO, IWT2_PO, WPAnalysis
    
        References
            The books by Daubechies and Wickerhauser.
            
        Warning : only Daubechies implemented for the moment !
    """
     
    if type == 'Daubechies':
        
        if par == 1:
            f = [1,1]/np.sqrt(2)

        if par == 4:
            f = [.482962913145,.836516303738,
                .224143868042,-.129409522551]
        
        if par == 6:
            f = [.332670552950,.806891509311,
            .459877502118,-.135011020010,
                -.085441273882,.035226291882]
        
        if par == 8:
            f = [ .230377813309,.714846570553,
                .630880767930,-.027983769417,
                -.187034811719,.030841381836,
                .032883011667,-.010597401785]
        
        if par == 10:
            f = [.160102397974,.603829269797,.724308528438,
                .138428145901,-.242294887066,-.032244869585,
                .077571493840,-.006241490213,-.012580751999,
                .003335725285]
        
        if par == 12:
            f = [.111540743350,.494623890398,.751133908021,
                .315250351709,-.226264693965,-.129766867567,
                .097501605587,.027522865530,-.031582039317,
                .000553842201,.004777257511,-.001077301085]
        
        if par == 14:
            f = [.077852054085,.396539319482,.729132090846,
                .469782287405,-.143906003929,-.224036184994,
                .071309219267,.080612609151,-.038029936935,
                -.016574541631,.012550998556,.000429577973,
                -.001801640704,.000353713800]
        
        if par == 16:
            f = [.054415842243,.312871590914,.675630736297,
                .585354683654,-.015829105256,-.284015542962,
                .000472484574,.128747426620,-.017369301002,
                -.044088253931,.013981027917,.008746094047,
                -.004870352993,-.000391740373,.000675449406,
                -.000117476784]
        
        if par == 18:
            f = [.038077947364,.243834674613,.604823123690,
                .657288078051,.133197385825,-.293273783279,
                -.096840783223,.148540749338,.030725681479,
                -.067632829061,.000250947115,.022361662124,
                -.004723204758,-.004281503682,.001847646883,
                .000230385764,-.000251963189,.000039347320]
        
        if par == 20:
            f = [.026670057901,.188176800078,.527201188932,
                .688459039454,.281172343661,-.249846424327,
                -.195946274377,.127369340336,.093057364604,
                -.071394147166,-.029457536822,.033212674059,
                .003606553567,-.010733175483,.001395351747,
                .001992405295,-.000685856695,-.000116466855,
                .000093588670,-.000013264203]
    
    else:
        raise ValueError("Wrong arguments, see comments for acceptable values")
        
    f = list(f/np.linalg.norm(f))
    
    if len(f)%2 == 0:
        f = [0] + f
    return f


def perform_wavortho_transf(f, Jmin, dir, h):
    """
        perform_wavortho_transf - compute orthogonal wavelet transform
        fw = perform_wavortho_transf(f,Jmin,dir,options);
        You can give the filter in options.h.
        Works in 2D only.
        Copyright (c) 2014 Gabriel Peyre
    """

    n = f.shape[1]
    Jmax = int(np.log2(n)) - 1
    # compute g filter
    u = np.power(-np.ones(len(h) - 1), range(1, len(h)))
    # alternate +1/-1
    g = np.concatenate(([0], h[-1:0:-1] * u))

    if dir == 1:
        ### FORWARD ###
        fW = f.copy()
        for j in np.arange(Jmax, Jmin - 1, -1):
            A = fW[:2 ** (j + 1):, :2 ** (j + 1):]
            for d in np.arange(1, 3):
                Coarse = subsampling(cconv(A, h, d), d)
                Detail = subsampling(cconv(A, g, d), d)
                A = np.concatenate((Coarse, Detail), axis=d - 1)
            fW[:2 ** (j + 1):, :2 ** (j + 1):] = A
        return fW
    else:
        ### BACKWARD ###
        fW = f.copy()
        f1 = fW.copy()
        for j in np.arange(Jmin, Jmax + 1):
            A = f1[:2 ** (j + 1):, :2 ** (j + 1):]
            for d in np.arange(1, 3):
                if d == 1:
                    Coarse = A[:2**j:, :]
                    Detail = A[2**j: 2**(j + 1):, :]
                else:
                    Coarse = A[:, :2 ** j:]
                    Detail = A[:, 2 ** j:2 ** (j + 1):]
                Coarse = cconv(upsampling(Coarse, d), reverse(h), d)
                Detail = cconv(upsampling(Detail, d), reverse(g), d)
                A = Coarse + Detail
            f1[:2 ** (j + 1):, :2 ** (j + 1):] = A
        return f1



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



def subsampling(x, d):
    # subsampling along dimension d by factor p=2
    p = 2
    if d == 1:
        y = x[::p, :]
    elif d == 2:
        y = x[:, ::p]
    else:
        raise Exception('Not implemented')
    return y

def upsampling(x, d):
    """
        up-sampling along dimension d by factor p=2
    """
    p = 2
    s = x.shape
    if d == 1:
        y = np.zeros((p * s[0], s[1]))
        y[::p, :] = x
    elif d == 2:
        y = np.zeros((s[0], p * s[1]))
        y[:, ::p] = x
    else:
        raise Exception('Not implemented')
    return y



def cconv(x, h, d):
    """
        Circular convolution along dimension d.
        h should be small and with odd size
    """
    if d == 2:
        # apply to transposed matrix
        return np.transpose(cconv(np.transpose(x), h, 1))
    y = np.zeros(x.shape)
    p = len(h)
    pc = int(round( float((p - 1) / 2 )))
    for i in range(0, p):
        y = y + h[i] * circshift1d(x, i - pc)
    return y


def reverse(x):
    """
        Reverse a vector.
    """
    return x[::-1]


def circshift(x, p):
    """
        Circular shift of an array.
    """
    y = x.copy()
    y = np.concatenate((y[p[0]::, :], y[:p[0]:, :]), axis=0)
    if x.shape[1] > 0 and len(p) > 1:
        y = np.concatenate((y[:, p[0]::], y[:, :p[0]:]), axis=1)
    return y

def circshift1d(x, k):
    """
        Circularly shift a 1D vector
    """
    return np.roll(x, -k, axis=0)


def plot_wavelet(fW, Jmin=0):
    """
        plot_wavelet - plot wavelets coefficients.
        U = plot_wavelet(fW, Jmin):
        Copyright (c) 2014 Gabriel Peyre
    """
    def rescaleWav(A):
        v = abs(A).max()
        B = A.copy()
        if v > 0:
            B = .5 + .5 * A / v
        return B
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

def rescale(f,a=0,b=1):
    """
        Rescale linearly the dynamic of a vector to fit within a range [a,b]
    """
    v = f.max() - f.min()
    g = (f - f.min()).copy()
    if v > 0:
        g = g / v
    return a + g*(b-a)
