import numpy as np
from numpy import linalg as la
import matplotlib.pyplot as plt
from pylab import fft2, ifft2
from scipy.fftpack import dct
from scipy.fftpack import idct
from skimage.io import imread
from skimage.util import random_noise
from skimage.util.dtype import img_as_float

import sys
sys.path.append('..')

from invprob import wavelet
# from invprob import signal

#########################################
# This is for production only
import importlib
importlib.reload(wavelet)
#########################################
seed = 78  # Seed for random events
np.random.seed(seed=seed)
dpi = 230  # Resolution for plotting (230 for small screen, 100 for large one)
plt.ion()
data_repo = "scripts/../data/images/"

# We can blur images
# comete = signal.load_image(data_repo + 'comete.png')
im = img_as_float(imread(data_repo + 'comete.png'))
_ = plt.figure(dpi=dpi)
_ = plt.imshow(im, cmap="gray", interpolation="none")


def create_kernel(kernel_size, kernel_std):
    x = np.concatenate((
                       np.arange(0, kernel_size / 2), np.arange(-kernel_size / 2, 0)))
    [Y, X] = np.meshgrid(x, x)
    kernel = np.exp((-X**2 - Y**2) / (2 * kernel_std**2))
    kernel = kernel / sum(kernel.flatten())
    return kernel


def blur(x, h):
    return np.real(ifft2(fft2(x) * fft2(h)))


kernel_size =512
kernel_std = 2
kernel = create_kernel(kernel_size, kernel_std)
imb = blur(im, kernel)
_ = plt.figure(dpi=dpi)
_ = plt.imshow(imb, cmap="gray")

# Replace fft with dct
def dct2(image):
    return dct(dct(image.T, norm='ortho').T, norm='ortho')

def idct2(coefficient):
    return idct(idct(coefficient.T, norm='ortho').T, norm='ortho')

# inverse works well? yes
_ = plt.figure(dpi=dpi)
_ = plt.imshow(idct2(dct2(im)), cmap="gray")

# convolution works well? no
imb_dct = np.real(idct2(dct2(im) * dct2(kernel)))
_ = plt.figure(dpi=dpi)
_ = plt.imshow(imb_dct, cmap="gray")


# Tentative import matlab code WORKS?

def fast_create_kernel(kernel_size, kernel_std, im_shape):
    kernel = create_kernel(kernel_size, kernel_std)
    padded_kernel = np.zeros(im_shape)
    padded_kernel[0:kernel.shape[0], 0:kernel.shape[1]] = kernel
    e = np.zeros(im_shape)
    e[0,0] = 1
    i = int(np.floor(kernel.shape[0]/2)+1)
    j = int(np.floor(kernel.shape[1]/2)+1)
    center = np.array([i, j])
    m = im_shape[0]
    n = im_shape[1]
    k = min(i-1,m-i,j-1,n-j)

    PP = padded_kernel[i-k-1:i+k, j-k-1:j+k] 
    Z1 = np.diag(np.ones((k+1)),k)
    Z2 = np.diag(np.ones((k)),k+1)
    PP = Z1@PP@Z1.T + Z1@PP@Z2.T + Z2@PP@Z1.T + Z2@PP@Z2.T

    Ps = np.zeros(im_shape)
    Ps[0:2*k+1, 0:2*k+1] = PP
    kernel_fourier = dct2(Ps)/dct2(e)
    return kernel_fourier

def fast_blur(x, h):
    return np.real(idct2(dct2(x) * h))

kernel_size = 9
kernel_std = 10
kernel_fourier = fast_create_kernel(kernel_size, kernel_std, im.shape)
imb_dct = fast_blur(im, kernel_fourier)
_ = plt.figure(dpi=dpi)
_ = plt.imshow(imb_dct, cmap="gray")

# ok it works. It is a symmetric operator? YESSSSS
a = np.random.randn(512,512)
b = np.random.randn(512,512)
np.sum(fast_blur(a, kernel_fourier) * b) - np.sum(fast_blur(b, kernel_fourier) * a)


# We implement the wavelet transform
imw = wavelet.transform(im)
_ = plt.figure(dpi=dpi)
_ = wavelet.plot_coeff(imw)
imww = wavelet.inverse_transform(imw)
_ = plt.figure(dpi=dpi)
_ = plt.imshow(imww, cmap="gray")

# We add some noise
im_noise = random_noise(im, mode="gaussian", seed=seed, var=0.1)
_ = plt.figure(dpi=dpi)
_ = plt.imshow(im_noise, cmap="gray")

im_noise = random_noise(im, mode="s&p", seed=seed, amount=0.35)
_ = plt.figure(dpi=dpi)
_ = plt.imshow(im_noise, cmap="gray")

im_noise = random_noise(im, mode="poisson", seed=seed)
_ = plt.figure(dpi=dpi)
_ = plt.imshow(im_noise, cmap="gray")

# Now we solve an inverse problem
im = img_as_float(imread(data_repo + 'comete.png'))
kernel_size = 9
kernel_std = 10
kernel_fourier = fast_create_kernel(kernel_size, kernel_std, im.shape)
y = fast_blur(im, kernel_fourier)

x = np.zeros(im.shape)
nb_iter = 100
L = np.max(np.abs(kernel_fourier))
stepsize = 1/L

for k in range(nb_iter):
    grad = fast_blur(fast_blur(x, kernel_fourier) - y, kernel_fourier)
    x = x - stepsize * grad

_ = plt.figure(dpi=dpi)
plt.subplot(1,3,1)
_ = plt.imshow(im, cmap="gray")
plt.subplot(1,3,2)
_ = plt.imshow(y, cmap="gray")
plt.subplot(1,3,3)
_ = plt.imshow(x, cmap="gray")


