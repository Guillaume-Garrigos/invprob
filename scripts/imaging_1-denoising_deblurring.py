import numpy as np
from numpy import linalg as la
import matplotlib.pyplot as plt
from pylab import fft2, ifft2

import sys
sys.path.append('..')

from invprob import wavelet
from invprob import signal

#########################################
# This is for production only
import importlib
importlib.reload(wavelet)
#########################################

np.random.seed(seed=78)  # Seed for np.random
dpi = 100  # Resolution for plotting (230 for small screen, 100 for large one)
plt.ion()
data_repo = "scripts/../data/images/"

# We can blur images
im = signal.load_image(data_repo + 'comete.png')
_ = plt.figure(dpi=dpi)
_ = plt.imshow(im, cmap="gray")


def create_kernel(kernel_width, im_size):
    x = np.concatenate((
                       np.arange(0, im_size / 2), np.arange(-im_size / 2, 0)))
    [Y, X] = np.meshgrid(x, x)
    kernel = np.exp((-X**2 - Y**2) / (2 * kernel_width**2))
    kernel = kernel / sum(kernel.flatten())
    return kernel


def blur(x, h):
    return np.real(ifft2(fft2(x) * fft2(h)))

im_size = im.shape[0]
kernel_width = 3
kernel = create_kernel(kernel_width, im_size)

imb = blur(im, kernel)
_ = plt.figure(dpi=dpi)
_ = plt.imshow(imb, cmap="gray")

# We implement the wavelet transform
imw = wavelet.transform(im)
_ = plt.figure(dpi=dpi)
_ = wavelet.plot_coeff(imw)
imww = wavelet.inverse_transform(imw)
_ = plt.figure(dpi=dpi)
_ = plt.imshow(imww, cmap="gray")

