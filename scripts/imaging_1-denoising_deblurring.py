import numpy as np
from numpy import linalg as la
import matplotlib.pyplot as plt
import imageio
import signal
from pylab import fft2, ifft2

np.random.seed(seed=78)  # Seed for np.random
dpi = 100  # Resolution for plotting (230 for small screen, 100 for large one)
plt.ion()
data_repo = "scripts/../data/images/"

# We can blur images
im = imageio.imread(data_repo + 'comete.png')
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


kernel_width = 3
kernel = create_kernel(kernel_width, 512)

imb = blur(im, kernel)
_ = plt.figure(dpi=dpi)
_ = plt.imshow(imb, cmap="gray")

# We implement the wavelet transform

